import json
from pathlib import Path

import numpy as np
import pytest

from mini_sim import Config, run
from mini_sim.cli import main as simulate
from mini_sim.workloads import WORKLOADS, extract_tile, generate_benchmark, sparsity_2_4


def test_preserves_user_workload_dimensions():
    assert WORKLOADS == {"resnet50_conv1": (12544, 64, 160),
                         "llama3_q_proj": (32, 1024, 4096),
                         "llama3_gate_proj": (256, 1024, 4096)}


@pytest.mark.parametrize("axis", ["K", "N"])
def test_keeps_largest_magnitudes_in_correct_axis(axis):
    w = np.array([[1, -8, 3, 4], [-9, 2, 1, -6], [4, 5, -7, 2], [3, -4, 2, 1]], dtype=np.float16)
    sparse, mask = sparsity_2_4(w, axis, return_mask=True)
    group_mask = mask.T if axis == "K" else mask
    groups = w.T if axis == "K" else w
    for i in range(4):
        expected = set(sorted(range(4), key=lambda j: (-abs(float(groups[i, j])), j))[:2])
        assert set(np.flatnonzero(group_mask[i])) == expected
    np.testing.assert_array_equal(sparse, np.where(mask, w, 0))
    assert sparse.dtype == w.dtype


def test_ties_and_retained_zeros_are_unambiguous():
    _, mask = sparsity_2_4(np.zeros((4, 4), dtype=np.float32), return_mask=True)
    assert mask[:2].all() and not mask[2:].any()


def test_tile_offsets_edge_padding_and_partial_product():
    a = np.arange(6 * 10, dtype=np.float32).reshape(6, 10)
    w = np.arange(10 * 7, dtype=np.float32).reshape(10, 7)
    at, wt, meta = extract_tile(a, w, size=4, rows=4, row_start=4, k_start=8, n_start=4)
    assert at.shape == (2, 4) and wt.shape == (4, 4)
    np.testing.assert_array_equal(at[:, :2], a[4:, 8:])
    np.testing.assert_array_equal(wt[:2, :3], w[8:, 4:])
    assert not at[:, 2:].any() and not wt[2:].any() and not wt[:, 3:].any()
    np.testing.assert_array_equal((at @ wt)[:, :3], a[4:, 8:] @ w[8:, 4:])
    assert meta["valid_k"] == 2 and meta["valid_n"] == 3


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
def test_generator_outputs_run_directly_in_original_hardware(tmp_path, dtype, capsys):
    definitions = {"tiny": (5, 8, 12)}
    manifest = generate_benchmark(tmp_path, workloads=definitions, tile_size=4,
        tile_rows=3, dtype=dtype, k_start=4, n_start=4, full_reference=True)
    entry = manifest["workloads"][0]
    assert entry["shape_MNK"] == [5, 8, 12]
    cycles = []
    for variant in ("dense", "sparse"):
        full = tmp_path / entry["files"][variant]["full"]
        tile = tmp_path / entry["files"][variant]["tile"]
        with np.load(full, allow_pickle=False) as data:
            assert data["A"].shape == (5, 12) and data["W"].shape == (12, 8)
            if variant == "sparse":
                assert (data["keep_mask"].reshape(3, 4, 8).sum(axis=1) == 2).all()
        report = tmp_path / f"{variant}.json"
        assert simulate(["--input", str(tile), "--json", str(report)]) == 0
        r = json.loads(report.read_text())
        assert r["config"]["dtype"] == dtype and r["config"]["rows"] == 3 and r["config"]["size"] == 4
        assert r["input_metadata"]["scope"] == "single_K_slice_contribution"
        cycles.append(r["total_cycles"])
        with np.load(tile, allow_pickle=False) as data:
            result = run(Config(size=4, rows=3, dtype=dtype), data["A"], data["W"])
            np.testing.assert_array_equal(result.output, data["C_reference"])
    assert cycles[0] == cycles[1]
    assert (tmp_path / entry["numpy_reference"]).is_file()
    capsys.readouterr()


def test_filtering_preserves_seeded_data(tmp_path):
    definitions = {"first": (3, 4, 4), "second": (3, 4, 4)}
    all_dir, selected = tmp_path / "all", tmp_path / "selected"
    generate_benchmark(all_dir, workloads=definitions, tile_size=4)
    generate_benchmark(selected, workloads=definitions, names=["second"], tile_size=4)
    with np.load(all_dir / "full/second_dense.npz") as a, np.load(selected / "full/second_dense.npz") as b:
        np.testing.assert_array_equal(a["A"], b["A"])
        np.testing.assert_array_equal(a["W"], b["W"])


@pytest.mark.parametrize("kwargs", [dict(k_start=1), dict(row_start=-1), dict(n_start=100),
    dict(sparsity_axis="X"), dict(tile_size=2), dict(names=["missing"]), dict(names=[]), dict(seed=-1)])
def test_invalid_generation_request_writes_nothing(tmp_path, kwargs):
    with pytest.raises(ValueError):
        generate_benchmark(tmp_path / "not_created", workloads={"tiny": (3, 4, 4)}, **kwargs)
    assert not (tmp_path / "not_created").exists()


def test_full_workload_is_rejected_with_tile_guidance(tmp_path, capsys):
    path = tmp_path / "full.npz"
    np.savez(path, A=np.ones((3, 8)), W=np.ones((8, 4)))
    with pytest.raises(SystemExit) as exc:
        simulate(["--input", str(path)])
    assert exc.value.code == 2
    assert "tiles/" in capsys.readouterr().err


def test_explicit_dtype_overrides_fixture_metadata(tmp_path, capsys):
    path = tmp_path / "tile.npz"
    np.savez(path, A=np.ones((1, 4)), W=np.eye(4), metadata_json=np.asarray('{"dtype":"fp16"}'))
    assert simulate(["--input", str(path), "--dtype", "bf16"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["config"]["dtype"] == "bf16"
