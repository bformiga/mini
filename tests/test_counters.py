import json
import numpy as np
import pytest

from mini_sim.cli import main
from mini_sim.compare import compare_dense_vs_sparse
from mini_sim.counters import dense_tile_counters, sparse_tile_counters, sum_counters
from mini_sim.experiment import ExperimentConfig, run_experiment, run_rectangular
from mini_sim.workloads import sparsity_2_4

# K = 4, N = 2: each column is one group of 4. 2:4 keeps 2 per column: [1, 0, 3, 0] -> 1, 3 and [0, 2, 0, 0] -> 0, 2.
# So 3 of 8 weights are nonzero, and one kept slot holds a zero.
W_SMALL = np.array([[1, 0], [0, 2], [3, 0], [0, 0]], dtype=np.float32)

def test_sparse_tile_counters_match_hand_count():
    c = sparse_tile_counters(2, np.array([[1, 0], [3, 2]], dtype=np.float32))
    # M * N * K
    assert c["dense_macs"] == (2 * 2 * 4) == 16
    # M * N * (K / 4 * 2)
    assert c["issued_macs"] == (2 * 2 * 2) == 8
    assert c["skipped_macs"] == 8
    assert c["useful_macs"] == (2 * 3) == 6
    assert c["zero_slot_macs"] == (2 * 1) == 2
    assert c["effective_weight_density"] == (3 / 8)


def test_dense_vs_sparse_counters_on_same_weights():
    d = dense_tile_counters(2, W_SMALL)
    s = sparse_tile_counters(2, np.array([[1, 0], [3, 2]], dtype=np.float32))
    assert (d["issued_macs"] == d["dense_macs"] == 16) and (d["skipped_macs"] == 0)
    assert (d["useful_macs"] == s["useful_macs"] == 6)
    assert d["zero_slot_macs"] == (16 - 6)
    assert (d["effective_weight_density"] == s["effective_weight_density"] == (3 / 8))


def test_report_counters_match_the_array_run_on_both_modes():
    a = np.ones((2, 4), dtype=np.float32)
    sparse = run_experiment(ExperimentConfig(mode="sparse", size=2, rows=2, dtype="fp32"), a, W_SMALL).report
    dense = run_experiment(ExperimentConfig(size=2, rows=2, dtype="fp32"), np.ones((2, 2), dtype=np.float32), W_SMALL[:2]).report
    assert (sparse["correctness_passed"] and dense["correctness_passed"])
    assert (sparse["dense_macs"], sparse["issued_macs"], sparse["skipped_macs"]) == (16, 8, 8)
    assert (sparse["useful_macs"], sparse["zero_slot_macs"]) == (6, 2)
    assert sparse["issued_macs"] == sparse["executed_multiplications"]
    assert (dense["issued_macs"] == dense["useful_multiplications"]) and (dense["skipped_macs"] == 0)


def test_dense_run_with_generated_weights_reports_counters():
    r = run_experiment(ExperimentConfig(size=4, rows=3, dtype="fp32", weight_pattern="2of4")).report
    assert (r["useful_macs"] == (3 * 8)) and (r["effective_weight_density"] == 0.5)
    assert r["issued_macs"] == (r["useful_macs"] + r["zero_slot_macs"])


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_edge_padding_is_never_useful_work(mode):
    # K pads to 16 or 24, N pads to 8
    m, k, n = 5, 12, 7
    a = np.arange(m * k, dtype=np.float32).reshape(m, k) % 4
    w = sparsity_2_4(np.arange(1, k * n + 1, dtype=np.float32).reshape(k, n) % 5 + 1, "K")
    r = run_rectangular(ExperimentConfig(mode=mode, size=4, dtype="fp32"), a, w, row_tile=3).report
    assert r["correctness_passed"]
    assert r["useful_macs"] == (m * int(np.count_nonzero(w)))
    assert r["issued_macs"] == r["executed_multiplications"] == (r["useful_macs"] + r["zero_slot_macs"])
    # sparse: all padding
    # dense: padding plus the pruned zeros
    pruned_zero_macs = m * int(np.count_nonzero(w == 0))
    if (mode == "dense"):
        padding_macs = r["zero_slot_macs"] - pruned_zero_macs
    else:
        padding_macs = r["zero_slot_macs"]
    assert padding_macs > 0
    assert r["effective_weight_density"] < (np.count_nonzero(w) / w.size)


def test_sum_counters_recomputes_density_from_totals():
    a = dense_tile_counters(1, np.ones((4, 2), dtype=np.float32))
    b = dense_tile_counters(1, np.zeros((4, 6), dtype=np.float32))
    total = sum_counters([a, b])
    assert total["effective_weight_density"] == (8 / 32)
    assert total["dense_macs"] == (a["dense_macs"] + b["dense_macs"])


def make_int_workload(m, k, n):
    """
    Make a workload with whole numbers, so fp32 sums are exact and tests can use ==
    """
    a = np.arange(m * k, dtype=np.float32).reshape(m, k) % 4
    w = (np.arange(k * n, dtype=np.float32).reshape(k, n) * 7) % 5 - 2
    return a, w


@pytest.mark.parametrize("load_path", ["direct", "atalla"])
def test_compare_runs_both_arrays_on_the_same_pruned_workload(load_path):
    a, w = make_int_workload(5, 16, 8)
    cfg = ExperimentConfig(size=4, dtype="fp32", load_path=load_path)
    report, c_dense, c_sparse = compare_dense_vs_sparse(cfg, a, w, row_tile=3)
    pruned = sparsity_2_4(w, "K")
    np.testing.assert_array_equal(c_dense, a @ pruned)
    np.testing.assert_array_equal(c_sparse, a @ pruned)
    assert report["both_correct"] and report["outputs_bit_identical"]
    assert report["useful_macs_equal"] and report["sparse"]["skipped_macs"] > 0
    assert report["dense"]["useful_macs"] == report["sparse"]["useful_macs"]
    assert report["dense"]["skipped_macs"] == 0
    assert report["sparse_over_dense"]["executed_multiplications"] == 0.5
    assert (report["sparse"]["tiles"] * 2) == report["dense"]["tiles"]
    assert report["sparse"]["metadata_bytes"] > 0
    assert report["dense"]["metadata_bytes"] == 0
    assert report["sparse_over_dense"]["weight_payload_bytes"] < 1.0


def test_compare_rejects_k_not_divisible_by_four():
    a, w = make_int_workload(3, 6, 4)
    with pytest.raises(ValueError, match="divisible by four"):
        compare_dense_vs_sparse(ExperimentConfig(size=4, dtype="fp32"), a, w)


def test_cli_compare_with_generated_data_and_missing_dimensions(tmp_path, capsys):
    out = tmp_path / "compare.json"
    assert main(["--compare", "--M", "8", "--N", "8", "--K", "16", "--size", "4",
                 "--dtype", "fp32", "--json", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["shape_MNK"] == [8, 8, 16] and r["both_correct"] and r["useful_macs_equal"]
    assert json.loads(capsys.readouterr().out) == r
    with pytest.raises(SystemExit) as exc:
        main(["--compare", "--M", "8", "--size", "4"])
    assert exc.value.code == 2
    assert "--M, --N and --K" in capsys.readouterr().err