from dataclasses import replace
import json

import numpy as np
import pytest

from mini_sim import Config, run
from mini_sim.benchmark import make_workload
from mini_sim.cli import main
from native import kernels


@pytest.mark.parametrize("size", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("mixed", [False, True])
def test_fractional_products_and_long_streams(size, dtype, mixed):
    result = run(Config(size=size, rows=3 * size + 7, dtype=dtype, mixed=mixed))
    assert result.report["correctness_passed"]
    np.testing.assert_array_equal(result.output.view(np.uint32), result.expected.view(np.uint32))
    assert len(result.report["output_cycles"]) == 3 * size + 7


def test_streamed_weight_orientation_and_first_result_timing():
    cfg = Config(size=4, rows=3, dtype="fp32")
    a = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [-2, 0, 1, 3]], dtype=float)
    w = np.arange(16).reshape(4, 4)
    result = run(cfg, a, w)
    np.testing.assert_array_equal(result.output, a @ w)
    assert result.report["weight_setup_cycles"] == 4
    assert result.report["first_row_latency_cycles"] == 4 + result.report["pipeline_depth"]
    assert result.report["total_cycles"] == 4 + 3 + 4 + result.report["pipeline_depth"] - 1


def test_stalls_bubbles_and_pipeline_latencies_preserve_results():
    cfg = Config(size=8, rows=29, mixed=True)
    normal = run(cfg)
    delayed = run(replace(cfg, stall_every=3, input_interval=4,
                          mul_latency=3, add2_latency=2, add4_latency=4), collect_trace=True)
    assert delayed.report["correctness_passed"]
    np.testing.assert_array_equal(delayed.output, normal.output)
    assert delayed.report["total_cycles"] > normal.report["total_cycles"]
    assert delayed.report["stall_cycles"] > 0
    assert all(not row["input_accepted"] and not row["outputs"] and not row["weight_accepted"]
               for row in delayed.trace if row["stall"])
    assert len(delayed.trace) == delayed.report["total_cycles"]


def test_2of4_data_still_runs_on_dense_hardware():
    cfg = Config(size=8, rows=31, weight_pattern="2of4")
    a, w = make_workload(cfg)
    assert np.all(np.count_nonzero(w.reshape(2, 4, 8), axis=1) == 2)
    sparse_data = run(cfg, a, w)
    dense_data = run(replace(cfg, weight_pattern="dense"))
    assert sparse_data.report["correctness_passed"]
    assert sparse_data.report["total_cycles"] == dense_data.report["total_cycles"]
    assert sparse_data.report["architecture"] == "original_dense_meissa"


def test_cli_writes_trace_report_and_output(tmp_path, capsys):
    report, trace, output, source = [tmp_path / name for name in ("report.json", "trace.csv", "out.npz", "in.npz")]
    np.savez(source, A=np.ones((7, 4)), W=np.eye(4))
    assert main(["--size", "4", "--rows", "7", "--input", str(source),
                 "--json", str(report), "--trace", str(trace), "--output", str(output)]) == 0
    r = json.loads(report.read_text())
    assert r["correctness_passed"] and r["workload_source"] == "external"
    assert len(trace.read_text().splitlines()) == r["total_cycles"] + 1
    with np.load(output) as data:
        np.testing.assert_array_equal(data["C"], np.ones((7, 4)))
    capsys.readouterr()


@pytest.mark.parametrize("kwargs", [dict(size=3), dict(size=0), dict(rows=0),
    dict(stall_every=1), dict(input_interval=0), dict(dtype="int8"),
    dict(mul_latency=0), dict(size=2, weight_pattern="2of4")])
def test_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_rejects_invalid_workload():
    cfg = Config(size=4, rows=2)
    with pytest.raises(ValueError, match="both"):
        run(cfg, np.ones((2, 4)))
    with pytest.raises(ValueError, match="expected"):
        run(cfg, np.ones((3, 4)), np.eye(4))
    with pytest.raises(ValueError, match="finite"):
        run(cfg, np.full((2, 4), np.nan), np.eye(4))


def test_reference_distinguishes_tree_rounding():
    a = [[2**24, 1, 1, 1]]
    w = np.ones((4, 4))
    cfg = Config(size=4, rows=1, dtype="fp32")
    pure = run(cfg, a, w)
    mixed = run(replace(cfg, mixed=True), a, w)
    assert pure.report["correctness_passed"] and mixed.report["correctness_passed"]
    assert pure.output[0, 0] == 16777218
    assert mixed.output[0, 0] == 16777220


@pytest.mark.skipif(not kernels.HAVE_NATIVE, reason="optional native library not built")
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
def test_native_harness_matches_fallback_under_stalls(mixed, dtype):
    cfg = Config(size=8, rows=35, dtype=dtype, mixed=mixed, stall_every=5)
    python = run(cfg)
    native = run(replace(cfg, native=True))
    assert native.report["correctness_passed"]
    np.testing.assert_array_equal(python.output, native.output)
    assert native.report["input_cycles"] == python.report["input_cycles"]
    assert native.report["output_cycles"] == python.report["output_cycles"]


