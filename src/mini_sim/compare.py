"""Dense vs 2:4 sparse on the same A and the same pruned W."""

import numpy as np

from dataclasses import replace
from .experiment import run_rectangular
from .workloads import sparsity_2_4

_REPORT_FIELDS = ("tiles", "serial_tile_cycles", "memory_cycles", "executed_multiplications", "weight_payload_bytes", "metadata_bytes", 
                  "dense_macs", "issued_macs", "skipped_macs", "zero_slot_macs", "useful_macs", "effective_weight_density", "correctness_passed")

_RATIO_FIELDS = ("executed_multiplications", "serial_tile_cycles", "weight_payload_bytes")

def _ratio(sparse, dense):
    if (dense):
        return sparse / dense
    else:
        return None

def _pick(report, fields):
    """
    Copy only the listed fields out of a report.
    """
    picked = {}
    for field in fields:
        picked[field] = report[field]
    return picked

def compare_dense_vs_sparse(config, a, w, *, row_tile=64):
    """
    Run the dense and the 2:4 sparse array on the same inputs and compare them.

    W is first pruned to 2:4 along K (in each group of 4, the 2 largest magnitudes are kept. 
    If magnitudes are equal, the one with the lower index in the group is kept), then both arrays run on A and the same pruned W.
    K must be divisible by 4, and row_tile sets how many rows of A are processed per tile.
    Cycle counts are sums of per-tile measurements.

    Returns: report, dense_output, sparse_output
    """
    a = np.asarray(a, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)

    if ((a.ndim != 2) or (w.ndim != 2) or (a.shape[1] != w.shape[0])):
        raise ValueError(
            f"A must have shape [M, K] and W must have shape [K, N] with the same K, "
            f"but A has shape {a.shape} and W has shape {w.shape}.")
    if (w.shape[0] % 4 != 0):
        raise ValueError(
            f"K must be divisible by four for 2:4 grouping, but K is {w.shape[0]}.")

    pruned = sparsity_2_4(w, "K")

    # The dense array has no mux, so mux_latency must be 0 for the dense config.
    dense_config = replace(config, mode="dense", mux_latency=0)
    sparse_config = replace(config, mode="sparse")
    dense = run_rectangular(dense_config, a, pruned, row_tile=row_tile)
    sparse = run_rectangular(sparse_config, a, pruned, row_tile=row_tile)
    d = dense.report
    s = sparse.report

    ratios = {}
    for field in _RATIO_FIELDS:
        ratios[field] = _ratio(s[field], d[field])

    # Compare the 2 outputs 1 element at a time
    rows = dense.output.shape[0]
    cols = dense.output.shape[1]
    identical = True
    max_diff = 0.0
    for i in range(rows):
        for j in range(cols):
            dense_val = float(dense.output[i][j])
            sparse_val = float(sparse.output[i][j])
            if (dense_val != sparse_val):
                identical = False
            diff = abs(dense_val - sparse_val)
            if (diff > max_diff):
                max_diff = diff
    
    # True only if both the dense and the sparse run match their reference output
    both_correct = bool(d["correctness_passed"] and s["correctness_passed"])

    report = {}
    report["schema_version"] = 1
    report["shape_MNK"] = [a.shape[0], w.shape[1], a.shape[1]]
    report["size"] = config.size
    report["dtype"] = config.dtype
    report["mixed"] = config.mixed
    report["load_path"] = config.load_path
    report["pruning"] = "2:4 along K, keep two largest magnitudes, equal magnitudes keep the lower index"
    report["dense"] = _pick(d, _REPORT_FIELDS)
    report["sparse"] = _pick(s, _REPORT_FIELDS)
    report["sparse_over_dense"] = ratios
    report["useful_macs_equal"] = (d["useful_macs"] == s["useful_macs"])
    report["outputs_bit_identical"] = identical
    report["max_abs_output_difference"] = max_diff
    report["both_correct"] = both_correct
    report["timing_scope"] = "sum_of_serial_tile_measurements_excludes_accumulation"
    report["rtl_timing_validated"] = False

    return report, dense.output, sparse.output