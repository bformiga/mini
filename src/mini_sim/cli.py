import argparse
import csv
import json
from pathlib import Path

import numpy as np
from .experiment import ExperimentConfig, run_experiment, run_rectangular
from .sparse_codec import decompress_2of4


def destination(name):
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description="Dense baseline and prototype structured 2:4 MEISSA experiments")
    p.add_argument("--mode", choices=["dense", "sparse"], help="infer from package or default to dense")
    p.add_argument("--load-path", choices=["direct", "atalla"], default="direct")
    p.add_argument("--mux-latency", type=int, default=0, help="uncalibrated sparse mux delay, cycles")
    p.add_argument("--output-read-interval", type=int, default=1)
    p.add_argument("--dram-latency", type=int, default=6)
    p.add_argument("--dram-burst-bytes", type=int, default=8)
    p.add_argument("--memory-queue-depth", type=int, default=16)
    p.add_argument("--spad-read-latency", type=int, default=2)
    p.add_argument("--spad-write-latency", type=int, default=1)
    p.add_argument("--full", action="store_true", help="tile full rectangular GEMM; FP32 software accumulation is untimed")
    p.add_argument("--row-tile", type=int, default=64)
    p.add_argument("--size", type=int, help="array size; infer from input or default to 16")
    p.add_argument("--rows", type=int, help="activation rows; infer from input or default to 64")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"],
                   help="infer from input metadata or default to bf16; explicit value overrides")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weight-pattern", choices=["dense", "2of4"], default="dense")
    p.add_argument("--mixed", action="store_true", help="use mixed four/two-input adder tree")
    p.add_argument("--mul-latency", type=int, default=1)
    p.add_argument("--add2-latency", type=int, default=1)
    p.add_argument("--add4-latency", type=int, default=3)
    p.add_argument("--input-interval", type=int, default=1)
    p.add_argument("--stall-every", type=int, default=0)
    p.add_argument("--native", action="store_true")
    p.add_argument("--input", help="NPZ with A[rows,size] and W[size,size]; overrides generated data")
    p.add_argument("--json", help="write benchmark report")
    p.add_argument("--trace", help="write per-cycle CSV")
    p.add_argument("--output", help="write actual C and reference C to NPZ")
    args = vars(p.parse_args(argv))
    paths = {key: args.pop(key) for key in ("input", "json", "trace", "output")}
    full, row_tile = args.pop("full"), args.pop("row_tile")
    try:
        a = w = None
        values = indices = None
        metadata = None
        if paths["input"]:
            with np.load(paths["input"], allow_pickle=False) as data:
                a = data["A"]
                w = data["W"] if "W" in data else None
                if "W_values" in data or "W_indices" in data:
                    values, indices = data["W_values"], data["W_indices"]
                    reconstructed = decompress_2of4(values, indices)
                    if w is not None and not np.array_equal(w, reconstructed):
                        raise ValueError("W disagrees with compressed values/indices")
                    w = reconstructed
                if "metadata_json" in data:
                    metadata = json.loads(str(data["metadata_json"].item()))
                    if not isinstance(metadata, dict):
                        raise ValueError("input metadata_json must contain a JSON object")
            if w is None:
                raise ValueError("input requires W or W_values and W_indices")
        args["mode"] = args["mode"] or (metadata or {}).get("execution_mode", "sparse" if values is not None else "dense")
        if args["mode"] == "sparse" and (metadata or {}).get("sparsity_axis") == "N":
            raise ValueError("structured MEISSA uses 2:4 along K; regenerate with --sparsity-axis K")
        if paths["input"] and not full:
            factor = 2 if args["mode"] == "sparse" else 1
            if a.ndim != 2 or w.ndim != 2 or w.shape[0] != factor*w.shape[1] or a.shape[1] != w.shape[0]:
                raise ValueError("mini-sim accepts single tiles A[rows,K], W[K,size] with K=size (dense) "
                                 "or K=2*size (sparse); use matching tiles/ files or --full")
        args["size"] = args["size"] if args["size"] is not None else (w.shape[1] if w is not None and not full else 16)
        args["rows"] = args["rows"] if args["rows"] is not None else (a.shape[0] if a is not None else 64)
        args["dtype"] = args["dtype"] or (metadata or {}).get("dtype", "bf16")
        cfg = ExperimentConfig(**args)
        if full:
            if a is None:
                raise ValueError("--full requires an input package")
            if paths["trace"]:
                raise ValueError("--trace is a single-tile option; omit it for --full")
            result = run_rectangular(cfg,a,w,row_tile=row_tile)
        else:
            result = run_experiment(cfg, a, w, values=values if cfg.mode=="sparse" else None,
                                    indices=indices if cfg.mode=="sparse" else None,
                                    collect_trace=bool(paths["trace"]))
        if paths["input"]:
            result.report["input_file"] = paths["input"]
            if metadata is not None:
                result.report["input_metadata"] = metadata
        report = json.dumps(result.report, indent=2, allow_nan=False) + "\n"
        if paths["json"]:
            destination(paths["json"]).write_text(report)
        if paths["trace"]:
            with destination(paths["trace"]).open("w", newline="") as stream:
                fields = list(dict.fromkeys(key for row in result.trace for key in row))
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(result.trace)
        if paths["output"]:
            with destination(paths["output"]).open("wb") as stream:
                np.savez(stream, C=result.output, reference=result.expected)
        print(report, end="")
        return 0 if result.report["correctness_passed"] else 1
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        p.exit(2, f"mini-sim: {exc}\n")

