import argparse
import csv
import json
from pathlib import Path

import numpy as np
from .benchmark import Config, run


def destination(name):
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description="Original dense MEISSA compute benchmark")
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
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
    try:
        cfg = Config(**args)
        a = w = None
        if paths["input"]:
            with np.load(paths["input"], allow_pickle=False) as data:
                a, w = data["A"], data["W"]
        result = run(cfg, a, w, collect_trace=bool(paths["trace"]))
        report = json.dumps(result.report, indent=2, allow_nan=False) + "\n"
        if paths["json"]:
            destination(paths["json"]).write_text(report)
        if paths["trace"]:
            with destination(paths["trace"]).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(result.trace[0]))
                writer.writeheader()
                writer.writerows(result.trace)
        if paths["output"]:
            with destination(paths["output"]).open("wb") as stream:
                np.savez(stream, C=result.output, reference=result.expected)
        print(report, end="")
        return 0 if result.report["correctness_passed"] else 1
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        p.exit(2, f"mini-sim: {exc}\n")

