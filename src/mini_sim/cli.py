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
    try:
        a = w = None
        metadata = None
        if paths["input"]:
            with np.load(paths["input"], allow_pickle=False) as data:
                a, w = data["A"], data["W"]
                if "metadata_json" in data:
                    metadata = json.loads(str(data["metadata_json"].item()))
                    if not isinstance(metadata, dict):
                        raise ValueError("input metadata_json must contain a JSON object")
            if a.ndim != 2 or w.ndim != 2 or w.shape[0] != w.shape[1] or a.shape[1] != w.shape[0]:
                raise ValueError("mini-sim accepts single tiles A[rows,size], W[size,size]; "
                                 "use the generator's tiles/ files rather than full/ inputs")
        args["size"] = args["size"] if args["size"] is not None else (w.shape[0] if w is not None else 16)
        args["rows"] = args["rows"] if args["rows"] is not None else (a.shape[0] if a is not None else 64)
        args["dtype"] = args["dtype"] or (metadata or {}).get("dtype", "bf16")
        cfg = Config(**args)
        result = run(cfg, a, w, collect_trace=bool(paths["trace"]))
        if paths["input"]:
            result.report["input_file"] = paths["input"]
            if metadata is not None:
                result.report["input_metadata"] = metadata
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

