"""Run the adapted workload generator from a checkout or installed environment."""

from pathlib import Path
import sys

# Allow `python generate_benchmarks.py` without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mini_sim.workloads import generate_benchmark, main, sparsity_2_4  # noqa: E402,F401


if __name__ == "__main__":
    raise SystemExit(main())

