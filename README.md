# Atalla mini simulator

A standalone **dense compute baseline** extracted from the supplied Atalla-Sim.
It benchmarks the original MEISSA multiplier grid, pipelined adder trees,
numeric reducer, and output buffer without booting the full accelerator.
This is the starting point for SoCET's future 2:4 structured sparsity model.

## Install and run

Requires Python 3.10+ and NumPy. No private submodules or compiler are required.

```sh
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
mini-sim --size 16 --rows 64 --json results/dense.json --trace results/dense.csv
python -m pytest -q
```

Without installing the package, run from the repository root:

```sh
PYTHONPATH=src python -m mini_sim --size 16 --rows 64
```

The `PYTHONPATH=...` syntax is for Linux/macOS. The installed `mini-sim` command
works on Windows too. Reports print to the terminal; file outputs are optional.

## What was isolated

| Files | Purpose |
| --- | --- |
| `src/systolic_array/systolic_array_meissa.py` | Original MEISSA compute model; benchmark target |
| `src/systolic_array/systolic_array_tpu.py` | Original alternative TPU array; retained with its component tests |
| `src/base/` | Original clocks, event machinery, queues, numeric conversion and debug helpers |
| `src/native/` | Original optional C++ acceleration and Python bindings |
| `src/mini_sim/benchmark.py` | New direct cycle driver and benchmark measurements |
| `src/mini_sim/reference.py` | New independent scalar arithmetic reference |
| `src/mini_sim/array_adapter.py` | Native credit synchronization correction, outside upstream files |
| `src/mini_sim/cli.py` | Command line, reports, traces, and NPZ inputs/outputs |
| `tests/systolic_array/` | Original standalone array tests |

The scheduler, ISA/compiler path, full vector core, scratchpads, DMA/DRAM,
full-system bridges, plotting stack and full-system performance monitor are
excluded. The benchmark calls the array's `tick()` once per cycle, as the
original standalone component tests do. It does not run an RTL simulator.

Upstream component files are copied without modifications. The additional
package initializer is empty. Source attribution and the original Apache 2.0
license are retained; the archive does not identify a Git commit, so no upstream
commit hash is claimed.

## Workload and timing boundary

The workload is `C = A @ W`, where `A` is `[rows,size]`, `W` is `[size,size]`,
and `size` is a power of two >= 2. Rectangular M is supported; K and N both
equal the physical array size. Multi-tile GEMM and cross-tile partial sums are
outside this baseline. MEISSA has no nonzero partial-sum input.

1. Stream `size` weight columns through the existing weight-shift input,
   in reverse column order so they land correctly. Setup is measured.
2. Offer complete activation vectors, respecting the original credit interface.
3. Tick until exactly `rows` outputs have retired, including pipeline drain.
4. Compare all results against the scalar reference. A mismatch returns exit 1;
   invalid configuration or unavailable requested native acceleration returns 2.

Inputs use BF16 by default. FP16 and FP32 (upstream `dtype=None`) are available.
Products and tree results use FP32. The reference follows the selected binary
or mixed four/two-input reduction order and output rounding; generic NumPy
matmul can differ because of its accumulation order. It shares upstream dtype
conversion helpers, but never calls the array's arithmetic implementation.

Cycle indices in traces and input/output lists start at zero. `total_cycles`
counts all ticks through the final output. `weight_setup_cycles` includes any
stall ticks during loading; `compute_and_drain_cycles` is the remaining time.
`first_row_latency_cycles` counts both the input tick and output tick. For an
unstalled first row, that latency is `size + pipeline_depth`.

The driver offers inputs **before** each tick. When credits reach zero, it
waits until the following tick to use a credit returned by retirement. This
can produce input bubbles in long streams; `input_backpressure_cycles` and
the trace expose the original behavior. No ideal-throughput correction is
applied. The model automatically retires into its host-side output list;
downstream consumer backpressure is not modeled.

## Benchmark options

```sh
# Original dense hardware processing 2:4 zero-patterned weights:
mini-sim --size 16 --rows 64 --weight-pattern 2of4 --json results/2of4-data.json

# Compare tree shapes, input bubbles, stalls and pipeline latencies:
mini-sim --size 8 --rows 32 --mixed --mul-latency 2 --add4-latency 3 \
  --input-interval 2 --stall-every 5 --trace results/stalls.csv

# Save computed C and arithmetic reference:
mini-sim --size 8 --rows 32 --output results/output.npz
```

`2of4` creates two retained weights in each group of four along K, per output
column. It is **test data on the original dense hardware**: no compression,
metadata, operand-selection logic, zero skipping, or sparse speedup is modeled.
Dense and zero-patterned weights have the same timing for the same configuration.
The new sparsity RTL has not yet been supplied.

Use `--input workload.npz` to supply arrays named `A` and `W`. Array size and
row count are inferred unless supplied explicitly. If the file includes
`metadata_json` with a `dtype`, that format is inferred too; an explicit
`--dtype` overrides it. Files without format metadata default to BF16.
External inputs replace generated data; the report marks
`workload_source=external`, and `weight_pattern` only describes the generator
configuration, not a validation of external inputs. Inputs must be finite;
workloads that overflow the selected reference arithmetic are rejected.

Reports include correctness, setup/compute/total cycles, first-result latency,
input acceptance/output arrival cycles, stalls, backpressure, throughput,
dense-equivalent FLOPs, and uncompressed boundary payload bytes. Dense-equivalent
FLOPs use the conventional `2*M*N*N`; separate mathematical multiply and binary
add counts are also provided. These counts do not claim to measure physical
adder hardware operations or energy. Payload bytes count each supplied operand
and produced output once (2 bytes for BF16/FP16, 4 for FP32), excluding memory
transactions, metadata, and internal shifting. Simulator wall time excludes
reference computation and is a host measurement, not hardware latency.

## Optional acceleration

On Linux with a C++14 compiler and Make:

```sh
make native
mini-sim --size 32 --rows 128 --native
python -m pytest -q
```

Default benchmarks explicitly use the Python/NumPy path. `--native` requires
the library and reports the actual backend. The benchmark adapter corrects an
upstream native-wrapper issue: the C++ kernel returns an output credit, then
Python `_emit()` returns it again. After each native call the adapter restores
the kernel's authoritative credit count. This preserves Python baseline timing;
the extracted source files themselves remain unchanged. Tests compare both
outputs and acceptance/retirement cycles across backends under stalls.
Build from an editable checkout;
the shared object is not distributed in the Python package. Native equivalence
tests skip when the library is absent. No whole-system or private golden-simulator
tests are included in this isolated project.

## Future sparsity branch

Keep this dense baseline as the comparison point. Once the 2:4 RTL is supplied,
add its compressed values/metadata interfaces, selection datapath, pipeline
latencies and flow control as a separate component model. Run the same A and
logically equivalent W against both models, comparing correctness, setup bytes,
cycles and throughput. Do not assume a 2x speedup: metadata, stalls and pipeline
overhead must be measured from the actual design.

## Your workload package

The adapted [`generate_benchmarks.py`](generate_benchmarks.py) preserves your
three synthetic workload shapes, listed in **M, N, K** order:

| Workload | M | N | K |
| --- | ---: | ---: | ---: |
| `resnet50_conv1` | 12544 | 64 | 160 |
| `llama3_q_proj` | 32 | 1024 | 4096 |
| `llama3_gate_proj` | 256 | 1024 | 4096 |

These names identify your chosen matrix dimensions; inputs are random synthetic
data, not model checkpoints or a complete neural-network implementation.

```sh
# Generate all three full input pairs and one runnable tile pair per workload:
python generate_benchmarks.py --output-dir benchmarks

# Run the generated tile directly; size, rows and FP16 format are inferred:
mini-sim --input benchmarks/tiles/resnet50_conv1_dense.npz --json results/resnet_dense.json
mini-sim --input benchmarks/tiles/resnet50_conv1_sparse.npz --json results/resnet_sparse.json

# Select another aligned K/N slice and different activation rows:
python generate_benchmarks.py --workload llama3_q_proj --tile-size 32 \
  --tile-rows 16 --k-start 32 --n-start 64 --row-start 8 --output-dir benchmarks/q_tile

# Generate a BF16 package matching the default MEISSA configuration:
python generate_benchmarks.py --dtype bf16 --output-dir benchmarks/bf16
```

After reinstalling with `python -m pip install -e ".[test]"`, `mini-generate`
is an equivalent command. Generation needs NumPy only, with seed 100 by default.
It uses NumPy's PCG64 rather than PyTorch's RNG, so a fixed seed is reproducible
within this generator but does not reproduce the original Torch tensors.
Each workload has an independent seed index; selecting just one workload does
not change its tensors. Generating again at the same output path replaces
matching generated files; use different directories to retain multiple variants.

| Output | Contents and purpose |
| --- | --- |
| `manifest.json` | Original shapes, dtype, grouping axis, tile offsets, filenames, and run commands |
| `full/<name>_dense.npz` | Complete `A[M,K]`, dense `W[K,N]`, JSON metadata |
| `full/<name>_sparse.npz` | Same A, magnitude-pruned W, retained-position mask, metadata |
| `tiles/<name>_dense.npz` | Runnable A/W tile, `C_reference`, metadata |
| `tiles/<name>_sparse.npz` | Corresponding tile with pruned weights, `C_reference`, metadata |

The original `np.save("...log", array)` actually wrote binary NumPy files named
`...log.npy`. Explicit `.npz` files now match the mini simulator's A/W loader.
`allow_pickle=False` can be used for every generated file; metadata is a scalar
JSON string, not a pickled Python object. Large generated data is git-ignored.

Pruning retains the two largest absolute values per four-element group. The
default groups along **K**, i.e. four consecutive rows of W per output column,
consistent with the mini simulator's existing convention. Your original script
grouped along **N**. Use `--sparsity-axis N` to retain that orientation. Ties
prefer earlier positions. The full sparse package records the retained-position
mask, since a retained weight may itself be zero. This mask is data provenance,
not a compressed hardware encoding. The RTL will determine the eventual format.

The generator defaults to FP16 to preserve your original data format. BF16 is
stored as rounded values in FP32 NumPy arrays, since NumPy has no native BF16
storage type here. Metadata records the intended arithmetic format.

**A tile is one K-slice contribution, not a complete layer benchmark.** Default
tiles use array size 16 and up to 64 activation rows, starting at M=K=N=0.
K/N offsets and array size must be multiples of four to preserve group boundaries.
Final K/N edges are zero-padded; the manifest records valid dimensions. The
full inputs are retained for later scheduling work, but the mini simulator does
not yet execute arbitrary full M/N/K GEMMs or accumulate multiple K slices.
No total layer latency is estimated from a single tile.

`C_reference` follows the binary MEISSA adder tree and selected output rounding.
The simulator independently recomputes its own reference; `--mixed` uses the
mixed-tree oracle instead of the file's binary-tree reference. Optional
`--full-reference` also writes full FP32 NumPy matmul outputs in
`full/<name>_numpy_reference.npz`; those are diagnostic outputs, not exact
MEISSA goldens. Both dense and pruned tile files run through the same original
dense datapath, so pruning alone provides no modeled speedup.

