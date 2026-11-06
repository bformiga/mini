# Atalla mini: dense baseline and structured 2:4 MEISSA experiments

This branch adds a **diagram-based sparse prototype**, original Atalla memory
components, and rectangular workload tiling to the existing dense baseline.
It has not been validated against your sparse RTL. The dense array files and
native kernels remain unchanged.

## Install and run

Python 3.10+ and NumPy are required. PyTorch and private submodules are not.

```sh
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q

# Existing dense compute baseline (unchanged default behavior):
mini-sim --size 16 --rows 64

# Prototype sparse array: 16x16 physical multipliers, logical W[32,16]:
mini-sim --mode sparse --size 16 --rows 64 --load-path atalla \
  --json results/sparse.json --trace results/sparse.csv
```

Without installation, Linux/macOS can use `PYTHONPATH=src python -m mini_sim`.
The `mini-sim` command works on Windows after installation too.

## What the two modes mean

| Mode, with physical size S | A | Logical W | Multiplier grid |
| --- | --- | --- | --- |
| dense | [rows,S] | [S,S] | S by S |
| sparse | [rows,2*S] | [2*S,S] | S by S |

Sparse mode assumes the same number of physical multiplier rows handles twice
as many logical K positions, as suggested by the supplied diagrams. Every pair
of physical rows shares four candidate activations. Each retained weight has a
two-bit index; the modeled 4:1 selection chooses its matching activation
**independently in each output column**. Products use the existing MEISSA FP32
reduction helpers, reducer, output banks, and credit interface.

This is an assumption about the proposed design, not confirmation from RTL.
Dense and sparse single tiles contain different amounts of logical work. Use
the same full A/W problem with `--full` when comparing tile decomposition.

## Generate workloads separately

```sh
python generate_benchmarks.py --output-dir benchmarks

# Original square tile on dense hardware:
mini-sim --input benchmarks/tiles/resnet50_conv1_dense.npz

# Pruned square tile still running on dense hardware:
mini-sim --input benchmarks/tiles/resnet50_conv1_sparse.npz

# New rectangular logical tile on the sparse prototype, with memory loading:
mini-sim --input benchmarks/tiles/resnet50_conv1_structured.npz --load-path atalla
```

The generator writes files only; it never runs the cycle simulator. Files can
be generated elsewhere and copied into the simulator environment. `mini-generate`
is an equivalent installed command. FP16 remains the generator's default;
`--dtype bf16` selects BF16. The loader infers dimensions, arithmetic format,
and sparse execution mode from structured packages. Explicit `--dtype` overrides
format metadata. Existing plain A/W files default to dense BF16 execution.

The original synthetic workload dimensions, in **M,N,K** order, are preserved:

| Workload | M | N | K |
| --- | ---: | ---: | ---: |
| resnet50_conv1 | 12544 | 64 | 160 |
| llama3_q_proj | 32 | 1024 | 4096 |
| llama3_gate_proj | 256 | 1024 | 4096 |

These are random tensors with your chosen dimensions, not trained checkpoints.
NumPy PCG64 uses seed 100 by default, with independent workload seed indices;
these tensors differ from Torch tensors generated with the same seed.

| Generated file | Contents |
| --- | --- |
| manifest.json | Shapes, dtype, grouping axis, tile offsets, paths and run commands |
| full/name_dense.npz | Complete A and dense W |
| full/name_sparse.npz | Complete A, pruned W, keep_mask; W_values/W_indices when axis is K |
| tiles/name_dense.npz | Original runnable dense tile and binary-tree C_reference |
| tiles/name_sparse.npz | Original pruned tile, still for dense execution |
| tiles/name_structured.npz | A[rows,2*S], W_values[S,S], W_indices[S,S], sparse C_reference |

NPZ replaces misleading `.log.npy` filenames produced by `np.save('name.log',...)`.
All files load with `allow_pickle=False`; metadata_json is a scalar JSON string.
Sparse packages store numeric indices as uint8 for convenience, while the
modeled wire format packs them into two bits per retained value. Weights with
more than two nonzeros per K-group are rejected, not silently pruned at runtime.
The generator performs pruning explicitly and resolves magnitude ties by position.

`--sparsity-axis K` is the default and is required by the new sparse model.
`--sparsity-axis N` preserves the uploaded script's old grouping direction but
does not produce structured-MEISSA packages. Retained positions can themselves
contain zero. `--full-reference` writes diagnostic NumPy FP32 full matmuls;
these are not MEISSA arithmetic goldens.

## Restored loading path

`--load-path direct` keeps the compute-only benchmark. `--load-path atalla`
uses the original **DRAM, Backend/DMA, Scratchpad and Frontend** implementations:

1. Put the serialized weight/metadata and activation payload in DRAM.
2. Transfer it through actual queued DMA transactions and scratchpad writes.
3. Read it back through the scratchpad frontend.
4. Decode that returned data, stream weight columns into the array, and issue
   activation vectors subject to credits. The memory data is actually used.

The original 64-bit DRAM interface is the default: eight-byte bursts. Memory
order is scratchpad tick followed by backend tick, matching the original
platform's memory phases. Defaults: DRAM latency 6, queue depth 16, scratchpad
read latency 2/write latency 1; these are configurable experiment settings.
The harness uses one scratchpad pad and serial staging, not the full platform's
four-pad reuse policy. There is no memory/compute overlap in this harness.

Reports separate memory cycles, array cycles, payload bytes, packed metadata
bytes, launched bursts, and burst-slot capacity. Final memory words are padded
to two bytes. Half the retained weight values does not imply exactly half the
transactions or total cycles, because metadata, burst rounding and activations
also take space and time. Output stores to DRAM are not modeled.

**This does not restore VLSU, VRF, GSAU issue/writeback, scheduler, or their
arbitration. It is not yet the original full Atalla workflow.**

## Rectangular full workloads

```sh
# Same logical pruned problem on the dense baseline:
mini-sim --input benchmarks/full/llama3_q_proj_sparse.npz --mode dense \
  --full --size 16 --load-path atalla --json results/full_dense.json

# Same problem on the sparse prototype:
mini-sim --input benchmarks/full/llama3_q_proj_sparse.npz --mode sparse \
  --full --size 16 --load-path atalla --json results/full_sparse.json
```

`--full` tiles arbitrary compatible A[M,K] and W[K,N] along M, N and K; final
edges are zero-padded. `--row-tile` defaults to 64. Sparse K must be divisible
by four, preserving the original group boundaries. Full runs may take a long
time; start with generated tile packages.

Partial tile outputs are combined in **software FP32**, in K-tile order.
The vector-core accumulation path and its timing are not modeled. Consequently
reports contain `serial_tile_cycles` and **`total_cycles: null`**. The cycle sum
is the sum of individually staged tile measurements, excludes accumulation and
output stores, and is not an end-to-end Atalla layer latency. There is no cross-
tile reuse, overlap or original instruction scheduling. Full CSV traces are not
implemented; `--trace` is for individual tiles.

## Precision, latency and flow control

BF16/FP16 input and output rounding and FP32 arithmetic are retained. FP32 mode
uses the upstream dtype=None arithmetic. The scalar sparse oracle independently
selects operands and follows the physical tree reduction order. Dense and sparse
results can differ in floating-point rounding because their reduction grouping
and K-tile boundaries differ. Each mode is checked against its corresponding
oracle; exact dense/sparse equality is tested on exactly representable cases.

`--mixed`, `--mul-latency`, `--add2-latency`, and `--add4-latency` configure the
original pipeline. `--mux-latency` defaults to **0 as an explicit uncalibrated
assumption**. For stationary weights its delay is represented in the per-column
product pipeline; it is not a transistor-level mux model or a measured RTL value.

`--output-read-interval 1` is the original always-ready sink. Larger values add
an explicit periodic consumer and return credits only when it reads a result.
`--stall-every` freezes the array; it is distinct from output-consumer stalls.
`--input-interval` controls offered activation traffic. Cycle indices start at
zero. Sparse reports distinguish output availability from output read cycles.

## Files and validation

| File/module | Role |
| --- | --- |
| src/systolic_array/ | Unmodified original MEISSA and alternative TPU array |
| src/base/ | Original simulation/numeric utilities |
| src/native/ | Original optional dense C++ kernels |
| src/memory/ | Restored original memory modules; empty package initializer added |
| src/mini_sim/benchmark.py | Original mini dense driver, unchanged |
| src/mini_sim/array_adapter.py | Existing native credit double-return correction, unchanged |
| src/mini_sim/structured_array.py | New logical lanes, metadata shift and column-specific operand selection |
| src/mini_sim/sparse_codec.py | New validation, compression and packed wire encoding |
| src/mini_sim/memory_path.py | New small harness around original memory models |
| src/mini_sim/experiment.py | New modes, flow control, reports and rectangular driver |
| src/mini_sim/workloads.py | Separate data generator, extended with compressed packages |
| src/mini_sim/reference.py | Independent dense oracle plus new sparse oracle |
| src/mini_sim/cli.py | Input formats, mode selection, memory parameters and --full |
| src/mini_sim/original_bridge.py | Extracted original bridge class with standalone imports/helper, used for tests |

Existing component and benchmark tests are retained. Original memory tests and
their small data fixtures are restored. New tests cover all six index patterns,
different masks by column, zero groups, packing, memory readback and burst
counts, stalls, credits, mux delay, and rectangular edge tiles. Dense boundary
outputs, credit values and arrival cycles are compared against the original
extracted GSAU bridge under identical boundary traffic. That comparison does
not validate the excluded upstream scheduling or sparse RTL timing.

Optional dense acceleration: `make native`, then `mini-sim --native`.
Sparse C++ acceleration is not implemented and sparse --native is rejected.
Native tests skip when the library is absent.

See [docs/structured_2of4.md](docs/structured_2of4.md) for the prototype contract
and remaining RTL questions. Original source attribution and LICENSE are retained.
