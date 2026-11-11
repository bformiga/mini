# Operation counters and dense-versus-sparse comparison

The counters and the comparison run on the same design-exploration model as the
rest of this branch (see [structured_2of4.md](structured_2of4.md)). No sparse RTL
was simulated, so none of the numbers here are validated RTL or full Atalla
timing.

## Counters

Every report, for a single tile or `--full`, contains operation counters. They
are counted from the weight data and are not timed. For `rows` rows of A and a
logical weight tile W[K,N]:

| Counter | Meaning |
| --- | --- |
| dense_macs | rows * N * K: the multiplies a dense array would do |
| issued_macs | Multiplies the array does. Equal to dense_macs on the dense array, half of it on the sparse array (two kept weights per group of four) |
| skipped_macs | dense_macs minus issued_macs: the multiplies the 2:4 structure avoids |
| zero_slot_macs | Issued multiplies whose weight is zero: a kept slot holding zero, or edge padding. The dense array also counts the pruned zeros |
| useful_macs | Issued multiplies whose weight is nonzero |
| effective_weight_density | Nonzero weights divided by logical weights |

`issued_macs` always equals `useful_macs` plus `zero_slot_macs`. On the same
pruned W, the dense and the sparse array report the same `useful_macs`. In
`--full` reports the counters are summed over all tiles, including edge padding,
and the density is recomputed from the summed counts.

## Comparing dense and sparse

`--compare` prunes W once (2:4 along K, keeping the two largest magnitudes; equal
magnitudes keep the lower index) and runs the dense array and the sparse array on
the same A and the same pruned W:

```sh
# Random data, generated from --seed:
mini-sim --compare --M 64 --N 96 --K 128 --size 16 --mixed \
  --load-path atalla --json results/compare.json

# A generated full package (see "Generate workloads separately" in the README):
mini-sim --compare --input benchmarks/full/NAME_sparse.npz --size 16
```

`--compare` needs `--input`, or all of `--M`, `--N` and `--K`. K must be divisible
by four. `--trace` and `--output` are single-tile options and are rejected. The
mode is set for you; `--size`, `--dtype`, `--load-path`, `--row-tile`, `--mixed`
and the latency options apply to both runs, except `--mux-latency`, which applies
to the sparse run only because the dense array has no mux.

The report has a `dense` and a `sparse` section with the counters, cycles and
bytes, plus `sparse_over_dense` ratios for executed multiplies, serial tile
cycles and weight payload bytes (metadata included). It also holds
`useful_macs_equal`, `outputs_bit_identical`, `max_abs_output_difference` and
`both_correct`. The command exits with 0 when both runs match their reference
output, and with 1 otherwise.

## What the comparison does and does not show

At the same S, a sparse tile covers twice the K of a dense tile, so the sparse
run needs half as many tiles. The cycle ratio reflects that and the timing
assumptions in the README's
[Precision, latency and flow control](../README.md#precision-latency-and-flow-control)
section. It has not been validated against RTL. Cycles are sums of per-tile measurements, as
described under
[Rectangular full workloads](../README.md#rectangular-full-workloads).

Dense and sparse outputs can differ by floating-point rounding, so each run is
checked against its own reference. `outputs_bit_identical` is true on exactly
representable data.

## Reproducing the results

Run everything from the repository root. After `python -m pip install -e ".[test]"`
(with the virtual environment activated) the `mini-sim` command is available.
Without installing, replace `mini-sim` with `PYTHONPATH=src python -m mini_sim`
on Linux/macOS, or in Windows PowerShell run `$env:PYTHONPATH = "src"` once and then
use `python -m mini_sim`.

**1. Check that the counter tests pass**

```sh
python -m pytest tests/test_counters.py -q
python -m pytest -q
```

The first command runs the counter, comparison and `--compare` tests (11 test
cases) and should end with `11 passed`. The second runs the whole suite; native
tests are skipped when the C++ library is absent.

**2. Counters for one tile**

```sh
# Dense array, 2:4-patterned weights: it still does every multiply.
mini-sim --size 16 --rows 64 --weight-pattern 2of4 --json results/dense_tile.json

# Sparse array: it does half the multiplies of its (twice as large) logical tile.
mini-sim --mode sparse --size 16 --rows 64 --json results/sparse_tile.json

python -c "import json; r = json.load(open('results/sparse_tile.json')); print({k: r[k] for k in ('dense_macs', 'issued_macs', 'skipped_macs', 'zero_slot_macs', 'useful_macs', 'effective_weight_density')})"
```

At size 16 the dense tile has K = 16 and the sparse tile has K = 32, so the two
reports describe different amounts of logical work: 16,384 dense and issued MACs
(8,192 useful) for the dense tile, and 32,768 dense MACs, 16,384 issued, 16,384
skipped and 16,384 useful for the sparse tile.

**3. Dense vs. sparse on the same GEMM**

```sh
mini-sim --compare --M 64 --N 96 --K 128 --size 16 --mixed --json results/compare_direct.json
mini-sim --compare --M 64 --N 96 --K 128 --size 16 --mixed --load-path atalla --json results/compare_atalla.json
```

Both commands print the report and write it to the `--json` file. Expected values
for this problem (BF16, mixed tree, any `--seed`):

| Field | Dense | Sparse |
| --- | ---: | ---: |
| tiles | 48 | 24 |
| executed_multiplications | 786,432 | 393,216 |
| skipped_macs | 0 | 393,216 |
| useful_macs | 393,216 | 393,216 |
| weight_payload_bytes | 24,576 | 13,824 (1,536 of it metadata) |
| serial_tile_cycles, `--load-path direct` | 4,992 | 2,496 |
| serial_tile_cycles, `--load-path atalla` | 32,160 | 27,168 |

`both_correct` and `useful_macs_equal` must be `true`. Cycle counts change if you
change the latency, `--mixed`, `--load-path` or memory options.

**4. Reading a report**

In single-tile and `--full` reports the counters sit at the top level, so any
field can be read with a one-line script such as the one in step 2. In a
`--compare` report they are nested under `dense` and `sparse`, and the ratios are
under `sparse_over_dense`. Repeating a command gives the same numbers. The one
exception is `simulator_wall_seconds` in single-tile reports, which is a
wall-clock measurement.