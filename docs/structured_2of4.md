# Structured MEISSA prototype contract

This branch is a runnable design exploration model based on the supplied
MEISSA diagrams. No sparse RTL source was supplied or simulated. Do not label
these measurements as validated sparse RTL or full Atalla execution timing.

## Physical versus logical shape

For physical size S, there are S rows by S columns of multipliers. Dense mode
consumes S logical K positions per tile. Sparse mode consumes 2*S positions:
physical rows 2g and 2g+1 select from logical activations 4g through 4g+3.
Every column holds its own two indices. The physical tree still reduces S
products per output column. Output propagation and de-skewing reuse the original
MEISSA methods. The original source files are not edited.

The logical activation wave has 2*S values, represented as two lanes per
physical row. All candidates shift to the next output column each cycle;
selection happens locally using that column's indices. It is incorrect to
select one activation vector globally: different weight columns have different
nonzero locations.

Weights and indices shift together, last output column first. Each physical
pair must have two distinct indices in 0..3. Zero or one actual nonzero is
represented with zero-valued filler entries. Invalid groups are rejected.

## Encoding assumption

An NPZ package stores A, W_values and W_indices, plus JSON metadata. W_values
and W_indices both have shape [S,S]. Optional W must agree with reconstruction.
Values are sent in column-major order, followed by the column-major two-bit
index stream (four indices per byte, first in low bits). There is no packet
header. This layout is a documented software contract pending the RTL format.

BF16 and FP16 values take two bytes; FP32 takes four. S=16 BF16/FP16 sparse
weights take 512 value bytes plus 64 metadata bytes. Their logical dense tile
has 32*16 weights, or 1024 bytes. Scratchpad transfer word and DRAM burst
rounding are measured separately. NPZ file size is not hardware traffic size.

## Clock boundary and assumptions

The optional memory path uses unchanged original memory component code and
its phase order. The harness stages one combined weight+activation byte stream
in one pad, then reads it back, then starts the array. The selected payload is
actually decoded from the returned memory bytes. It does not merely add a
formula-derived delay to a host matrix multiplication.

There is no VLSU/register file/GSAU timing between the frontend and the array,
no overlap with array compute, and no output DMA store. Memory parameters,
zero-cycle default mux latency and always-ready default consumer are explicit
experiment assumptions. Sparse mux delay is modeled as an extra stationary-
weight per-column product delay. Consumer throttling is separate from global
pipeline stall. Reports expose timing scope and unvalidated-RTL status.

The dense direct mode is compared with the original bridge under the same
accepted weight/activation schedule. This establishes equality at that boundary,
not equality of complete workload schedules before that boundary. The native
credit adapter from the previous dense mini remains in place; dense default
benchmarks use the Python backend.

## Full rectangular GEMM

The serial runner divides M into row batches, N into S-column tiles, and K
into S-position dense or 2*S-position sparse tiles. It zero-pads edge tiles.
It checks every array result against an arithmetic oracle, then accumulates
both actual and reference tiles sequentially in software FP32.

It deliberately does not invent vector-core accumulation timing. Therefore:
- serial_tile_cycles is a measured sum at the selected tile boundary;
- total_cycles is null for full GEMM;
- accumulation_cycles_included and full_atalla_workflow_equivalent are false.

Full runs include neither original reuse/overlap policies nor DRAM output stores.
Dense and sparse floating-point answers may differ due to reduction association
and different K-tile splits. Use exactly representable values for strict cross-
mode equality, and use each mode's own tree oracle for fractional correctness.

## RTL details still needed for calibration

1. Confirm fixed physical S rows versus fixed logical K capacity.
2. Confirm K grouping, within-group index ordering and weight metadata packing.
3. Confirm whether metadata/value streams have independent ports or cycles.
4. Confirm activation input-buffer width, fill timing, and valid/ready signals.
5. Confirm whether the mux adds a registered stage and its stall behavior.
6. Confirm output-buffer read handshake and credit return timing.
7. Confirm memory layout, pad mapping, VLSU/GSAU path and multi-tile accumulation.

None of those unknowns is presented as a proven RTL property in the reports.
