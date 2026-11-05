"""Direct cycle driver for the original MEISSA model, without the full SoC."""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .array_adapter import BenchmarkMEISSA
from .reference import reference_product


@dataclass(frozen=True)
class Config:
    size: int = 16
    rows: int = 64
    dtype: str = "bf16"
    seed: int = 0
    weight_pattern: str = "dense"
    mixed: bool = False
    mul_latency: int = 1
    add2_latency: int = 1
    add4_latency: int = 3
    input_interval: int = 1
    stall_every: int = 0
    native: bool = False

    def __post_init__(self):
        if self.size < 2 or self.size & (self.size - 1):
            raise ValueError("size must be a power of two >= 2")
        if self.rows < 1 or self.input_interval < 1:
            raise ValueError("rows and input_interval must be positive")
        if min(self.mul_latency, self.add2_latency, self.add4_latency) < 1:
            raise ValueError("pipeline latencies must be positive")
        if self.dtype not in ("bf16", "fp16", "fp32"):
            raise ValueError("dtype must be bf16, fp16, or fp32")
        if self.stall_every < 0 or self.stall_every == 1:
            raise ValueError("stall_every must be 0 (disabled) or >= 2")
        if self.weight_pattern not in ("dense", "2of4"):
            raise ValueError("weight_pattern must be dense or 2of4")
        if self.weight_pattern == "2of4" and self.size % 4:
            raise ValueError("2of4 weights require size divisible by 4")


@dataclass
class Result:
    report: dict
    output: np.ndarray
    expected: np.ndarray
    trace: list


def make_workload(config):
    rng = np.random.default_rng(config.seed)
    a = rng.uniform(-1, 1, (config.rows, config.size)).astype(np.float32)
    w = rng.uniform(-1, 1, (config.size, config.size)).astype(np.float32)
    if config.weight_pattern == "2of4":
        # Groups run along K, independently for each output column.
        for col in range(config.size):
            for k in range(0, config.size, 4):
                w[k + rng.choice(4, 2, replace=False), col] = 0
    return a, w


def run(config=Config(), activations=None, weights=None, *, collect_trace=False):
    """Compute A[M,N] @ W[N,N]; count streamed setup and pipeline drain.

    Results are always checked against the scalar oracle. External memory,
    compression, metadata decoding and an output consumer are not modeled.
    """
    if (activations is None) != (weights is None):
        raise ValueError("provide both activations and weights, or neither")
    external = activations is not None
    if not external:
        activations, weights = make_workload(config)
    a, w = np.asarray(activations, dtype=np.float32), np.asarray(weights, dtype=np.float32)
    if a.shape != (config.rows, config.size) or w.shape != (config.size, config.size):
        raise ValueError("expected A[rows,size] and W[size,size]")
    if not np.isfinite(a).all() or not np.isfinite(w).all():
        raise ValueError("inputs must be finite")

    dtype = None if config.dtype == "fp32" else config.dtype
    sa = BenchmarkMEISSA(config.size, dtype=dtype, use_mixed_adder=config.mixed,
                            mul_latency=config.mul_latency, add2_latency=config.add2_latency,
                            add4_latency=config.add4_latency, use_native=config.native)
    if config.native and not sa.use_native:
        raise RuntimeError("native kernel unavailable; run make native or omit --native")
    expected = reference_product(a, w, dtype=dtype, mixed=config.mixed)
    if not np.isfinite(expected).all():
        raise ValueError("workload overflows selected arithmetic; use smaller values or a wider dtype")
    cycle = loaded = sent = 0
    setup_end = first_input = last_input = None
    accepted_cycles, output_cycles, trace = [], [], []
    stall_cycles = backpressure_cycles = 0
    # At most every other cycle stalls. Bound includes all intentional bubbles.
    limit = 4 * (config.size + config.rows * config.input_interval + sa.drain_cycles() + 8)
    started = perf_counter()
    while len(sa.get_buffer()) < config.rows:
        if cycle >= limit:
            raise RuntimeError("simulation timeout: incomplete outputs")
        stalled = bool(config.stall_every and (cycle + 1) % config.stall_every == 0)
        sa.set_control(weight_en=False, stall=stalled)
        accepted = weight_accepted = False
        before = len(sa.get_buffer())
        credits_before = sa.credits
        if stalled:
            stall_cycles += 1
        elif loaded < config.size:
            # Shift register: last output column must be pushed first.
            weight_accepted = sa.enqueue_weights(w[:, config.size - 1 - loaded].tolist())
            if not weight_accepted:
                raise RuntimeError("weight input unexpectedly rejected")
            sa.set_control(weight_en=True)
            loaded += 1
            if loaded == config.size:
                setup_end = cycle + 1
        elif sent < config.rows and (last_input is None or cycle - last_input >= config.input_interval):
            accepted = sa.enqueue(a[sent].tolist())
            if accepted:
                sent += 1
                last_input = cycle
                accepted_cycles.append(cycle)
                if first_input is None:
                    first_input = cycle
            else:
                backpressure_cycles += 1
        sa.tick(float(cycle))
        produced = len(sa.get_buffer()) - before
        output_cycles.extend([cycle] * produced)
        if collect_trace:
            trace.append(dict(cycle=cycle, stall=int(stalled), weight_accepted=int(weight_accepted),
                              input_accepted=int(accepted), outputs=produced,
                              credits_before=credits_before, credits_after=sa.credits))
        cycle += 1

    wall_seconds = perf_counter() - started
    output = np.asarray(sa.get_buffer(), dtype=np.float32)
    passed = np.array_equal(output, expected, equal_nan=True)
    # Mathematically useful operations, not the upstream's internal counters.
    multiplies = config.rows * config.size * config.size
    adds = config.rows * config.size * (config.size - 1)
    elem_bytes = 4 if config.dtype == "fp32" else 2
    payload_bytes = elem_bytes * (a.size + w.size + output.size)
    compute_cycles = cycle - setup_end
    report = dict(schema_version=1, architecture="original_dense_meissa",
                  config=asdict(config), workload_source="external" if external else "generated",
                  native_used=sa.use_native, correctness_passed=passed,
                  max_absolute_error=(float(np.max(np.abs(output.astype(float) - expected.astype(float))))
                                      if np.isfinite(output).all() else None),
                  pipeline_depth=sa.pipeline_depth, total_cycles=cycle,
                  weight_setup_cycles=setup_end, compute_and_drain_cycles=compute_cycles,
                  first_input_cycle=first_input, first_output_cycle=output_cycles[0],
                  last_output_cycle=output_cycles[-1],
                  first_row_latency_cycles=output_cycles[0] - first_input + 1,
                  input_cycles=accepted_cycles, output_cycles=output_cycles,
                  stall_cycles=stall_cycles, input_backpressure_cycles=backpressure_cycles,
                  useful_multiplications=multiplies, useful_binary_additions=adds,
                  dense_equivalent_flops=2 * multiplies,
                  dense_equivalent_flops_per_total_cycle=2 * multiplies / cycle,
                  output_rows_per_compute_cycle=config.rows / compute_cycles,
                  uncompressed_payload_bytes=int(payload_bytes),
                  weight_nonzeros=int(np.count_nonzero(w)),
                  weight_elements=int(w.size), simulator_wall_seconds=wall_seconds)
    return Result(report, output, expected, trace)

