"""Dense/sparse tile experiments with an optional original-Atalla memory path."""

from dataclasses import asdict, dataclass, replace
from time import perf_counter
import numpy as np

from .benchmark import Config, Result, run as run_dense
from .array_adapter import BenchmarkMEISSA
from .structured_array import StructuredMEISSA
from .sparse_codec import (compress_2of4, decompress_2of4, pack_sparse, unpack_sparse,
                           encode_numbers, decode_numbers, validate_compressed)
from .reference import reference_product, sparse_reference_product
from .memory_path import transfer_payload


@dataclass(frozen=True)
class ExperimentConfig(Config):
    mode: str = "dense"
    load_path: str = "direct"
    mux_latency: int = 0
    output_read_interval: int = 1
    dram_latency: int = 6
    dram_burst_bytes: int = 8
    memory_queue_depth: int = 16
    spad_read_latency: int = 2
    spad_write_latency: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.mode not in ("dense", "sparse") or self.load_path not in ("direct", "atalla"):
            raise ValueError("mode must be dense/sparse; load_path must be direct/atalla")
        if self.mux_latency < 0 or (self.mode == "dense" and self.mux_latency):
            raise ValueError("mux_latency must be nonnegative and applies only to sparse mode")
        if self.mode == "sparse" and self.native:
            raise ValueError("--native is supported only for the dense array")
        if min(self.output_read_interval,self.dram_latency,self.dram_burst_bytes,
               self.memory_queue_depth,self.spad_read_latency,self.spad_write_latency) < 1:
            raise ValueError("memory timing, widths, queue depth and output interval must be positive")


def _with_consumer(base):
    """Separate output availability from consumption and credit return."""
    class ConsumerArray(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.consumed = 0
        def _emit(self,row):
            credits = self.credits
            super()._emit(row)
            self.credits = credits  # Return only when the harness reads the row.
        def consume_one(self):
            if self.consumed == len(self.output_rows):
                return False
            self.consumed += 1
            self.credits = min(self.max_credits,self.credits+1)
            return True
    return ConsumerArray


def run_experiment(config=ExperimentConfig(), activations=None, weights=None, *,
                   values=None, indices=None, collect_trace=False):
    # Preserve existing dense benchmark behavior exactly in its default mode.
    if config.mode == "dense" and config.load_path == "direct" and config.output_read_interval == 1:
        if values is not None or indices is not None:
            raise ValueError("compressed inputs require --mode sparse")
        result = run_dense(config,activations,weights,collect_trace=collect_trace)
        result.report.update(timing_scope="array_boundary",rtl_timing_validated=False,
                             full_atalla_workflow_equivalent=False,
                             physical_shape=[config.size,config.size],
                             logical_weight_shape=[config.size,config.size])
        return result
    if config.native and config.output_read_interval != 1:
        raise ValueError("explicit output-consumer backpressure requires the Python backend")
    if (values is None) != (indices is None):
        raise ValueError("provide both compressed values and indices")
    logical_k = config.size*(2 if config.mode == "sparse" else 1)
    if activations is None:
        if weights is not None or values is not None:
            raise ValueError("external weights require activations")
        rng = np.random.default_rng(config.seed)
        a = rng.uniform(-1,1,(config.rows,logical_k)).astype(np.float32)
        w = rng.uniform(-1,1,(logical_k,config.size)).astype(np.float32)
        if config.mode == "sparse" or config.weight_pattern == "2of4":
            from .workloads import sparsity_2_4
            w = sparsity_2_4(w,"K")
    else:
        a = np.asarray(activations,dtype=np.float32)
        if values is not None:
            v, ix = validate_compressed(values,indices)
            w = decompress_2of4(v,ix)
            if weights is not None and not np.array_equal(w,np.asarray(weights,dtype=np.float32)):
                raise ValueError("W disagrees with compressed values/indices")
        elif weights is not None:
            w = np.asarray(weights,dtype=np.float32)
        else:
            raise ValueError("external activations require W or compressed weights")
    if a.shape != (config.rows,logical_k) or w.shape != (logical_k,config.size):
        raise ValueError(f"expected A[{config.rows},{logical_k}], W[{logical_k},{config.size}] for {config.mode} mode")
    if not np.isfinite(a).all() or not np.isfinite(w).all():
        raise ValueError("inputs must be finite")
    if config.mode == "sparse":
        if values is None:
            v,ix = compress_2of4(w)
        weight_payload = pack_sparse(v,ix,config.dtype)
    else:
        weight_payload = encode_numbers(w.T,config.dtype)
    act_payload = encode_numbers(a,config.dtype)
    cycle = 0
    trace = []
    memory_report = dict(cycles=0, useful_bytes=0, transferred_bytes=0, issued_bursts=0)
    started = perf_counter()
    if config.load_path == "atalla":
        loaded,memory_report,trace = transfer_payload(weight_payload+act_payload,config,
                                                    collect_trace=collect_trace)
        cycle = memory_report["cycles"]
        weight_payload,act_payload = loaded[:len(weight_payload)],loaded[len(weight_payload):]
    # The array is fed from the data read back through memory, not the originals.
    a = decode_numbers(act_payload,a.shape,config.dtype)
    dtype = None if config.dtype == "fp32" else config.dtype
    kwargs = dict(dtype=dtype,use_mixed_adder=config.mixed,mul_latency=config.mul_latency,
                  add2_latency=config.add2_latency,add4_latency=config.add4_latency,use_native=config.native)
    if config.mode == "sparse":
        v,ix = unpack_sparse(weight_payload,(config.size,config.size),config.dtype)
        array_cls = StructuredMEISSA
        kwargs["mux_latency"] = config.mux_latency
        expected = sparse_reference_product(a,v,ix,dtype=dtype,mixed=config.mixed)
    else:
        w = decode_numbers(weight_payload,(config.size,config.size),config.dtype).T
        array_cls = BenchmarkMEISSA
        expected = reference_product(a,w,dtype=dtype,mixed=config.mixed)
    if not np.isfinite(expected).all():
        raise ValueError("workload overflows selected arithmetic")
    explicit_consumer = config.output_read_interval != 1
    if explicit_consumer:
        array_cls = _with_consumer(array_cls)
    sa = array_cls(config.size,**kwargs)
    if config.native and not sa.use_native:
        raise RuntimeError("native kernel unavailable; run make native or omit --native")
    array_start = cycle
    sent = loaded_cols = seen = consumed = 0
    first_input = last_input = setup_end = None
    inputs, outputs, reads = [], [], []
    stalls = backpressure = 0
    limit = cycle + 8*(config.size + config.rows*(config.input_interval+config.output_read_interval)
                      + sa.drain_cycles()+8)
    while consumed < config.rows:
        if cycle > limit:
            raise RuntimeError("array simulation timed out")
        relative = cycle-array_start
        stalled = bool(config.stall_every and (relative+1)%config.stall_every == 0)
        sa.set_control(weight_en=False,stall=stalled)
        accepted = weight_accepted = read = False
        before = sa.credits
        # Explicit consumer reads at most one already-available row before tick.
        if explicit_consumer and relative % config.output_read_interval == 0:
            read = sa.consume_one()
            if read:
                consumed += 1
                reads.append(cycle)
        if stalled:
            stalls += 1
        elif loaded_cols < config.size:
            col = config.size-1-loaded_cols
            if config.mode == "sparse":
                weight_accepted = sa.enqueue_sparse_weights(v[:,col],ix[:,col])
            else:
                weight_accepted = sa.enqueue_weights(w[:,col].tolist())
            if not weight_accepted:
                raise RuntimeError("weight input unexpectedly rejected")
            sa.set_control(weight_en=True)
            loaded_cols += 1
            if loaded_cols == config.size:
                setup_end = cycle+1
        elif sent < config.rows and (last_input is None or cycle-last_input >= config.input_interval):
            accepted = sa.enqueue(a[sent].tolist())
            if accepted:
                sent += 1
                last_input = cycle
                inputs.append(cycle)
                if first_input is None:
                    first_input = cycle
            else:
                backpressure += 1
        sa.tick(float(cycle))
        produced = len(sa.get_buffer())-seen
        outputs.extend([cycle]*produced)
        seen += produced
        if not explicit_consumer:
            consumed = seen
            reads.extend([cycle]*produced)
        if collect_trace:
            trace.append(dict(cycle=cycle,stage="array",stall=int(stalled),
                weight_accepted=int(weight_accepted),input_accepted=int(accepted),outputs=produced,
                output_read=int(read) if explicit_consumer else produced,
                credits_before=before,credits_after=sa.credits))
        cycle += 1
    output = np.asarray(sa.get_buffer(),dtype=np.float32)
    actual_muls = config.rows*config.size*config.size
    report = dict(schema_version=2,architecture="structured_2of4_meissa" if config.mode == "sparse" else "original_dense_meissa",
        config=asdict(config), correctness_passed=bool(np.array_equal(output,expected)),
        max_absolute_error=float(np.max(np.abs(output.astype(float)-expected.astype(float)))) if np.isfinite(output).all() else None,
        physical_shape=[config.size,config.size],logical_weight_shape=[logical_k,config.size],
        total_cycles=cycle,memory=memory_report,array_cycles=cycle-array_start,
        weight_setup_cycles=setup_end-array_start,compute_and_drain_cycles=cycle-setup_end,
        pipeline_depth=sa.pipeline_depth,first_input_cycle=first_input,
        first_output_cycle=outputs[0],last_output_cycle=outputs[-1],
        first_row_latency_cycles=outputs[0]-first_input+1,input_cycles=inputs,
        output_cycles=outputs,output_read_cycles=reads,stall_cycles=stalls,
        input_backpressure_cycles=backpressure,executed_multiplications=actual_muls,
        dense_equivalent_flops=2*config.rows*logical_k*config.size,
        weight_payload_bytes=len(weight_payload),activation_payload_bytes=len(act_payload),
        metadata_bytes=(config.size*config.size+3)//4 if config.mode == "sparse" else 0,
        native_used=sa.use_native,simulator_wall_seconds=perf_counter()-started,
        timing_scope="serial_memory_frontend_then_array" if config.load_path == "atalla" else "array_boundary",
        rtl_timing_validated=False,full_atalla_workflow_equivalent=False,
        assumptions=dict(same_physical_multiplier_count=True,mux_latency_cycles=config.mux_latency,
            sparse_encoding="column_major_values_then_packed_2bit_indices",
            memory_overlap=False,vrf_vlsu_gsau_timing_included=False,
            output_sink="periodic_consumer" if explicit_consumer else "always_ready"))
    return Result(report,output,expected,trace)


def run_rectangular(config,a,w,*,row_tile=64):
    """Serial full GEMM tiling; software FP32 accumulation is explicitly untimed."""
    a,w = np.asarray(a,dtype=np.float32),np.asarray(w,dtype=np.float32)
    if a.ndim != 2 or w.ndim != 2 or a.shape[1] != w.shape[0] or min(*a.shape,*w.shape)<1:
        raise ValueError("expected nonempty A[M,K] and W[K,N]")
    if row_tile < 1:
        raise ValueError("row_tile must be positive")
    if config.mode == "sparse":
        compress_2of4(w)  # Validate original groups before zero padding.
    m,k = a.shape
    n = w.shape[1]
    ks = config.size*(2 if config.mode == "sparse" else 1)
    out,ref = np.zeros((m,n),dtype=np.float32),np.zeros((m,n),dtype=np.float32)
    cycles = total_muls = weight_bytes = metadata_bytes = bursts = memory_cycles = 0
    tiles = 0
    tile_checks = True
    for i in range(0,m,row_tile):
        nr = min(row_tile,m-i)
        for j in range(0,n,config.size):
            nc = min(config.size,n-j)
            for q in range(0,k,ks):
                nk = min(ks,k-q)
                at = np.zeros((nr,ks),dtype=np.float32)
                wt = np.zeros((ks,config.size),dtype=np.float32)
                at[:,:nk] = a[i:i+nr,q:q+nk]
                wt[:nk,:nc] = w[q:q+nk,j:j+nc]
                r = run_experiment(replace(config,rows=nr),at,wt)
                out[i:i+nr,j:j+nc] += r.output[:,:nc]
                ref[i:i+nr,j:j+nc] += r.expected[:,:nc]
                tile_checks &= r.report["correctness_passed"]
                cycles += r.report["total_cycles"]
                total_muls += r.report.get("executed_multiplications",r.report.get("useful_multiplications",0))
                weight_bytes += r.report.get("weight_payload_bytes",config.size**2*(4 if config.dtype=="fp32" else 2))
                metadata_bytes += r.report.get("metadata_bytes",0)
                memory_cycles += r.report.get("memory",{}).get("cycles",0)
                bursts += r.report.get("memory",{}).get("issued_bursts",0)
                tiles += 1
    return Result(dict(schema_version=2,architecture="structured_2of4_meissa" if config.mode=="sparse" else "original_dense_meissa",
        config=asdict(config),shape_MNK=[m,n,k],physical_shape=[config.size,config.size],tiles=tiles,
        correctness_passed=bool(tile_checks and np.array_equal(out,ref)),
        serial_tile_cycles=cycles,total_cycles=None,memory_cycles=memory_cycles,issued_dram_bursts=bursts,
        executed_multiplications=total_muls,weight_payload_bytes=weight_bytes,metadata_bytes=metadata_bytes,
        max_absolute_error=float(np.max(np.abs(out.astype(float)-ref.astype(float)))),
        timing_scope="sum_of_serial_tile_measurements_excludes_accumulation",
        accumulation="software_fp32_sequential_K_tiles",accumulation_cycles_included=False,
        rtl_timing_validated=False,full_atalla_workflow_equivalent=False),out,ref,[])
