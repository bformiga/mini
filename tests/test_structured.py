from dataclasses import replace
import itertools
import json
import numpy as np
import pytest

from mini_sim.experiment import ExperimentConfig,run_experiment,run_rectangular
from mini_sim.sparse_codec import (compress_2of4,decompress_2of4,pack_sparse,unpack_sparse)
from mini_sim.workloads import sparsity_2_4,generate_benchmark
from mini_sim.memory_path import transfer_payload
from mini_sim.cli import main


@pytest.mark.parametrize("positions",list(itertools.combinations(range(4),2)))
def test_each_of_the_six_patterns_and_column_specific_selection(positions):
    a = np.asarray([[1,2,3,4],[4,3,2,1]],dtype=np.float32)
    w = np.zeros((4,2),dtype=np.float32)
    w[list(positions),0] = [2,3]
    other = tuple(p for p in range(4) if p not in positions)
    w[list(other),1] = [5,-1]
    r = run_experiment(ExperimentConfig(mode="sparse",size=2,rows=2,dtype="fp32"),a,w)
    np.testing.assert_array_equal(r.output,a@w)
    assert r.report["correctness_passed"]


@pytest.mark.parametrize("dtype",["bf16","fp16","fp32"])
@pytest.mark.parametrize("size",[2,4,8,16])
@pytest.mark.parametrize("mixed",[False,True])
def test_sparse_fractional_arithmetic_and_stalls(dtype,size,mixed):
    cfg = ExperimentConfig(mode="sparse",size=size,rows=2*size+9,dtype=dtype,mixed=mixed)
    original = run_experiment(cfg)
    delayed = run_experiment(replace(cfg,mux_latency=2,stall_every=3,
                                    output_read_interval=7,input_interval=1),collect_trace=True)
    assert original.report["correctness_passed"] and delayed.report["correctness_passed"]
    np.testing.assert_array_equal(original.output,delayed.output)
    assert delayed.report["pipeline_depth"] == original.report["pipeline_depth"]+2
    assert delayed.report["total_cycles"] > original.report["total_cycles"]
    assert len(delayed.report["output_read_cycles"]) == cfg.rows
    assert delayed.report["input_backpressure_cycles"] > 0


@pytest.mark.parametrize("dtype",["bf16","fp16","fp32"])
def test_compressed_round_trip_and_wire_bytes(dtype):
    w=np.zeros((8,4),dtype=np.float32)
    w[1::4]=2
    w[3::4]=-4
    v,ix=compress_2of4(w)
    wire=pack_sparse(v,ix,dtype)
    expected=16*(4 if dtype=="fp32" else 2)+4
    assert len(wire)==expected
    vv,ii=unpack_sparse(wire,v.shape,dtype)
    np.testing.assert_array_equal(ix,ii)
    np.testing.assert_array_equal(decompress_2of4(vv,ii),w)


def test_zero_groups_are_encodable_without_inventing_nonzero_values():
    w=np.zeros((8,3),dtype=np.float32)
    w[5,1]=7
    v,ix=compress_2of4(w)
    np.testing.assert_array_equal(decompress_2of4(v,ix),w)
    assert np.all(ix[::2]!=ix[1::2])


def test_wire_order_with_distinct_masks_in_every_column():
    rng=np.random.default_rng(23)
    w=sparsity_2_4(rng.integers(-8,9,(16,8)).astype(np.float32),"K")
    v,ix=compress_2of4(w)
    vv,ii=unpack_sparse(pack_sparse(v,ix,"bf16"),v.shape,"bf16")
    np.testing.assert_array_equal(ii,ix)
    np.testing.assert_array_equal(decompress_2of4(vv,ii),w)


def test_memory_and_direct_paths_compute_identical_fractional_results():
    cfg=ExperimentConfig(mode="sparse",size=8,rows=19,dtype="bf16",mixed=True)
    direct=run_experiment(cfg)
    memory=run_experiment(replace(cfg,load_path="atalla"))
    np.testing.assert_array_equal(direct.output,memory.output)
    assert memory.report["total_cycles"]==direct.report["total_cycles"]+memory.report["memory"]["cycles"]


def test_rejects_nonstructured_weights_bad_indices_and_native():
    with pytest.raises(ValueError,match="violate"):
        compress_2of4(np.ones((4,2)))
    with pytest.raises(ValueError,match="distinct"):
        decompress_2of4(np.ones((2,2)),np.zeros((2,2),dtype=int))
    with pytest.raises(ValueError,match="0..3"):
        decompress_2of4(np.ones((2,2)),np.full((2,2),4))
    with pytest.raises(ValueError,match="native"):
        ExperimentConfig(mode="sparse",native=True)


@pytest.mark.parametrize("length",[1,9,36,64,65,128,135])
def test_original_memory_models_move_payloads_without_host_side_bypass(length):
    data=bytes((i*17)%256 for i in range(length))
    cfg=ExperimentConfig()
    got,report,trace=transfer_payload(data,cfg,collect_trace=True)
    assert got==data
    assert report["useful_bytes"]==length
    assert report["transferred_bytes"]==length+length%2
    assert report["issued_bursts"] == (length+length%2+7)//8
    assert len(trace)==report["cycles"]
    assert report["scratchpad_stats"]["tiles"][0]["reads"] == (length+length%2+63)//64


def test_memory_timing_and_metadata_are_measured():
    cfg=ExperimentConfig(mode="sparse",size=4,rows=5,load_path="atalla")
    r=run_experiment(cfg)
    delayed=run_experiment(replace(cfg,dram_latency=30))
    assert delayed.report["total_cycles"]>r.report["total_cycles"]
    np.testing.assert_array_equal(delayed.output,r.output)
    assert r.report["metadata_bytes"]==4 and r.report["weight_payload_bytes"]==36
    assert r.report["memory"]["useful_bytes"]==36+5*8*2
    assert r.report["total_cycles"]==r.report["memory"]["cycles"]+r.report["array_cycles"]
    assert r.report["full_atalla_workflow_equivalent"] is False


@pytest.mark.parametrize("load_path",["direct","atalla"])
def test_rectangular_gemm_same_problem_dense_and_sparse(load_path):
    a=np.arange(5*12,dtype=np.float32).reshape(5,12)%4
    w=sparsity_2_4(np.arange(12*7,dtype=np.float32).reshape(12,7)%5,"K")
    dense=run_rectangular(ExperimentConfig(size=4,dtype="fp32",load_path=load_path),a,w,row_tile=3)
    sparse=run_rectangular(ExperimentConfig(mode="sparse",size=4,dtype="fp32",load_path=load_path),a,w,row_tile=3)
    for r in (dense,sparse):
        np.testing.assert_array_equal(r.output,a@w)
        assert r.report["correctness_passed"]
        assert r.report["total_cycles"] is None
        assert r.report["accumulation_cycles_included"] is False
    assert sparse.report["tiles"]<dense.report["tiles"]
    assert sparse.report["executed_multiplications"]<dense.report["executed_multiplications"]


def test_generator_cli_memory_sparse_and_rectangular_flow(tmp_path,capsys):
    generate_benchmark(tmp_path,workloads={"tiny":(5,8,12)},tile_size=4,tile_rows=3)
    package=tmp_path/"tiles/tiny_structured.npz"
    with np.load(package) as p:
        assert p["A"].shape==(3,8) and p["W_values"].shape==(4,4)
    report=tmp_path/"report.json"
    trace=tmp_path/"trace.csv"
    assert main(["--input",str(package),"--load-path","atalla","--json",str(report),"--trace",str(trace)])==0
    r=json.loads(report.read_text())
    assert r["config"]["mode"]=="sparse" and r["config"]["size"]==4
    assert "memory" in trace.read_text() and "array" in trace.read_text()
    assert main(["--input",str(tmp_path/"full/tiny_sparse.npz"),"--full","--size","4",
                 "--json",str(report)])==0
    r=json.loads(report.read_text())
    assert r["shape_MNK"]==[5,8,12] and r["correctness_passed"]
    capsys.readouterr()

