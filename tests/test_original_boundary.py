"""Dense boundary timing is compared with the original extracted bridge."""

import numpy as np
import pytest
from mini_sim.benchmark import Config,run
from mini_sim.original_bridge import GSAUTPUBridge
from systolic_array.systolic_array_meissa import SystolicArrayMEISSA


@pytest.mark.parametrize("size",[4,8,16])
@pytest.mark.parametrize("mixed",[False,True])
@pytest.mark.parametrize("dtype",["bf16","fp16"])
def test_original_bridge_same_boundary_traffic_same_outputs_and_cycles(size,mixed,dtype):
    rng=np.random.default_rng(7)
    a=rng.uniform(-1,1,(3*size,size))
    w=rng.uniform(-1,1,(size,size))
    cfg=Config(size=size,rows=len(a),dtype=dtype,mixed=mixed)
    mini=run(cfg,a,w,collect_trace=True)
    original=SystolicArrayMEISSA(size,dtype=dtype,use_mixed_adder=mixed,use_native=False)
    class Endpoint:
        vector_len=size
        request=None
        def pop_systolic_request(self):
            req=self.request
            self.request=None
            return req
        def push_systolic_response(self,response):
            return True
    endpoint=Endpoint()
    bridge=GSAUTPUBridge(endpoint,original)
    sent=loaded=seen=0
    arrival=[]
    for row in mini.trace:
        if row["weight_accepted"]:
            endpoint.request=dict(vdata=w[:,size-1-loaded].tolist(),dtype=dtype,is_weight=True)
            loaded+=1
        elif row["input_accepted"]:
            endpoint.request=dict(vdata=a[sent].tolist(),dtype=dtype,meta={"row":sent})
            sent+=1
        bridge.tick(float(row["cycle"]))
        produced=len(original.get_buffer())-seen
        arrival.extend([row["cycle"]]*produced)
        seen+=produced
        assert original.credits==row["credits_after"]
    np.testing.assert_array_equal(original.get_buffer(),mini.output)
    assert arrival==mini.report["output_cycles"]
