"""Original Atalla GSAU/array bridge, extracted for boundary-equivalence tests.

Class body copied without edits from the supplied sysarr_tpu_system.py.
The surrounding imports and FP16 packing helper allow standalone use.
"""
from __future__ import annotations
from typing import Optional, List, Dict
import numpy as np
from base.clocked_object import Clocked

def _fp16_bits(value):
    return int(np.asarray([value], dtype=np.float16).view(np.uint16)[0])

class GSAUTPUBridge(Clocked):
    """Connects the GSAU to the systolic array.

    Each cycle it takes at most one request from the GSAU -- a weight row
    (shifted into the array) or an activation row (streamed through it) -- and
    returns each output row to the GSAU with its request's metadata, skipping
    the array's warm-up rows. After finish_inputs() it pushes zero rows to
    flush the last outputs out. _pack_rsp sets the result format (FP16 bit
    patterns here; ArrayValueBridge in atalla_platform.py returns values).
    """
    def __init__(self, vc: VectorCore, sa: SystolicArrayTPU, mirror: Optional[TPUReference] = None):
        super().__init__()
        self.vc = vc
        self.sa = sa
        self.mirror = mirror
        self.size = sa.size
        self._pending_meta: List[Dict] = []
        self._out_read_idx = 0
        self._warmup = self.sa.warmup_cycles()
        self._flush_pending = 0
        self._zero_row = [0.0] * self.size
        self._input_done = False
        self.now = 0
        self.trace_hook = None

    def finish_inputs(self) -> None:
        self._input_done = True
        self._flush_pending = self.sa.flush_cycles()
        if self.mirror is not None:
            self.mirror.finish_inputs()

    def _pack_rsp(self, out_row: List[float], meta: Dict) -> Dict:
        vec = [0.0] * self.vc.vector_len
        for i, val in enumerate(out_row[: self.size]):
            vec[i] = _fp16_bits(val)
        return {"vdata": vec, "meta": dict(meta), "dtype": meta.get("dtype")}

    def tick(self, time: Optional[float] = None) -> None:
        if time is not None:
            self.now = time

        self.sa.set_control(weight_en=False, mac_shift=False, start=False, stall=False)

        req = self.vc.pop_systolic_request()
        did_req = False
        did_flush = False
        row = self._zero_row
        is_weight = False
        expect_output = False
        if req is not None:
            vdata = [float(x) for x in req.get("vdata", [])]
            row = vdata[: self.size]
            if len(row) < self.size:
                row += [0.0] * (self.size - len(row))

            if bool(req.get("is_weight", False)):
                did_req = True
                is_weight = True
                assert self.sa.enqueue_weights(row, dtype=req.get("dtype"))
                self.sa.set_control(weight_en=True, mac_shift=False, start=False, stall=False)
            else:
                did_req = True
                expect_output = bool(req.get("expect_output", True))
                assert self.sa.enqueue(row, dtype=req.get("dtype"))
                assert self.sa.enqueue_psums([0.0] * self.size, dtype=req.get("dtype"))
                self.sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)
                if expect_output:
                    self._pending_meta.append(dict(req.get("meta", {})))
            if self.trace_hook is not None:
                self.trace_hook(
                    {
                        "stage": "request",
                        "cycle": int(self.now),
                        "is_weight": is_weight,
                        "expect_output": expect_output,
                        "meta": dict(req.get("meta", {}) or {}),
                    }
                )
        elif self._input_done and self._flush_pending > 0:
            if self.sa.enqueue(self._zero_row, dtype=self.sa.dtype, count_algo=False):
                did_flush = True
                assert self.sa.enqueue_psums([0.0] * self.size, dtype=self.sa.dtype)
                self.sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)
                self._flush_pending -= 1

        if self.mirror is not None:
            self.mirror.tick_from_bridge(
                did_req=did_req,
                row=row,
                is_weight=is_weight,
                expect_output=expect_output,
                did_flush=did_flush,
                time=time,
            )

        self.sa.tick(time)

        buf = self.sa.get_buffer()
        while self._out_read_idx < len(buf):
            if self._out_read_idx < self._warmup:
                self._out_read_idx += 1
                continue
            if not self._pending_meta:
                break
            out_row = [float(x) for x in buf[self._out_read_idx]]
            meta = self._pending_meta.pop(0)
            if self.trace_hook is not None:
                self.trace_hook(
                    {
                        "stage": "response",
                        "cycle": int(self.now),
                        "meta": dict(meta),
                    }
                )
            assert self.vc.push_systolic_response(self._pack_rsp(out_row, meta))
            self._out_read_idx += 1


