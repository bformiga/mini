"""Diagram-based 2:4 MEISSA extension, with unchanged dense arithmetic helpers.

S physical multiplier rows represent 2*S logical K positions. Each pair of
physical rows selects two of four activations independently in each column.
Mux latency is an explicit uncalibrated parameter, not a claimed RTL value.
"""

import numpy as np
from .array_adapter import BenchmarkMEISSA
from .sparse_codec import validate_compressed


class StructuredMEISSA(BenchmarkMEISSA):
    def __init__(self, size, *, mux_latency=0, **kwargs):
        if size < 2 or size % 2:
            raise ValueError("structured MEISSA requires an even physical row count")
        if kwargs.get("use_native", False):
            raise ValueError("sparse native execution is not implemented")
        if mux_latency < 0:
            raise ValueError("mux latency must be nonnegative")
        # For stationary weights, an extra mux stage is a per-column product
        # delay. Reuse the upstream pipeline storage and reduction semantics.
        kwargs["mul_latency"] = kwargs.get("mul_latency", 1) + mux_latency
        super().__init__(size, **kwargs)
        self.mux_latency = mux_latency
        self._act = np.zeros((2*size, size), dtype=np.float32)
        self.indices = np.tile(np.arange(size, dtype=np.uint8)[:, None] % 2, (1, size))
        self._pending_indices = None

    def enqueue(self, activations, dtype=None, *, count_algo=True):
        if len(activations) != 2*self.size:
            raise ValueError("sparse activations must contain 2*size logical K values")
        if not self.can_accept():
            return False
        self._pending_act = self._cast(activations, dtype)
        self._pending_count_algo = bool(count_algo)
        self._drained = False
        if count_algo:
            self.algo_counts["act_rows"] += 1
        return True

    def enqueue_weights(self, weights, dtype=None, *, count_algo=True):
        raise ValueError("sparse weights require indices; use enqueue_sparse_weights")

    def enqueue_sparse_weights(self, weights, indices, dtype=None):
        v, ix = validate_compressed(np.asarray(weights)[:, None], np.asarray(indices)[:, None])
        if len(weights) != self.size:
            raise ValueError("compressed column must match physical array rows")
        accepted = super().enqueue_weights(v[:, 0].tolist(), dtype=dtype)
        if accepted:
            self._pending_indices = ix[:, 0].copy()
        return accepted

    def load_weights(self, weights):
        raise ValueError("sparse weights must be streamed with their position indices")

    def _shift_weights(self):
        shifting = self.weight_en and self._pending_wgt is not None
        super()._shift_weights()
        if shifting:
            self.indices[:, 1:] = self.indices[:, :-1]
            self.indices[:, 0] = self._pending_indices
        self._pending_indices = None

    def _advance_trees(self):
        # Each output column can have a different mask. Select per column,
        # rather than incorrectly selecting one activation vector for all W.
        logical = self._act
        addresses = 4*(np.arange(self.size)[:, None]//2) + self.indices
        self._act = logical[addresses, np.arange(self.size)[None, :]]
        try:
            super()._advance_trees()
        finally:
            self._act = logical

