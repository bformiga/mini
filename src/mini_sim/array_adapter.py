"""Compatibility corrections kept outside the unmodified upstream files."""

from systolic_array.systolic_array_meissa import SystolicArrayMEISSA


class BenchmarkMEISSA(SystolicArrayMEISSA):
    def _run_native(self, cycles):
        produced = super()._run_native(cycles)
        # atalla_meissa_run already returns credits for retired output rows.
        # Upstream then calls _emit(), which returns them a second time in
        # the Python mirror. Restore the authoritative kernel value so the
        # next _sync_to_native cannot write inflated credits back to C++.
        self.credits = self._state.credits
        return produced

