"""Original Atalla memory components with a small serial tile-load harness.

This restores actual DRAM bursts, DMA queues, SRAM writes and frontend reads.
It does not claim the scheduler/VLSU/VRF/GSAU timing or original tile policy.
"""

from memory.dram import DRAM
from memory.backend import Backend
from memory.scratchpad import Scratchpad


def transfer_payload(payload, config, *, start_cycle=0, collect_trace=False):
    if not payload:
        return payload, dict(cycles=0, useful_bytes=0, transferred_bytes=0, issued_bursts=0), []
    padded = payload + b"\0" * (len(payload) % 2)
    width = 64  # Original 32 banks, 16 bits per bank.
    rows = (len(padded)+width-1)//width
    spad = Scratchpad(num_banks=32, bank_size=max(32, rows), num_tiles=1,
        elem_bytes=2, queue_depth=config.memory_queue_depth,
        read_latency=config.spad_read_latency, write_latency=config.spad_write_latency)
    dram = DRAM()
    dram.write(0, padded)
    backend = Backend(dram_latency=config.dram_latency, dram_q_depth=config.memory_queue_depth,
        dram_burst_bytes=config.dram_burst_bytes, elem_bytes=2, delay_cycles=1)
    spad.attach_backend(backend, tile_id=0)
    backend.attach_dram(dram)
    full, tail = divmod(len(padded), width)
    jobs = []
    if full:
        jobs.append((0, 0, full, 32))
    if tail:
        jobs.append((full, full*width, 1, tail//2))
    completed = set()
    issued = read_issued = 0
    received = {}
    trace = []
    cycle = start_cycle
    dma_end = None
    limit = 1000 + 8*(rows+len(padded))*(config.dram_latency + config.spad_read_latency + 2)
    while len(received) != rows:
        if cycle-start_cycle > limit:
            raise RuntimeError("original-memory load timed out")
        if issued < len(jobs):
            slot, address, nr, cols = jobs[issued]
            tx = backend.driver_to_backend_start_load(slot, address, nr, cols,
                callback=lambda tx: completed.add(tx))
            if tx >= 0:
                issued += 1
        reading = False
        if len(completed) == len(jobs) and issued == len(jobs):
            if dma_end is None:
                dma_end = cycle
            if read_issued < rows:
                slot = read_issued
                def receive(lanes, slot=slot):
                    received[slot] = b"".join(lanes).ljust(width, b"\0")
                reading = spad.frontends[0].read(slot, 0, receive, now=cycle)
                if reading:
                    read_issued += 1
        # Same memory phase order as the original platform: spad then backend.
        spad.tick(float(cycle))
        backend.tick(float(cycle))
        if collect_trace:
            trace.append(dict(cycle=cycle, stage="memory", frontend_read_accepted=int(reading),
                              dma_transactions_completed=len(completed), memory_rows_returned=len(received)))
        cycle += 1
    output = b"".join(received[r] for r in range(rows))[:len(payload)]
    if output != payload:
        raise RuntimeError("original-memory payload readback differs from DRAM input")
    report = dict(cycles=cycle-start_cycle, dma_cycles=dma_end-start_cycle,
        frontend_read_cycles=cycle-dma_end, useful_bytes=len(payload),
        transferred_bytes=len(padded), issued_bursts=backend.total_dram_bursts_issued,
        burst_slot_capacity_bytes=backend.total_dram_bursts_issued*config.dram_burst_bytes,
        backend_stalls=backend.total_backend_stalls, scratchpad_stats=spad.get_stats())
    return output, report, trace

