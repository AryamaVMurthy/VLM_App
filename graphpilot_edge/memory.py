from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BufferRequest:
    buffer_id: str
    start_ms: int
    end_ms: int
    size_bytes: int
    memory_type: str


@dataclass(frozen=True)
class AllocationResult:
    assignment: dict[str, str]
    peak_bytes: int


@dataclass
class _Block:
    block_id: str
    size_bytes: int
    memory_type: str
    end_ms: int


def allocate_intervals(requests: list[BufferRequest]) -> AllocationResult:
    ordered = sorted(requests, key=lambda req: (req.start_ms, req.end_ms, req.buffer_id))
    active: list[_Block] = []
    free_blocks: list[_Block] = []
    assignment: dict[str, str] = {}
    next_block_id = 0
    peak_bytes = 0

    for request in ordered:
        expired = [block for block in active if block.end_ms <= request.start_ms]
        active = [block for block in active if block.end_ms > request.start_ms]
        free_blocks.extend(expired)

        compatible = [
            block
            for block in free_blocks
            if block.memory_type == request.memory_type and block.size_bytes >= request.size_bytes
        ]
        if compatible:
            block = min(compatible, key=lambda item: (item.size_bytes, item.block_id))
            free_blocks.remove(block)
        else:
            next_block_id += 1
            block = _Block(
                block_id=f"block_{next_block_id}",
                size_bytes=request.size_bytes,
                memory_type=request.memory_type,
                end_ms=request.end_ms,
            )
        block.end_ms = request.end_ms
        assignment[request.buffer_id] = block.block_id
        active.append(block)
        live_bytes = sum(item.size_bytes for item in active)
        peak_bytes = max(peak_bytes, live_bytes)

    return AllocationResult(assignment=assignment, peak_bytes=peak_bytes)
