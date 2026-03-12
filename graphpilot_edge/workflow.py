from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from .errors import ValidationError

ALLOWED_STREAM_MODES = {"full", "chunk", "token"}


@dataclass(frozen=True)
class WorkflowEdge:
    source_stage_id: str
    target_stage_id: str
    stream_mode: str

    def __post_init__(self) -> None:
        if self.stream_mode not in ALLOWED_STREAM_MODES:
            raise ValidationError(
                f"Unsupported stream mode '{self.stream_mode}'. Allowed modes: {sorted(ALLOWED_STREAM_MODES)}"
            )


@dataclass(frozen=True)
class WorkflowDag:
    workflow_id: str
    stage_ids: tuple[str, ...]
    edges: tuple[WorkflowEdge, ...]
    chunk_sizes: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.stage_ids:
            raise ValidationError("Workflow DAG must declare at least one stage.")
        if len(set(self.stage_ids)) != len(self.stage_ids):
            raise ValidationError("Workflow DAG contains duplicate stage IDs.")
        stage_set = set(self.stage_ids)
        for edge in self.edges:
            if edge.source_stage_id not in stage_set or edge.target_stage_id not in stage_set:
                raise ValidationError(
                    f"Workflow edge '{edge.source_stage_id}->{edge.target_stage_id}' references an unknown stage."
                )
        seen_chunk_stages: set[str] = set()
        for stage_id, chunk_size in self.chunk_sizes:
            if stage_id in seen_chunk_stages:
                raise ValidationError(
                    f"Workflow DAG '{self.workflow_id}' declares duplicate chunk size entries for stage '{stage_id}'."
                )
            if stage_id not in stage_set:
                raise ValidationError(
                    f"Workflow DAG '{self.workflow_id}' chunk_sizes references an unknown stage '{stage_id}'."
                )
            if chunk_size <= 0:
                raise ValidationError(
                    f"Workflow DAG '{self.workflow_id}' chunk size for stage '{stage_id}' must be > 0."
                )
            seen_chunk_stages.add(stage_id)
        self.topological_order()

    def predecessors(self, stage_id: str) -> tuple[str, ...]:
        self._require_stage(stage_id)
        predecessors = [
            edge.source_stage_id for edge in self.edges if edge.target_stage_id == stage_id
        ]
        return tuple(sorted(predecessors))

    def successors(self, stage_id: str) -> tuple[str, ...]:
        self._require_stage(stage_id)
        successors = [
            edge.target_stage_id for edge in self.edges if edge.source_stage_id == stage_id
        ]
        return tuple(sorted(successors))

    def root_stage_ids(self) -> tuple[str, ...]:
        incoming = {stage_id: 0 for stage_id in self.stage_ids}
        for edge in self.edges:
            incoming[edge.target_stage_id] += 1
        return tuple(stage_id for stage_id in self.stage_ids if incoming[stage_id] == 0)

    def topological_order(self) -> tuple[str, ...]:
        index = {stage_id: pos for pos, stage_id in enumerate(self.stage_ids)}
        incoming = {stage_id: 0 for stage_id in self.stage_ids}
        outgoing: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            incoming[edge.target_stage_id] += 1
            outgoing[edge.source_stage_id].append(edge.target_stage_id)
        ready = deque(sorted(self.root_stage_ids(), key=index.__getitem__))
        order: list[str] = []
        while ready:
            stage_id = ready.popleft()
            order.append(stage_id)
            for successor in sorted(outgoing[stage_id], key=index.__getitem__):
                incoming[successor] -= 1
                if incoming[successor] == 0:
                    ready.append(successor)
        if len(order) != len(self.stage_ids):
            raise ValidationError(f"Workflow DAG '{self.workflow_id}' contains a cycle.")
        return tuple(order)

    def chunk_size(self, stage_id: str) -> int | None:
        self._require_stage(stage_id)
        for current_stage_id, chunk_size in self.chunk_sizes:
            if current_stage_id == stage_id:
                return chunk_size
        return None

    def _require_stage(self, stage_id: str) -> None:
        if stage_id not in self.stage_ids:
            raise ValidationError(
                f"Workflow DAG '{self.workflow_id}' does not contain stage '{stage_id}'."
            )
