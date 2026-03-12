from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class KvSession:
    session_id: str
    kv_bytes: int


@dataclass(frozen=True)
class DegradationAction:
    action_id: str
    freed_bytes: int
    added_latency_ms: float
    quality_loss: float


def can_admit_session(
    sessions: Iterable[KvSession],
    workbuf_bytes: int,
    other_live_bytes: int,
    budget_bytes: int,
    margin_bytes: int,
) -> bool:
    total = sum(session.kv_bytes for session in sessions) + workbuf_bytes + other_live_bytes
    return total <= budget_bytes - margin_bytes


def choose_degradation_action(
    actions: Iterable[DegradationAction], latency_weight: float
) -> DegradationAction:
    action_list = list(actions)
    if not action_list:
        raise ValueError("At least one degradation action is required.")
    return min(
        action_list,
        key=lambda action: (
            (action.quality_loss + latency_weight * action.added_latency_ms)
            / max(action.freed_bytes, 1),
            action.quality_loss,
            action.added_latency_ms,
            action.action_id,
        ),
    )
