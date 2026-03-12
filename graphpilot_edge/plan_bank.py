from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import CandidatePlan


@dataclass(frozen=True)
class PlanBank:
    state_ids: tuple[str, ...]

    def require_state(self, state_id: str) -> None:
        if state_id not in self.state_ids:
            raise KeyError(f"Unknown plan-bank state '{state_id}'.")


def load_plan_bank(path: Path) -> PlanBank:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PlanBank(
        state_ids=tuple(state["state_id"] for state in payload.get("plan_bank_states", ()))
    )


def select_plan_for_state(
    bank: PlanBank, state_id: str, plans: Iterable[CandidatePlan]
) -> CandidatePlan:
    bank.require_state(state_id)
    plan_list = list(plans)
    if not plan_list:
        raise ValueError("No candidate plans available for selection.")
    if state_id == "lowmem":
        return min(plan_list, key=lambda plan: (plan.total_memory_mb, plan.total_latency_ms, plan.plan_id))
    return min(plan_list, key=lambda plan: (plan.total_latency_ms, plan.total_memory_mb, plan.plan_id))
