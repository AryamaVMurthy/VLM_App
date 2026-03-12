from __future__ import annotations

from .cost_model import (
    ObjectiveWeights,
    SimulationCalibration,
    compute_contention_factor,
    compute_energy_mj,
    compute_execution_time_ms,
    compute_objective_score,
    compute_transfer_cost_ms,
)
from .memory import BufferRequest, allocate_intervals
from .models import (
    CandidatePlan,
    RequestSimulationResult,
    RequestSpec,
    SimulationEvent,
    SimulationResult,
    StageTiming,
    StreamSimulationResult,
)
from .workflow import WorkflowDag


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile))))
    return float(ordered[rank])


def simulate_candidate_plan(
    workflow: WorkflowDag,
    plan: CandidatePlan,
    resource_capacities: dict[str, int],
    *,
    bandwidth_bytes_per_ms: float = 1024.0 * 1024.0,
    transfer_fixed_overhead_ms: float = 0.0,
    transfer_layout_ms: float = 0.0,
    objective_weights: ObjectiveWeights | None = None,
    calibration: SimulationCalibration | None = None,
) -> SimulationResult:
    backends = {option.backend for option in plan.stage_options}
    missing = sorted(backend for backend in backends if backend not in resource_capacities)
    if missing:
        raise ValueError(
            "Missing explicit resource capacity for backend(s): " + ", ".join(missing)
        )

    backend_next_free = {backend: 0 for backend in resource_capacities}
    backend_cold = {backend: True for backend in resource_capacities}
    events: list[SimulationEvent] = []
    stage_timings: dict[str, StageTiming] = {}
    copy_bytes = 0
    copy_time_ms = 0.0
    energy_mj = 0.0

    for stage_id in workflow.topological_order():
        option = plan.option_for_stage(stage_id)
        dependency_ready_ms = 0
        for predecessor in workflow.predecessors(stage_id):
            predecessor_option = plan.option_for_stage(predecessor)
            same_memory = predecessor_option.backend == option.backend
            transfer_ms = compute_transfer_cost_ms(
                size_bytes=predecessor_option.output_bytes,
                src_backend=predecessor_option.backend,
                dst_backend=option.backend,
                same_memory=same_memory,
                bandwidth_bytes_per_ms=bandwidth_bytes_per_ms,
                fixed_overhead_ms=transfer_fixed_overhead_ms,
                layout_ms=transfer_layout_ms,
            )
            dependency_ready_ms = max(
                dependency_ready_ms,
                stage_timings[predecessor].finish_ms + int(round(transfer_ms)),
            )
            if predecessor_option.backend != option.backend:
                copy_bytes += predecessor_option.output_bytes
                copy_time_ms += transfer_ms
        start_ms = max(dependency_ready_ms, backend_next_free[option.backend])
        thermal_factor = 1.0
        contention_factor = 1.0
        if calibration is not None:
            thermal_factor = calibration.backend_thermal_factors.get(
                option.backend, calibration.workflow_thermal_scale
            )
            contention_factor = compute_contention_factor(
                backend=option.backend,
                sensitivities=calibration.contention_sensitivities,
                utilizations=calibration.backend_utilizations,
            )
        exec_ms = compute_execution_time_ms(
            base_latency_ms=float(option.latency_ms),
            thermal_factor=thermal_factor,
            contention_factor=contention_factor,
            compile_cost_ms=float(option.compile_cost_ms),
            cold=backend_cold[option.backend],
        )
        finish_ms = start_ms + int(round(exec_ms))
        backend_next_free[option.backend] = finish_ms
        backend_cold[option.backend] = False
        energy_mj += compute_energy_mj(avg_power_mw=option.average_power_mw, time_ms=exec_ms)

        timing = StageTiming(
            stage_id=stage_id,
            backend=option.backend,
            start_ms=start_ms,
            finish_ms=finish_ms,
        )
        stage_timings[stage_id] = timing
        events.append(
            SimulationEvent(
                stage_id=stage_id,
                backend=option.backend,
                event_type="start",
                timestamp_ms=start_ms,
            )
        )
        events.append(
            SimulationEvent(
                stage_id=stage_id,
                backend=option.backend,
                event_type="finish",
                timestamp_ms=finish_ms,
            )
        )

    makespan_ms = max((timing.finish_ms for timing in stage_timings.values()), default=0)
    orchestration_overhead_ms = int(
        round(calibration.orchestration_overhead_ms)
    ) if calibration is not None else 0
    buffer_requests = []
    for stage_id in workflow.topological_order():
        option = plan.option_for_stage(stage_id)
        successors = workflow.successors(stage_id)
        death_ms = (
            max(stage_timings[successor].start_ms for successor in successors)
            if successors
            else stage_timings[stage_id].finish_ms
        )
        buffer_requests.append(
            BufferRequest(
                buffer_id=stage_id,
                start_ms=stage_timings[stage_id].start_ms,
                end_ms=death_ms,
                size_bytes=option.output_bytes or (option.memory_mb * 1024 * 1024),
                memory_type=option.backend,
            )
        )
    allocation = allocate_intervals(buffer_requests)
    ttft_ms = None
    if "responder.primary" in stage_timings:
        responder = plan.option_for_stage("responder.primary")
        ttft_stage_ms = responder.ttft_ms if responder.ttft_ms is not None else responder.latency_ms
        if calibration is not None:
            ttft_stage_ms = int(
                round(
                    compute_execution_time_ms(
                        base_latency_ms=float(ttft_stage_ms),
                        thermal_factor=calibration.backend_thermal_factors.get(
                            responder.backend, calibration.workflow_thermal_scale
                        ),
                        contention_factor=compute_contention_factor(
                            backend=responder.backend,
                            sensitivities=calibration.contention_sensitivities,
                            utilizations=calibration.backend_utilizations,
                        ),
                    )
                )
            )
        ttft_ms = stage_timings["responder.primary"].start_ms + (
            ttft_stage_ms
        )
    ttfs_ms = None
    if "tts.primary" in stage_timings:
        tts = plan.option_for_stage("tts.primary")
        ttfs_stage_ms = (
            tts.tts_first_audio_ms if tts.tts_first_audio_ms is not None else tts.latency_ms
        )
        if calibration is not None:
            ttfs_stage_ms = int(
                round(
                    compute_execution_time_ms(
                        base_latency_ms=float(ttfs_stage_ms),
                        thermal_factor=calibration.backend_thermal_factors.get(
                            tts.backend, calibration.workflow_thermal_scale
                        ),
                        contention_factor=compute_contention_factor(
                            backend=tts.backend,
                            sensitivities=calibration.contention_sensitivities,
                            utilizations=calibration.backend_utilizations,
                        ),
                    )
                )
            )
        ttfs_ms = stage_timings["tts.primary"].start_ms + (
            ttfs_stage_ms
        )
    makespan_ms += orchestration_overhead_ms
    if ttft_ms is not None:
        ttft_ms += orchestration_overhead_ms
    if ttfs_ms is not None:
        ttfs_ms += orchestration_overhead_ms
    quality_loss = sum(option.quality_loss for option in plan.stage_options)
    objective_score = None
    if objective_weights is not None:
        ttfs_proxy = float(ttfs_ms if ttfs_ms is not None else makespan_ms)
        objective_score = compute_objective_score(
            p95_e2e_ms=float(makespan_ms),
            p95_ttfs_ms=ttfs_proxy,
            avg_energy_mj=energy_mj,
            peak_memory_bytes=allocation.peak_bytes,
            copy_bytes=copy_bytes,
            quality_loss=quality_loss,
            weights=objective_weights,
        )
    return SimulationResult(
        workflow_id=workflow.workflow_id,
        plan_id=plan.plan_id,
        makespan_ms=makespan_ms,
        events=tuple(events),
        stage_timings=stage_timings,
        copy_bytes=copy_bytes,
        copy_time_ms=copy_time_ms,
        peak_memory_bytes=allocation.peak_bytes,
        energy_mj=energy_mj,
        ttft_ms=ttft_ms,
        ttfs_ms=ttfs_ms,
        quality_loss=quality_loss,
        objective_score=objective_score,
    )


def simulate_request_stream(
    requests: tuple[RequestSpec, ...] | list[RequestSpec],
    resource_capacities: dict[str, int],
    *,
    bandwidth_bytes_per_ms: float = 1024.0 * 1024.0,
    transfer_fixed_overhead_ms: float = 0.0,
    transfer_layout_ms: float = 0.0,
    objective_weights: ObjectiveWeights | None = None,
    calibration: SimulationCalibration | None = None,
) -> StreamSimulationResult:
    request_list = sorted(tuple(requests), key=lambda item: (item.arrival_ms, item.request_id))
    if not request_list:
        raise ValueError("At least one request is required for stream simulation.")

    backends = {
        option.backend for request in request_list for option in request.plan.stage_options
    }
    missing = sorted(backend for backend in backends if backend not in resource_capacities)
    if missing:
        raise ValueError(
            "Missing explicit resource capacity for backend(s): " + ", ".join(missing)
        )

    backend_next_free = {backend: 0 for backend in resource_capacities}
    backend_cold = {backend: True for backend in resource_capacities}
    events: list[SimulationEvent] = []
    stage_timings: dict[tuple[str, str], StageTiming] = {}
    copy_bytes = 0
    copy_time_ms = 0.0
    energy_mj = 0.0
    quality_loss = 0.0

    for request in request_list:
        workflow = request.workflow
        plan = request.plan
        for stage_id in workflow.topological_order():
            option = plan.option_for_stage(stage_id)
            dependency_ready_ms = request.arrival_ms
            for predecessor in workflow.predecessors(stage_id):
                predecessor_option = plan.option_for_stage(predecessor)
                predecessor_timing = stage_timings[(request.request_id, predecessor)]
                same_memory = predecessor_option.backend == option.backend
                transfer_ms = compute_transfer_cost_ms(
                    size_bytes=predecessor_option.output_bytes,
                    src_backend=predecessor_option.backend,
                    dst_backend=option.backend,
                    same_memory=same_memory,
                    bandwidth_bytes_per_ms=bandwidth_bytes_per_ms,
                    fixed_overhead_ms=transfer_fixed_overhead_ms,
                    layout_ms=transfer_layout_ms,
                )
                dependency_ready_ms = max(
                    dependency_ready_ms,
                    predecessor_timing.finish_ms + int(round(transfer_ms)),
                )
                if predecessor_option.backend != option.backend:
                    copy_bytes += predecessor_option.output_bytes
                    copy_time_ms += transfer_ms
            start_ms = max(request.arrival_ms, dependency_ready_ms, backend_next_free[option.backend])
            thermal_factor = 1.0
            contention_factor = 1.0
            if calibration is not None:
                thermal_factor = calibration.backend_thermal_factors.get(
                    option.backend, calibration.workflow_thermal_scale
                )
                contention_factor = compute_contention_factor(
                    backend=option.backend,
                    sensitivities=calibration.contention_sensitivities,
                    utilizations=calibration.backend_utilizations,
                )
            exec_ms = compute_execution_time_ms(
                base_latency_ms=float(option.latency_ms),
                thermal_factor=thermal_factor,
                contention_factor=contention_factor,
                compile_cost_ms=float(option.compile_cost_ms),
                cold=backend_cold[option.backend],
            )
            finish_ms = start_ms + int(round(exec_ms))
            backend_next_free[option.backend] = finish_ms
            backend_cold[option.backend] = False
            energy_mj += compute_energy_mj(avg_power_mw=option.average_power_mw, time_ms=exec_ms)
            quality_loss += option.quality_loss
            timing = StageTiming(
                stage_id=stage_id,
                backend=option.backend,
                start_ms=start_ms,
                finish_ms=finish_ms,
                request_id=request.request_id,
            )
            stage_timings[(request.request_id, stage_id)] = timing
            events.append(
                SimulationEvent(
                    stage_id=stage_id,
                    backend=option.backend,
                    event_type="start",
                    timestamp_ms=start_ms,
                    request_id=request.request_id,
                )
            )
            events.append(
                SimulationEvent(
                    stage_id=stage_id,
                    backend=option.backend,
                    event_type="finish",
                    timestamp_ms=finish_ms,
                    request_id=request.request_id,
                )
            )

    orchestration_overhead_ms = int(
        round(calibration.orchestration_overhead_ms)
    ) if calibration is not None else 0

    buffer_requests = []
    request_results: dict[str, RequestSimulationResult] = {}
    ttfs_values: list[int] = []
    queue_delays: list[int] = []
    deadline_misses = 0
    request_makespans: list[int] = []
    for request in request_list:
        workflow = request.workflow
        plan = request.plan
        first_start_ms = min(
            stage_timings[(request.request_id, stage_id)].start_ms for stage_id in workflow.stage_ids
        )
        finish_ms = max(
            stage_timings[(request.request_id, stage_id)].finish_ms for stage_id in workflow.stage_ids
        )
        queue_delay_ms = first_start_ms - request.arrival_ms
        makespan_ms = finish_ms - request.arrival_ms + orchestration_overhead_ms
        ttft_ms = None
        if "responder.primary" in workflow.stage_ids:
            responder = plan.option_for_stage("responder.primary")
            ttft_stage_ms = responder.ttft_ms if responder.ttft_ms is not None else responder.latency_ms
            ttft_ms = (
                stage_timings[(request.request_id, "responder.primary")].start_ms
                - request.arrival_ms
                + ttft_stage_ms
                + orchestration_overhead_ms
            )
        ttfs_ms = None
        if "tts.primary" in workflow.stage_ids:
            tts = plan.option_for_stage("tts.primary")
            ttfs_stage_ms = (
                tts.tts_first_audio_ms if tts.tts_first_audio_ms is not None else tts.latency_ms
            )
            ttfs_ms = (
                stage_timings[(request.request_id, "tts.primary")].start_ms
                - request.arrival_ms
                + ttfs_stage_ms
                + orchestration_overhead_ms
            )
            ttfs_values.append(ttfs_ms)
        deadline_missed = request.deadline_ms is not None and makespan_ms > request.deadline_ms
        if deadline_missed:
            deadline_misses += 1
        queue_delays.append(queue_delay_ms)
        request_makespans.append(makespan_ms)
        request_results[request.request_id] = RequestSimulationResult(
            request_id=request.request_id,
            workflow_id=workflow.workflow_id,
            plan_id=plan.plan_id,
            arrival_ms=request.arrival_ms,
            start_ms=first_start_ms,
            finish_ms=finish_ms,
            makespan_ms=makespan_ms,
            queue_delay_ms=queue_delay_ms,
            ttft_ms=ttft_ms,
            ttfs_ms=ttfs_ms,
            deadline_ms=request.deadline_ms,
            deadline_missed=deadline_missed,
        )
        for stage_id in workflow.topological_order():
            option = plan.option_for_stage(stage_id)
            successors = workflow.successors(stage_id)
            death_ms = (
                max(
                    stage_timings[(request.request_id, successor)].start_ms for successor in successors
                )
                if successors
                else stage_timings[(request.request_id, stage_id)].finish_ms
            )
            buffer_requests.append(
                BufferRequest(
                    buffer_id=f"{request.request_id}:{stage_id}",
                    start_ms=stage_timings[(request.request_id, stage_id)].start_ms,
                    end_ms=death_ms,
                    size_bytes=option.output_bytes or (option.memory_mb * 1024 * 1024),
                    memory_type=option.backend,
                )
            )

    allocation = allocate_intervals(buffer_requests)
    stream_makespan_ms = (
        max(result.finish_ms for result in request_results.values())
        - min(request.arrival_ms for request in request_list)
        + orchestration_overhead_ms
    )
    objective_score = None
    if objective_weights is not None:
        objective_score = compute_objective_score(
            p95_e2e_ms=_percentile(request_makespans, 0.95),
            p95_ttfs_ms=_percentile(ttfs_values or request_makespans, 0.95),
            avg_energy_mj=energy_mj / len(request_list),
            peak_memory_bytes=allocation.peak_bytes,
            copy_bytes=copy_bytes,
            quality_loss=quality_loss / len(request_list),
            weights=objective_weights,
        )
    return StreamSimulationResult(
        makespan_ms=stream_makespan_ms,
        request_results=request_results,
        events=tuple(events),
        copy_bytes=copy_bytes,
        copy_time_ms=copy_time_ms,
        peak_memory_bytes=allocation.peak_bytes,
        energy_mj=energy_mj,
        quality_loss=quality_loss,
        objective_score=objective_score,
        p95_queue_delay_ms=_percentile(queue_delays, 0.95),
        deadline_miss_rate=deadline_misses / len(request_list),
    )
