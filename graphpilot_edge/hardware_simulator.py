from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Mapping


@dataclass(frozen=True)
class HardwareInstance:
    resource_id: str
    resource_type: str
    supported_op_classes: tuple[str, ...]
    service_rates: Mapping[str, float]
    memory_bandwidth_bytes_per_ms: float
    launch_overhead_ms: float
    busy_power_mw: float
    idle_power_mw: float
    queue_depth: int
    batching_beta: float
    batching_tau: float
    thermal_time_constant_ms: float
    thermal_resistance_c_per_mw: float
    thermal_threshold_c: float
    thermal_gamma: float
    contention_domain: str

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ValueError("resource_id cannot be blank")
        if self.queue_depth < 1:
            raise ValueError("queue_depth must be >= 1")
        if self.memory_bandwidth_bytes_per_ms <= 0:
            raise ValueError("memory_bandwidth_bytes_per_ms must be > 0")
        if self.launch_overhead_ms < 0:
            raise ValueError("launch_overhead_ms must be >= 0")
        if self.busy_power_mw < 0 or self.idle_power_mw < 0:
            raise ValueError("power values must be >= 0")
        if self.batching_tau <= 0:
            raise ValueError("batching_tau must be > 0")
        if self.thermal_time_constant_ms <= 0:
            raise ValueError("thermal_time_constant_ms must be > 0")
        if self.thermal_gamma < 0:
            raise ValueError("thermal_gamma must be >= 0")
        unsupported = [op_class for op_class, rate in self.service_rates.items() if rate <= 0]
        if unsupported:
            raise ValueError(
                f"service_rates must be > 0 for every op class, got non-positive entries for {unsupported}"
            )


@dataclass(frozen=True)
class PartitionAssignment:
    partition_id: str
    resource_id: str
    op_volume: Mapping[str, float]
    memory_bytes: int
    output_bytes: int

    def __post_init__(self) -> None:
        if not self.partition_id:
            raise ValueError("partition_id cannot be blank")
        if self.memory_bytes < 0:
            raise ValueError("memory_bytes must be >= 0")
        if self.output_bytes < 0:
            raise ValueError("output_bytes must be >= 0")


@dataclass(frozen=True)
class TaskProfile:
    task_id: str
    op_volume: Mapping[str, float]
    memory_bytes: int
    input_bytes: int
    output_bytes: int
    fallback_partitions: tuple[PartitionAssignment, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id cannot be blank")
        if self.memory_bytes < 0 or self.input_bytes < 0 or self.output_bytes < 0:
            raise ValueError("memory/input/output bytes must be >= 0")
        invalid = [op_class for op_class, volume in self.op_volume.items() if volume < 0]
        if invalid:
            raise ValueError(f"op_volume must be >= 0 for every op class, got {invalid}")


@dataclass(frozen=True)
class HardwarePrediction:
    resource_id: str
    execution_time_ms: float
    compute_time_ms: float
    memory_time_ms: float
    transfer_time_ms: float
    energy_mj: float
    batching_gain: float
    thermal_slowdown: float
    contention_multiplier: float
    used_fallback: bool = False


@dataclass(frozen=True)
class ResourceQueueState:
    free_at_ms: float
    queued_tasks: int = 0

    def __post_init__(self) -> None:
        if self.free_at_ms < 0:
            raise ValueError("free_at_ms must be >= 0")
        if self.queued_tasks < 0:
            raise ValueError("queued_tasks must be >= 0")


def compute_batching_gain(batch_size: int, beta: float, tau: float) -> float:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if tau <= 0:
        raise ValueError("tau must be > 0")
    return 1.0 + beta * (1.0 - exp(-batch_size / tau))


def evolve_thermal_state(
    current_temp_c: float,
    ambient_temp_c: float,
    delta_ms: float,
    time_constant_ms: float,
    thermal_resistance_c_per_mw: float,
    power_mw: float,
) -> float:
    if delta_ms < 0:
        raise ValueError("delta_ms must be >= 0")
    if time_constant_ms <= 0:
        raise ValueError("time_constant_ms must be > 0")
    decay = exp(-delta_ms / time_constant_ms)
    return (
        ambient_temp_c
        + (current_temp_c - ambient_temp_c) * decay
        + thermal_resistance_c_per_mw * power_mw * (1.0 - decay)
    )


def compute_thermal_slowdown(temp_c: float, threshold_c: float, gamma: float) -> float:
    if gamma < 0:
        raise ValueError("gamma must be >= 0")
    return 1.0 + gamma * max(0.0, temp_c - threshold_c)


class HardwareSimulator:
    def __init__(
        self,
        *,
        resources: tuple[HardwareInstance, ...] | list[HardwareInstance],
        transfer_bandwidth_bytes_per_ms: Mapping[tuple[str, str], float],
        transfer_fixed_overhead_ms: Mapping[tuple[str, str], float],
        transfer_layout_overhead_ms: Mapping[tuple[str, str], float],
        contention_sensitivities: Mapping[tuple[str, str], float],
        ambient_temperature_c: float,
    ) -> None:
        if not resources:
            raise ValueError("resources must not be empty")
        self._resources = {resource.resource_id: resource for resource in resources}
        self.transfer_bandwidth_bytes_per_ms = dict(transfer_bandwidth_bytes_per_ms)
        self.transfer_fixed_overhead_ms = dict(transfer_fixed_overhead_ms)
        self.transfer_layout_overhead_ms = dict(transfer_layout_overhead_ms)
        self.contention_sensitivities = dict(contention_sensitivities)
        self.ambient_temperature_c = ambient_temperature_c

    def predict_task(
        self,
        *,
        task: TaskProfile,
        resource_id: str,
        batch_size: int,
        current_temp_c: Mapping[str, float],
        utilizations: Mapping[str, float],
    ) -> HardwarePrediction:
        resource = self._require_resource(resource_id)
        unsupported = set(task.op_volume) - set(resource.supported_op_classes)
        if unsupported:
            if not task.fallback_partitions:
                raise ValueError(
                    f"Task '{task.task_id}' uses unsupported op classes {sorted(unsupported)} on resource '{resource_id}' without explicit fallback partitions."
                )
            return self._predict_partitioned_task(
                task=task,
                primary_resource_id=resource_id,
                batch_size=batch_size,
                current_temp_c=current_temp_c,
                utilizations=utilizations,
            )

        return self._predict_single_assignment(
            op_volume=task.op_volume,
            memory_bytes=task.memory_bytes,
            resource=resource,
            batch_size=batch_size,
            current_temp_c=current_temp_c.get(resource_id, self.ambient_temperature_c),
            utilizations=utilizations,
            used_fallback=False,
            transfer_time_ms=0.0,
        )

    def predict_earliest_finish(
        self,
        *,
        task: TaskProfile,
        resource_id: str,
        arrival_ms: float,
        queue_state: ResourceQueueState,
        batch_size: int,
        current_temp_c: Mapping[str, float],
        utilizations: Mapping[str, float],
    ) -> tuple[float, float]:
        resource = self._require_resource(resource_id)
        if queue_state.queued_tasks >= resource.queue_depth:
            raise ValueError(
                f"Resource '{resource_id}' queue depth exceeded: queued_tasks={queue_state.queued_tasks}, queue_depth={resource.queue_depth}."
            )
        prediction = self.predict_task(
            task=task,
            resource_id=resource_id,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        start_ms = max(arrival_ms, queue_state.free_at_ms)
        finish_ms = start_ms + prediction.execution_time_ms
        return start_ms, finish_ms

    def compute_transfer_time_ms(
        self,
        *,
        size_bytes: int,
        src_resource_id: str,
        dst_resource_id: str,
        same_memory_compatible: bool = False,
        shared_buffer_available: bool = False,
        shared_buffer_map_ms: float = 0.0,
    ) -> float:
        if size_bytes < 0:
            raise ValueError("size_bytes must be >= 0")
        if src_resource_id == dst_resource_id and same_memory_compatible:
            return 0.0
        if shared_buffer_available:
            return shared_buffer_map_ms
        bandwidth = self.transfer_bandwidth_bytes_per_ms[(src_resource_id, dst_resource_id)]
        fixed = self.transfer_fixed_overhead_ms.get((src_resource_id, dst_resource_id), 0.0)
        layout = self.transfer_layout_overhead_ms.get((src_resource_id, dst_resource_id), 0.0)
        return fixed + (size_bytes / bandwidth) + layout

    def _predict_partitioned_task(
        self,
        *,
        task: TaskProfile,
        primary_resource_id: str,
        batch_size: int,
        current_temp_c: Mapping[str, float],
        utilizations: Mapping[str, float],
    ) -> HardwarePrediction:
        partition_predictions: list[HardwarePrediction] = []
        total_transfer_ms = 0.0
        previous_partition: PartitionAssignment | None = None
        for partition in task.fallback_partitions:
            resource = self._require_resource(partition.resource_id)
            unsupported = set(partition.op_volume) - set(resource.supported_op_classes)
            if unsupported:
                raise ValueError(
                    f"Partition '{partition.partition_id}' assigns unsupported op classes {sorted(unsupported)} to resource '{partition.resource_id}'."
                )
            prediction = self._predict_single_assignment(
                op_volume=partition.op_volume,
                memory_bytes=partition.memory_bytes,
                resource=resource,
                batch_size=batch_size,
                current_temp_c=current_temp_c.get(resource.resource_id, self.ambient_temperature_c),
                utilizations=utilizations,
                used_fallback=True,
                transfer_time_ms=0.0,
            )
            partition_predictions.append(prediction)
            if previous_partition is not None and previous_partition.resource_id != partition.resource_id:
                total_transfer_ms += self.compute_transfer_time_ms(
                    size_bytes=previous_partition.output_bytes,
                    src_resource_id=previous_partition.resource_id,
                    dst_resource_id=partition.resource_id,
                )
            previous_partition = partition

        total_execution_ms = sum(prediction.execution_time_ms for prediction in partition_predictions)
        total_energy_mj = sum(prediction.energy_mj for prediction in partition_predictions)
        return HardwarePrediction(
            resource_id=primary_resource_id,
            execution_time_ms=total_execution_ms + total_transfer_ms,
            compute_time_ms=sum(prediction.compute_time_ms for prediction in partition_predictions),
            memory_time_ms=sum(prediction.memory_time_ms for prediction in partition_predictions),
            transfer_time_ms=total_transfer_ms,
            energy_mj=total_energy_mj,
            batching_gain=partition_predictions[0].batching_gain if partition_predictions else 1.0,
            thermal_slowdown=max(
                (prediction.thermal_slowdown for prediction in partition_predictions),
                default=1.0,
            ),
            contention_multiplier=max(
                (prediction.contention_multiplier for prediction in partition_predictions),
                default=1.0,
            ),
            used_fallback=True,
        )

    def _predict_single_assignment(
        self,
        *,
        op_volume: Mapping[str, float],
        memory_bytes: int,
        resource: HardwareInstance,
        batch_size: int,
        current_temp_c: float,
        utilizations: Mapping[str, float],
        used_fallback: bool,
        transfer_time_ms: float,
    ) -> HardwarePrediction:
        batching_gain = compute_batching_gain(
            batch_size=batch_size,
            beta=resource.batching_beta,
            tau=resource.batching_tau,
        )
        compute_time_ms = (
            sum(op_volume.get(op_class, 0.0) / resource.service_rates[op_class] for op_class in op_volume)
            / batching_gain
            if op_volume
            else 0.0
        )
        memory_time_ms = memory_bytes / resource.memory_bandwidth_bytes_per_ms
        thermal_slowdown = compute_thermal_slowdown(
            temp_c=current_temp_c,
            threshold_c=resource.thermal_threshold_c,
            gamma=resource.thermal_gamma,
        )
        contention_multiplier = self._contention_multiplier(resource.resource_id, utilizations)
        execution_time_ms = resource.launch_overhead_ms + max(compute_time_ms, memory_time_ms) * thermal_slowdown * contention_multiplier
        energy_mj = resource.busy_power_mw * execution_time_ms / 1000.0
        return HardwarePrediction(
            resource_id=resource.resource_id,
            execution_time_ms=execution_time_ms,
            compute_time_ms=compute_time_ms,
            memory_time_ms=memory_time_ms,
            transfer_time_ms=transfer_time_ms,
            energy_mj=energy_mj,
            batching_gain=batching_gain,
            thermal_slowdown=thermal_slowdown,
            contention_multiplier=contention_multiplier,
            used_fallback=used_fallback,
        )

    def _contention_multiplier(
        self,
        resource_id: str,
        utilizations: Mapping[str, float],
    ) -> float:
        multiplier = 1.0
        target = self._require_resource(resource_id)
        for other_id, utilization in utilizations.items():
            if other_id == resource_id:
                continue
            other = self._require_resource(other_id)
            if other.contention_domain != target.contention_domain:
                continue
            multiplier += self.contention_sensitivities.get((resource_id, other_id), 0.0) * utilization
        return multiplier

    def _require_resource(self, resource_id: str) -> HardwareInstance:
        try:
            return self._resources[resource_id]
        except KeyError as exc:
            raise KeyError(f"Unknown hardware resource '{resource_id}'.") from exc
