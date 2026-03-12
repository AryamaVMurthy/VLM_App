from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .catalog import load_workflow_catalog
from .hardware_simulator import HardwareSimulator, TaskProfile
from .models import CandidatePlan, StageOption
from .workflow import WorkflowDag, WorkflowEdge

MEBIBYTE = 1024 * 1024
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW_TEMPLATES_PATH = ROOT_DIR / "configs" / "graphpilot_edge" / "workflow_templates.json"


@dataclass(frozen=True)
class ModelGraphNodeProfile:
    stage_id: str
    family: str
    variant_id: str
    task_profile: TaskProfile
    knob_values: Mapping[str, Any] = field(default_factory=dict)
    quality_loss: float = 0.0
    ttft_hint_ms: int | None = None
    tts_first_audio_hint_ms: int | None = None


@dataclass(frozen=True)
class ModelGraphScenario:
    scenario_id: str
    workflow: WorkflowDag
    node_profiles: Mapping[str, ModelGraphNodeProfile]

    def __post_init__(self) -> None:
        missing = sorted(stage_id for stage_id in self.workflow.stage_ids if stage_id not in self.node_profiles)
        if missing:
            raise ValueError(
                f"ModelGraphScenario '{self.scenario_id}' is missing node profiles for stages {missing}."
            )

    def instantiate_candidate_plan(
        self,
        *,
        simulator: HardwareSimulator,
        resource_assignment: Mapping[str, str],
        batch_size: int = 1,
        current_temp_c: Mapping[str, float] | None = None,
        utilizations: Mapping[str, float] | None = None,
    ) -> CandidatePlan:
        current_temp_c = current_temp_c or {}
        utilizations = utilizations or {}
        missing = sorted(stage_id for stage_id in self.workflow.stage_ids if stage_id not in resource_assignment)
        if missing:
            raise ValueError(
                f"Scenario '{self.scenario_id}' is missing resource assignments for stages {missing}."
            )
        stage_options: list[StageOption] = []
        for stage_id in self.workflow.topological_order():
            node = self.node_profiles[stage_id]
            resource_id = resource_assignment[stage_id]
            prediction = simulator.predict_task(
                task=node.task_profile,
                resource_id=resource_id,
                batch_size=batch_size,
                current_temp_c=current_temp_c,
                utilizations=utilizations,
            )
            resource = simulator.resource(resource_id)
            stage_options.append(
                StageOption(
                    stage_id=stage_id,
                    variant_id=node.variant_id,
                    backend=resource.resource_type,
                    latency_ms=max(1, int(math.ceil(prediction.execution_time_ms))),
                    memory_mb=max(1, int(math.ceil(node.task_profile.memory_bytes / MEBIBYTE))),
                    output_bytes=node.task_profile.output_bytes,
                    average_power_mw=resource.busy_power_mw,
                    quality_loss=node.quality_loss,
                    compile_cost_ms=0,
                    ttft_ms=node.ttft_hint_ms,
                    tts_first_audio_ms=node.tts_first_audio_hint_ms,
                )
            )
        return CandidatePlan(
            workflow_id=self.workflow.workflow_id,
            stage_options=tuple(stage_options),
        )


def build_llm_scenario(
    *,
    scenario_id: str,
    prompt_tokens: int,
    output_tokens: int,
    context_tokens: int = 0,
    max_output_tokens: int | None = None,
    context_cap: int | None = None,
    variant_id: str = "gemma3_1b_it",
) -> ModelGraphScenario:
    total_context = prompt_tokens + context_tokens
    effective_context = min(total_context, context_cap or total_context)
    effective_output = min(output_tokens, max_output_tokens or output_tokens)
    quality_loss = 0.0
    if context_cap is not None and context_cap < total_context:
        quality_loss += (total_context - context_cap) / max(total_context, 1) * 0.2
    if max_output_tokens is not None and max_output_tokens < output_tokens:
        quality_loss += (output_tokens - max_output_tokens) / max(output_tokens, 1) * 0.3

    kv_bytes = _llm_kv_bytes(tokens=effective_context, output_tokens=effective_output)
    workflow = WorkflowDag(
        workflow_id=scenario_id,
        stage_ids=(
            "llm.prompt_assembly",
            "llm.prefill",
            "llm.decode",
            "llm.postprocess",
        ),
        edges=(
            WorkflowEdge("llm.prompt_assembly", "llm.prefill", "full"),
            WorkflowEdge("llm.prefill", "llm.decode", "full"),
            WorkflowEdge("llm.decode", "llm.postprocess", "full"),
        ),
    )
    node_profiles = {
        "llm.prompt_assembly": ModelGraphNodeProfile(
            stage_id="llm.prompt_assembly",
            family="llm",
            variant_id=f"{variant_id}:prompt_assembly",
            task_profile=TaskProfile(
                task_id="llm.prompt_assembly",
                op_volume={"control": float(total_context), "gather": float(total_context)},
                memory_bytes=max(64 * 1024, effective_context * 64),
                input_bytes=effective_context * 4,
                output_bytes=effective_context * 8,
            ),
            knob_values={"context_cap": context_cap, "max_output_tokens": max_output_tokens},
            quality_loss=quality_loss * 0.1,
        ),
        "llm.prefill": ModelGraphNodeProfile(
            stage_id="llm.prefill",
            family="llm",
            variant_id=f"{variant_id}:prefill",
            task_profile=TaskProfile(
                task_id="llm.prefill",
                op_volume={
                    "gemm": float(effective_context * 96),
                    "attn": float(effective_context * 48),
                    "elem": float(effective_context * 24),
                },
                memory_bytes=max(4 * MEBIBYTE, effective_context * 4096),
                input_bytes=effective_context * 8,
                output_bytes=effective_context * 16,
            ),
            knob_values={"context_cap": context_cap, "max_output_tokens": max_output_tokens},
            quality_loss=quality_loss * 0.4,
        ),
        "llm.decode": ModelGraphNodeProfile(
            stage_id="llm.decode",
            family="llm",
            variant_id=f"{variant_id}:decode",
            task_profile=TaskProfile(
                task_id="llm.decode",
                op_volume={
                    "gemm": float(effective_output * 32),
                    "attn": float(effective_output * 24),
                    "elem": float(effective_output * 12),
                },
                memory_bytes=max(2 * MEBIBYTE, kv_bytes),
                input_bytes=effective_context * 16,
                output_bytes=effective_output * 8,
            ),
            knob_values={"context_cap": context_cap, "max_output_tokens": max_output_tokens},
            quality_loss=quality_loss * 0.4,
            ttft_hint_ms=max(1, effective_context // 2),
        ),
        "llm.postprocess": ModelGraphNodeProfile(
            stage_id="llm.postprocess",
            family="llm",
            variant_id=f"{variant_id}:postprocess",
            task_profile=TaskProfile(
                task_id="llm.postprocess",
                op_volume={"control": float(effective_output), "elem": float(effective_output)},
                memory_bytes=max(64 * 1024, effective_output * 128),
                input_bytes=effective_output * 8,
                output_bytes=effective_output * 8,
            ),
            knob_values={"context_cap": context_cap, "max_output_tokens": max_output_tokens},
            quality_loss=quality_loss * 0.1,
        ),
    }
    return ModelGraphScenario(scenario_id=scenario_id, workflow=workflow, node_profiles=node_profiles)


def build_vlm_scenario(
    *,
    scenario_id: str,
    prompt_tokens: int,
    output_tokens: int,
    image_resolution: int,
    visual_token_budget: int,
    variant_id: str = "fastvlm",
) -> ModelGraphScenario:
    image_pixels = image_resolution * image_resolution
    workflow = WorkflowDag(
        workflow_id=scenario_id,
        stage_ids=(
            "vlm.preprocess",
            "vlm.vision_encoder",
            "vlm.projector",
            "vlm.multimodal_prefill",
            "vlm.decode",
        ),
        edges=(
            WorkflowEdge("vlm.preprocess", "vlm.vision_encoder", "full"),
            WorkflowEdge("vlm.vision_encoder", "vlm.projector", "full"),
            WorkflowEdge("vlm.projector", "vlm.multimodal_prefill", "full"),
            WorkflowEdge("vlm.multimodal_prefill", "vlm.decode", "full"),
        ),
    )
    node_profiles = {
        "vlm.preprocess": ModelGraphNodeProfile(
            stage_id="vlm.preprocess",
            family="vlm",
            variant_id=f"{variant_id}:preprocess",
            task_profile=TaskProfile(
                task_id="vlm.preprocess",
                op_volume={"control": float(image_pixels / 2048), "elem": float(image_pixels / 1024)},
                memory_bytes=max(2 * MEBIBYTE, image_pixels * 4),
                input_bytes=image_pixels * 3,
                output_bytes=image_pixels,
            ),
            knob_values={"image_resolution": image_resolution, "visual_token_budget": visual_token_budget},
        ),
        "vlm.vision_encoder": ModelGraphNodeProfile(
            stage_id="vlm.vision_encoder",
            family="vlm",
            variant_id=f"{variant_id}:vision_encoder",
            task_profile=TaskProfile(
                task_id="vlm.vision_encoder",
                op_volume={"conv": float(image_pixels / 32), "gemm": float(image_pixels / 48)},
                memory_bytes=max(8 * MEBIBYTE, image_pixels * 8),
                input_bytes=image_pixels,
                output_bytes=visual_token_budget * 64,
            ),
            knob_values={"image_resolution": image_resolution, "visual_token_budget": visual_token_budget},
        ),
        "vlm.projector": ModelGraphNodeProfile(
            stage_id="vlm.projector",
            family="vlm",
            variant_id=f"{variant_id}:projector",
            task_profile=TaskProfile(
                task_id="vlm.projector",
                op_volume={"gemm": float(visual_token_budget * 12), "elem": float(visual_token_budget * 4)},
                memory_bytes=max(2 * MEBIBYTE, visual_token_budget * 4096),
                input_bytes=visual_token_budget * 64,
                output_bytes=visual_token_budget * 48,
            ),
            knob_values={"visual_token_budget": visual_token_budget},
        ),
        "vlm.multimodal_prefill": ModelGraphNodeProfile(
            stage_id="vlm.multimodal_prefill",
            family="vlm",
            variant_id=f"{variant_id}:multimodal_prefill",
            task_profile=TaskProfile(
                task_id="vlm.multimodal_prefill",
                op_volume={
                    "gemm": float((prompt_tokens + visual_token_budget) * 64),
                    "attn": float((prompt_tokens + visual_token_budget) * 40),
                    "elem": float((prompt_tokens + visual_token_budget) * 12),
                },
                memory_bytes=max(4 * MEBIBYTE, (prompt_tokens + visual_token_budget) * 32768),
                input_bytes=(prompt_tokens + visual_token_budget) * 48,
                output_bytes=(prompt_tokens + visual_token_budget) * 32,
            ),
            knob_values={"prompt_tokens": prompt_tokens, "visual_token_budget": visual_token_budget},
        ),
        "vlm.decode": ModelGraphNodeProfile(
            stage_id="vlm.decode",
            family="vlm",
            variant_id=f"{variant_id}:decode",
            task_profile=TaskProfile(
                task_id="vlm.decode",
                op_volume={"gemm": float(output_tokens * 28), "attn": float(output_tokens * 20)},
                memory_bytes=max(2 * MEBIBYTE, output_tokens * 4096),
                input_bytes=output_tokens * 16,
                output_bytes=output_tokens * 8,
            ),
            knob_values={"output_tokens": output_tokens},
            ttft_hint_ms=max(1, prompt_tokens // 2),
        ),
    }
    return ModelGraphScenario(scenario_id=scenario_id, workflow=workflow, node_profiles=node_profiles)


def build_stt_scenario(
    *,
    scenario_id: str,
    audio_duration_ms: int,
    chunk_size_ms: int,
    beam_size: int,
    variant_id: str = "whisper_stt",
) -> ModelGraphScenario:
    audio_samples = max(1, audio_duration_ms * 16)
    workflow = WorkflowDag(
        workflow_id=scenario_id,
        stage_ids=("stt.audio_frontend", "stt.decode"),
        edges=(WorkflowEdge("stt.audio_frontend", "stt.decode", "chunk"),),
        chunk_sizes=(("stt.audio_frontend", chunk_size_ms),),
    )
    node_profiles = {
        "stt.audio_frontend": ModelGraphNodeProfile(
            stage_id="stt.audio_frontend",
            family="stt",
            variant_id=f"{variant_id}:frontend",
            task_profile=TaskProfile(
                task_id="stt.audio_frontend",
                op_volume={"audio": float(audio_samples / 512), "elem": float(audio_samples / 1024)},
                memory_bytes=max(512 * 1024, audio_samples * 4),
                input_bytes=audio_samples * 2,
                output_bytes=max(1, audio_samples // 8),
            ),
            knob_values={"chunk_size_ms": chunk_size_ms, "beam_size": beam_size},
        ),
        "stt.decode": ModelGraphNodeProfile(
            stage_id="stt.decode",
            family="stt",
            variant_id=f"{variant_id}:decode",
            task_profile=TaskProfile(
                task_id="stt.decode",
                op_volume={
                    "gemm": float((audio_duration_ms / max(chunk_size_ms, 1)) * beam_size * 24),
                    "attn": float((audio_duration_ms / max(chunk_size_ms, 1)) * beam_size * 10),
                },
                memory_bytes=max(2 * MEBIBYTE, beam_size * 512 * 1024),
                input_bytes=max(1, audio_samples // 8),
                output_bytes=max(64, audio_duration_ms // 4),
            ),
            knob_values={"chunk_size_ms": chunk_size_ms, "beam_size": beam_size},
            ttft_hint_ms=max(1, chunk_size_ms // 2),
        ),
    }
    return ModelGraphScenario(scenario_id=scenario_id, workflow=workflow, node_profiles=node_profiles)


def build_tts_scenario(
    *,
    scenario_id: str,
    text_tokens: int,
    chunk_size_chars: int,
    quality_mode: str,
    variant_id: str = "android_tts",
) -> ModelGraphScenario:
    quality_factor = {"fast": 0.8, "balanced": 1.0, "high": 1.3}.get(quality_mode, 1.0)
    workflow = WorkflowDag(
        workflow_id=scenario_id,
        stage_ids=("tts.text_normalize", "tts.acoustic", "tts.vocoder"),
        edges=(
            WorkflowEdge("tts.text_normalize", "tts.acoustic", "full"),
            WorkflowEdge("tts.acoustic", "tts.vocoder", "chunk"),
        ),
        chunk_sizes=(("tts.acoustic", chunk_size_chars),),
    )
    node_profiles = {
        "tts.text_normalize": ModelGraphNodeProfile(
            stage_id="tts.text_normalize",
            family="tts",
            variant_id=f"{variant_id}:normalize",
            task_profile=TaskProfile(
                task_id="tts.text_normalize",
                op_volume={"control": float(text_tokens), "elem": float(text_tokens)},
                memory_bytes=max(256 * 1024, text_tokens * 128),
                input_bytes=text_tokens * 4,
                output_bytes=text_tokens * 8,
            ),
            knob_values={"chunk_size_chars": chunk_size_chars, "quality_mode": quality_mode},
        ),
        "tts.acoustic": ModelGraphNodeProfile(
            stage_id="tts.acoustic",
            family="tts",
            variant_id=f"{variant_id}:acoustic",
            task_profile=TaskProfile(
                task_id="tts.acoustic",
                op_volume={"gemm": float(text_tokens * 18 * quality_factor), "attn": float(text_tokens * 8 * quality_factor)},
                memory_bytes=max(2 * MEBIBYTE, int(text_tokens * 4096 * quality_factor)),
                input_bytes=text_tokens * 8,
                output_bytes=max(chunk_size_chars * 8, text_tokens * 16),
            ),
            knob_values={"chunk_size_chars": chunk_size_chars, "quality_mode": quality_mode},
        ),
        "tts.vocoder": ModelGraphNodeProfile(
            stage_id="tts.vocoder",
            family="tts",
            variant_id=f"{variant_id}:vocoder",
            task_profile=TaskProfile(
                task_id="tts.vocoder",
                op_volume={"conv": float(text_tokens * 24 * quality_factor), "elem": float(text_tokens * 12 * quality_factor)},
                memory_bytes=max(3 * MEBIBYTE, int(text_tokens * 6144 * quality_factor)),
                input_bytes=max(chunk_size_chars * 8, text_tokens * 16),
                output_bytes=max(chunk_size_chars * 32, text_tokens * 64),
            ),
            knob_values={"chunk_size_chars": chunk_size_chars, "quality_mode": quality_mode},
            tts_first_audio_hint_ms=max(1, chunk_size_chars // 2),
        ),
    }
    return ModelGraphScenario(scenario_id=scenario_id, workflow=workflow, node_profiles=node_profiles)


def build_retrieval_scenario(
    *,
    scenario_id: str,
    query_tokens: int,
    corpus_size: int,
    top_k: int,
    rerank: bool,
    variant_id: str = "embeddinggemma_300m",
) -> ModelGraphScenario:
    stage_ids = ["retrieval.embedder", "retrieval.ann"]
    edges = [
        WorkflowEdge("retrieval.embedder", "retrieval.ann", "full"),
    ]
    if rerank:
        stage_ids.append("retrieval.rerank")
        edges.append(WorkflowEdge("retrieval.ann", "retrieval.rerank", "full"))
    workflow = WorkflowDag(
        workflow_id=scenario_id,
        stage_ids=tuple(stage_ids),
        edges=tuple(edges),
    )
    node_profiles = {
        "retrieval.embedder": ModelGraphNodeProfile(
            stage_id="retrieval.embedder",
            family="retrieval",
            variant_id=f"{variant_id}:embedder",
            task_profile=TaskProfile(
                task_id="retrieval.embedder",
                op_volume={"gemm": float(query_tokens * 12), "gather": float(query_tokens * 4)},
                memory_bytes=max(2 * MEBIBYTE, query_tokens * 2048),
                input_bytes=query_tokens * 4,
                output_bytes=768 * 4,
            ),
            knob_values={"top_k": top_k, "rerank": rerank},
        ),
        "retrieval.ann": ModelGraphNodeProfile(
            stage_id="retrieval.ann",
            family="retrieval",
            variant_id=f"{variant_id}:ann",
            task_profile=TaskProfile(
                task_id="retrieval.ann",
                op_volume={"ann": float(corpus_size * max(top_k, 1))},
                memory_bytes=max(2 * MEBIBYTE, corpus_size * 64),
                input_bytes=768 * 4,
                output_bytes=max(256, top_k * 512),
            ),
            knob_values={"top_k": top_k, "rerank": rerank},
        ),
    }
    if rerank:
        node_profiles["retrieval.rerank"] = ModelGraphNodeProfile(
            stage_id="retrieval.rerank",
            family="retrieval",
            variant_id=f"{variant_id}:rerank",
            task_profile=TaskProfile(
                task_id="retrieval.rerank",
                op_volume={"gemm": float(top_k * query_tokens * 10), "attn": float(top_k * query_tokens * 4)},
                memory_bytes=max(2 * MEBIBYTE, top_k * query_tokens * 1024),
                input_bytes=max(256, top_k * 512),
                output_bytes=max(256, top_k * 256),
            ),
            knob_values={"top_k": top_k, "rerank": rerank},
        )
    return ModelGraphScenario(scenario_id=scenario_id, workflow=workflow, node_profiles=node_profiles)


def build_cnn_scenario(
    *,
    scenario_id: str,
    image_resolution: int,
    variant_id: str = "mobile_cnn",
) -> ModelGraphScenario:
    pixels = image_resolution * image_resolution
    workflow = WorkflowDag(
        workflow_id=scenario_id,
        stage_ids=("cnn.preprocess", "cnn.backbone", "cnn.head"),
        edges=(
            WorkflowEdge("cnn.preprocess", "cnn.backbone", "full"),
            WorkflowEdge("cnn.backbone", "cnn.head", "full"),
        ),
    )
    node_profiles = {
        "cnn.preprocess": ModelGraphNodeProfile(
            stage_id="cnn.preprocess",
            family="cnn",
            variant_id=f"{variant_id}:preprocess",
            task_profile=TaskProfile(
                task_id="cnn.preprocess",
                op_volume={"control": float(pixels / 1024), "elem": float(pixels / 512)},
                memory_bytes=max(1 * MEBIBYTE, pixels * 4),
                input_bytes=pixels * 3,
                output_bytes=pixels,
            ),
        ),
        "cnn.backbone": ModelGraphNodeProfile(
            stage_id="cnn.backbone",
            family="cnn",
            variant_id=f"{variant_id}:backbone",
            task_profile=TaskProfile(
                task_id="cnn.backbone",
                op_volume={"conv": float(pixels / 8), "elem": float(pixels / 16)},
                memory_bytes=max(8 * MEBIBYTE, pixels * 8),
                input_bytes=pixels,
                output_bytes=pixels // 4,
            ),
        ),
        "cnn.head": ModelGraphNodeProfile(
            stage_id="cnn.head",
            family="cnn",
            variant_id=f"{variant_id}:head",
            task_profile=TaskProfile(
                task_id="cnn.head",
                op_volume={"gemm": float(pixels / 16), "elem": float(pixels / 32)},
                memory_bytes=max(2 * MEBIBYTE, pixels * 2),
                input_bytes=pixels // 4,
                output_bytes=1024,
            ),
        ),
    }
    return ModelGraphScenario(scenario_id=scenario_id, workflow=workflow, node_profiles=node_profiles)


def build_vit_scenario(
    *,
    scenario_id: str,
    image_resolution: int,
    token_budget: int,
    variant_id: str = "mobile_vit",
) -> ModelGraphScenario:
    pixels = image_resolution * image_resolution
    workflow = WorkflowDag(
        workflow_id=scenario_id,
        stage_ids=("vit.preprocess", "vit.encoder", "vit.head"),
        edges=(
            WorkflowEdge("vit.preprocess", "vit.encoder", "full"),
            WorkflowEdge("vit.encoder", "vit.head", "full"),
        ),
    )
    node_profiles = {
        "vit.preprocess": ModelGraphNodeProfile(
            stage_id="vit.preprocess",
            family="vit",
            variant_id=f"{variant_id}:preprocess",
            task_profile=TaskProfile(
                task_id="vit.preprocess",
                op_volume={"control": float(pixels / 1024), "elem": float(token_budget)},
                memory_bytes=max(1 * MEBIBYTE, pixels * 4),
                input_bytes=pixels * 3,
                output_bytes=token_budget * 32,
            ),
            knob_values={"token_budget": token_budget},
        ),
        "vit.encoder": ModelGraphNodeProfile(
            stage_id="vit.encoder",
            family="vit",
            variant_id=f"{variant_id}:encoder",
            task_profile=TaskProfile(
                task_id="vit.encoder",
                op_volume={"gemm": float(token_budget * 24), "attn": float(token_budget * 18)},
                memory_bytes=max(8 * MEBIBYTE, token_budget * 8192),
                input_bytes=token_budget * 32,
                output_bytes=token_budget * 32,
            ),
            knob_values={"token_budget": token_budget},
        ),
        "vit.head": ModelGraphNodeProfile(
            stage_id="vit.head",
            family="vit",
            variant_id=f"{variant_id}:head",
            task_profile=TaskProfile(
                task_id="vit.head",
                op_volume={"gemm": float(token_budget * 8), "elem": float(token_budget * 4)},
                memory_bytes=max(2 * MEBIBYTE, token_budget * 1024),
                input_bytes=token_budget * 32,
                output_bytes=1024,
            ),
            knob_values={"token_budget": token_budget},
        ),
    }
    return ModelGraphScenario(scenario_id=scenario_id, workflow=workflow, node_profiles=node_profiles)


def build_sequential_glue_scenario(
    *,
    scenario_id: str,
    item_count: int,
    variant_id: str = "sequential_glue",
) -> ModelGraphScenario:
    workflow = WorkflowDag(
        workflow_id=scenario_id,
        stage_ids=("glue.tokenize", "glue.validate", "glue.pack"),
        edges=(
            WorkflowEdge("glue.tokenize", "glue.validate", "full"),
            WorkflowEdge("glue.validate", "glue.pack", "full"),
        ),
    )
    node_profiles = {
        "glue.tokenize": ModelGraphNodeProfile(
            stage_id="glue.tokenize",
            family="glue",
            variant_id=f"{variant_id}:tokenize",
            task_profile=TaskProfile(
                task_id="glue.tokenize",
                op_volume={"control": float(item_count), "gather": float(item_count)},
                memory_bytes=max(256 * 1024, item_count * 256),
                input_bytes=item_count * 8,
                output_bytes=item_count * 16,
            ),
        ),
        "glue.validate": ModelGraphNodeProfile(
            stage_id="glue.validate",
            family="glue",
            variant_id=f"{variant_id}:validate",
            task_profile=TaskProfile(
                task_id="glue.validate",
                op_volume={"control": float(item_count * 2), "elem": float(item_count)},
                memory_bytes=max(256 * 1024, item_count * 256),
                input_bytes=item_count * 16,
                output_bytes=item_count * 16,
            ),
        ),
        "glue.pack": ModelGraphNodeProfile(
            stage_id="glue.pack",
            family="glue",
            variant_id=f"{variant_id}:pack",
            task_profile=TaskProfile(
                task_id="glue.pack",
                op_volume={"control": float(item_count), "elem": float(item_count)},
                memory_bytes=max(256 * 1024, item_count * 128),
                input_bytes=item_count * 16,
                output_bytes=item_count * 12,
            ),
        ),
    }
    return ModelGraphScenario(scenario_id=scenario_id, workflow=workflow, node_profiles=node_profiles)


def build_primitive_operator_scenario(
    *,
    scenario_id: str,
    op_class: str,
    op_volume: float,
    memory_bytes: int,
    output_bytes: int,
    family: str = "primitive",
    variant_id: str | None = None,
) -> ModelGraphScenario:
    stage_id = f"{family}.{op_class}"
    workflow = WorkflowDag(
        workflow_id=scenario_id,
        stage_ids=(stage_id,),
        edges=(),
    )
    node_profiles = {
        stage_id: ModelGraphNodeProfile(
            stage_id=stage_id,
            family=family,
            variant_id=variant_id or f"{family}:{op_class}",
            task_profile=TaskProfile(
                task_id=stage_id,
                op_volume={op_class: op_volume},
                memory_bytes=memory_bytes,
                input_bytes=max(256, output_bytes // 2),
                output_bytes=output_bytes,
            ),
        )
    }
    return ModelGraphScenario(scenario_id=scenario_id, workflow=workflow, node_profiles=node_profiles)


def build_assistant_workflow_scenario(
    *,
    scenario_id: str,
    workflow_id: str,
    audio_duration_ms: int,
    stt_chunk_size_ms: int = 800,
    stt_beam_size: int = 4,
    planner_prompt_tokens: int = 96,
    planner_output_tokens: int = 32,
    responder_prompt_tokens: int = 256,
    responder_output_tokens: int = 96,
    responder_context_tokens: int = 0,
    responder_context_cap: int | None = None,
    responder_max_output_tokens: int | None = None,
    image_resolution: int = 896,
    visual_token_budget: int = 256,
    retrieval_query_tokens: int = 32,
    retrieval_corpus_size: int = 1_000,
    retrieval_top_k: int = 4,
    retrieval_rerank: bool = False,
    tts_chunk_size_chars: int = 80,
    tts_quality_mode: str = "balanced",
    workflow_templates_path: Path = DEFAULT_WORKFLOW_TEMPLATES_PATH,
) -> ModelGraphScenario:
    workflows = load_workflow_catalog(workflow_templates_path)
    template = workflows.get(workflow_id)
    if template is None:
        raise ValueError(f"Unknown workflow template '{workflow_id}'.")

    chunk_entries = {
        stage_id: chunk_size for stage_id, chunk_size in template.chunk_sizes
    }
    if "asr.primary" in template.stage_ids:
        chunk_entries["asr.primary"] = stt_chunk_size_ms
    workflow = WorkflowDag(
        workflow_id=scenario_id,
        stage_ids=template.stage_ids,
        edges=template.edges,
        chunk_sizes=tuple(sorted(chunk_entries.items())),
    )

    node_profiles: dict[str, ModelGraphNodeProfile] = {}
    for stage_id in workflow.stage_ids:
        if stage_id == "asr.primary":
            stt = build_stt_scenario(
                scenario_id=f"{scenario_id}.asr",
                audio_duration_ms=audio_duration_ms,
                chunk_size_ms=stt_chunk_size_ms,
                beam_size=stt_beam_size,
            )
            node_profiles[stage_id] = _aggregate_node_profiles(
                stage_id=stage_id,
                family="assistant",
                variant_id="whisper_stt:stage",
                ordered_nodes=(
                    stt.node_profiles["stt.audio_frontend"],
                    stt.node_profiles["stt.decode"],
                ),
            )
            continue
        if stage_id == "planner.primary":
            planner = build_llm_scenario(
                scenario_id=f"{scenario_id}.planner",
                prompt_tokens=planner_prompt_tokens,
                output_tokens=planner_output_tokens,
                max_output_tokens=planner_output_tokens,
                variant_id="gemma3_1b_it_planner",
            )
            node_profiles[stage_id] = _aggregate_node_profiles(
                stage_id=stage_id,
                family="assistant",
                variant_id="gemma3_1b_it:planner",
                ordered_nodes=tuple(planner.node_profiles[node_id] for node_id in planner.workflow.topological_order()),
            )
            continue
        if stage_id == "vlm.fastvlm.primary":
            vlm = build_vlm_scenario(
                scenario_id=f"{scenario_id}.vlm",
                prompt_tokens=planner_output_tokens,
                output_tokens=max(32, responder_output_tokens // 2),
                image_resolution=image_resolution,
                visual_token_budget=visual_token_budget,
            )
            node_profiles[stage_id] = _aggregate_node_profiles(
                stage_id=stage_id,
                family="assistant",
                variant_id="fastvlm:stage",
                # Keep stage-level assistant VLM focused on the accelerated model core.
                # Image preprocessing belongs to the sequential-glue workload family and
                # should not be hidden inside an NPU-friendly stage aggregate.
                ordered_nodes=tuple(
                    vlm.node_profiles[node_id]
                    for node_id in (
                        "vlm.vision_encoder",
                        "vlm.projector",
                        "vlm.multimodal_prefill",
                        "vlm.decode",
                    )
                ),
            )
            continue
        if stage_id == "retrieval.embedder.primary":
            retrieval = build_retrieval_scenario(
                scenario_id=f"{scenario_id}.retrieval",
                query_tokens=retrieval_query_tokens,
                corpus_size=retrieval_corpus_size,
                top_k=retrieval_top_k,
                rerank=retrieval_rerank,
            )
            node_profiles[stage_id] = _aggregate_node_profiles(
                stage_id=stage_id,
                family="assistant",
                variant_id="embeddinggemma_300m:stage",
                ordered_nodes=tuple(
                    retrieval.node_profiles[node_id] for node_id in retrieval.workflow.topological_order()
                ),
            )
            continue
        if stage_id == "responder.primary":
            responder = build_llm_scenario(
                scenario_id=f"{scenario_id}.responder",
                prompt_tokens=responder_prompt_tokens,
                output_tokens=responder_output_tokens,
                context_tokens=responder_context_tokens,
                context_cap=responder_context_cap,
                max_output_tokens=responder_max_output_tokens,
                variant_id="gemma3_1b_it_responder",
            )
            node_profiles[stage_id] = _aggregate_node_profiles(
                stage_id=stage_id,
                family="assistant",
                variant_id="gemma3_1b_it:responder",
                ordered_nodes=tuple(
                    responder.node_profiles[node_id] for node_id in responder.workflow.topological_order()
                ),
            )
            continue
        if stage_id == "tts.primary":
            tts = build_tts_scenario(
                scenario_id=f"{scenario_id}.tts",
                text_tokens=responder_output_tokens,
                chunk_size_chars=tts_chunk_size_chars,
                quality_mode=tts_quality_mode,
            )
            node_profiles[stage_id] = _aggregate_node_profiles(
                stage_id=stage_id,
                family="assistant",
                variant_id="android_tts:stage",
                # Text normalization is sequential glue; keep the assistant TTS stage
                # focused on acoustic and vocoder work.
                ordered_nodes=tuple(
                    tts.node_profiles[node_id]
                    for node_id in (
                        "tts.acoustic",
                        "tts.vocoder",
                    )
                ),
            )
            continue
        raise ValueError(
            f"Assistant workflow scenario builder does not know how to model stage '{stage_id}'."
        )

    return ModelGraphScenario(scenario_id=scenario_id, workflow=workflow, node_profiles=node_profiles)


def _aggregate_node_profiles(
    *,
    stage_id: str,
    family: str,
    variant_id: str,
    ordered_nodes: tuple[ModelGraphNodeProfile, ...],
) -> ModelGraphNodeProfile:
    if not ordered_nodes:
        raise ValueError(f"Cannot aggregate stage '{stage_id}' without node profiles.")
    op_volume: dict[str, float] = {}
    for node in ordered_nodes:
        for op_class, volume in node.task_profile.op_volume.items():
            op_volume[op_class] = op_volume.get(op_class, 0.0) + volume
    knob_values: dict[str, Any] = {}
    for node in ordered_nodes:
        knob_values.update(node.knob_values)
    ttft_hint_ms = next((node.ttft_hint_ms for node in ordered_nodes if node.ttft_hint_ms is not None), None)
    tts_first_audio_hint_ms = next(
        (
            node.tts_first_audio_hint_ms
            for node in ordered_nodes
            if node.tts_first_audio_hint_ms is not None
        ),
        None,
    )
    return ModelGraphNodeProfile(
        stage_id=stage_id,
        family=family,
        variant_id=variant_id,
        task_profile=TaskProfile(
            task_id=stage_id,
            op_volume=op_volume,
            memory_bytes=max(node.task_profile.memory_bytes for node in ordered_nodes),
            input_bytes=ordered_nodes[0].task_profile.input_bytes,
            output_bytes=ordered_nodes[-1].task_profile.output_bytes,
        ),
        knob_values=knob_values,
        quality_loss=sum(node.quality_loss for node in ordered_nodes),
        ttft_hint_ms=ttft_hint_ms,
        tts_first_audio_hint_ms=tts_first_audio_hint_ms,
    )


def _llm_kv_bytes(*, tokens: int, output_tokens: int) -> int:
    layers = 18
    kv_heads = 4
    head_dim = 256
    dtype_bytes = 2
    total_tokens = max(1, tokens + output_tokens)
    return 2 * layers * kv_heads * head_dim * total_tokens * dtype_bytes
