# GraphPilot CASES Paper Markdown-First Redesign

## Goal

Replace the current hardcoded CASES paper generator with a markdown-first paper pipeline that builds a cleaner, more visual, checkpoint-pinned IEEE CASES paper from canonical GraphPilot evidence. The new paper must reduce layout fragility, increase figure density, keep prose concise, and make every rendered page manually inspectable and reproducible.

## Scope Boundary

This redesign is anchored to the existing verified GraphPilot evidence surface:
- canonical checkpoint manifests under `artifacts/graphpilot_edge/checkpoints/`
- canonical artifact packs under `artifacts/graphpilot_edge/reports/`
- checkpoint-pinned experiment, calibration, characterization, and registry summaries
- `Truth-docs/graphpilot_edge_revision_report.pdf` as the revision boundary for claims, formulas, and required studies

The paper claim stays unchanged: GraphPilot-Edge is a profiler-driven runtime and calibrated simulator for continuous multimodal assistant DAGs on heterogeneous mobile SoCs, with support-safe placement, explicit fallback accounting, streaming-aware scheduling, and memory/KV-aware control. The paper must not broaden beyond the verified system.

## Problems With The Current Builder

1. Prose is hardcoded in `scripts/build_graphpilot_cases_paper.py`, which makes structure and iteration slow.
2. Figures are generated independently of a paper outline, so the layout does not optimize around the visual story.
3. The current PDF overuses text and underuses visuals.
4. Float placement is brittle, leading to whitespace, awkward figure grouping, and weak page composition.
5. Manual review is possible only after LaTeX build, but there is no markdown-stage paper source that humans can edit and inspect cleanly.

## Chosen Approach

### Approach A — Recommended: markdown-first with controlled LaTeX emission

- Author the paper in markdown sections under `papers/graphpilot_cases_markdown/`.
- Keep tables and figures programmatic and checkpoint-pinned.
- Add a deterministic markdown-to-LaTeX renderer inside the repo instead of depending on Pandoc.
- Emit a LaTeX source tree that remains CASES/IEEEtran-compatible.
- Make visual density a first-class constraint: architecture, workflow, calibration, baseline, ablation, and sensitivity figures must drive the narrative.

Why this approach:
- It gives a human-editable paper source.
- It removes the current hardcoded prose bottleneck.
- It keeps deterministic control over LaTeX output and avoids external tool drift.
- It makes manual PDF review easier because section text is cleanly separated from layout logic.

### Approach B — hardcoded LaTeX rewrite

- Rewrite the existing builder but keep prose embedded in Python.

Why not chosen:
- It does not solve authoring quality.
- It keeps the paper source opaque.
- It makes future revisions expensive.

### Approach C — adopt Pandoc pipeline

- Author markdown and use Pandoc to convert to LaTeX.

Why not chosen:
- `pandoc` is not available in this environment.
- It introduces an external dependency and template drift risk.
- It gives less precise control over IEEE float behavior than a constrained internal renderer.

## Paper Structure

The paper will be rebuilt around a smaller set of denser visuals and tighter prose.

### Primary sections

1. Abstract
2. Introduction
3. System Overview
4. Problem Formulation
5. Support-Safe Runtime and Simulator Design
6. Experimental Methodology
7. Results
8. Ablations and Sensitivity
9. Limitations and Artifact Notes
10. Conclusion

### Visual priorities

The paper should emphasize:
- deployment/runtime architecture
- offline/online split
- workload universe coverage
- calibration quality and sim-to-real deltas
- primary workflow A/B/C results
- continuous-stream behavior
- baseline comparison matrix for selected workloads
- ablation and sensitivity summaries

### Content that should be de-emphasized

- long narrative background paragraphs
- repetitive explanations already encoded in figures or tables
- weak zero-margin plots that do not add information

## Figure/Table Design Rules

1. Every figure must come from canonical checkpoint-pinned evidence.
2. Prefer multi-panel overview figures over many tiny isolated plots.
3. Figures must show non-zero, decision-relevant results when available.
4. Tables must be narrow, aligned, and only carry dense numbers that are hard to visualize.
5. All captions must explain why the figure matters for the claim.
6. No empty placeholder panels and no zero-information graphics.

## Builder Architecture

### Markdown source

Store section sources under:
- `papers/graphpilot_cases_markdown/paper.md`
- or `papers/graphpilot_cases_markdown/sections/*.md`

The renderer will support a constrained markdown subset:
- headings
- paragraphs
- bullet lists
- numbered lists
- simple emphasis
- code or verbatim blocks where needed for artifact notes
- explicit raw-LaTeX include blocks for figures/tables only

### Generated outputs

The build should emit:
- rendered markdown bundle for human review
- generated LaTeX section files
- `main.tex`
- built `main.pdf`
- page-render PNGs for manual inspection
- build summary with page count and artifact references

### Figure build chain

The figure builder should keep using checkpoint summaries as inputs, but should shift from many separate weak figures to a curated set of paper-first composites.

## Validation Requirements

1. Write failing tests first for the markdown-first builder behavior.
2. Regenerate figures from canonical checkpoint inputs.
3. Build the paper PDF.
4. Render every page to images.
5. Manually inspect for:
   - clipped text
   - overlap
   - excessive whitespace
   - broken table alignment
   - unreadable labels
6. Iterate until the PDF is visually clean.

## Risks And Controls

### Risk: markdown renderer becomes too permissive
Control: implement a strict subset only and fail fast on unsupported constructs.

### Risk: float drift still causes whitespace
Control: reduce float count, use more top-of-page placement, merge related panels, and keep section prose shorter.

### Risk: figures remain weak because source data are weak
Control: choose more informative checkpoint-derived metrics, especially calibration residuals, single-backend deltas, retrieval CPU-vs-GPU, and selected baseline score matrices.

### Risk: paper claim drifts beyond evidence
Control: all claims stay bound to the revision report and checkpoint-pinned evidence.

## Acceptance Criteria

The redesign is complete when:
- a markdown-first source exists in the repo
- the paper builder consumes markdown before LaTeX generation
- the CASES PDF builds reproducibly from a canonical checkpoint
- rendered pages have no overlaps or major whitespace defects
- the paper contains a stronger visual story than the current version
- all builder/figure tests pass
- the final artifact pack points to the rebuilt paper
