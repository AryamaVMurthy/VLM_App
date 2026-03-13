# Runtime Notes and Artifact Surface

```latex
\FloatBarrier
```

The deployed runtime remains intentionally honest. FastVLM stays on the working LiteRT NPU path, while text and speech remain on verified CPU paths until broader accelerator support becomes support-safe. That narrowness is a measured limitation, not a hidden fallback.

```latex
{{FEASIBILITY_TABLE_TEX}}
```

The active paper build is checkpoint-pinned. The artifact surface is rooted in `\texttt{{{CHECKPOINT_MANIFEST_REF}}}`, and the revision-bounded truth document is `\texttt{{{TRUTH_SOURCE_REF}}}`. The release package includes the runtime code, simulator, registries, experiment summaries, and audit outputs needed to regenerate the figures and tables.
