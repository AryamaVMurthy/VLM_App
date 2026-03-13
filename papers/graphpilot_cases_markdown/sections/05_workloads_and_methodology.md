# Workloads and Methodology

The paper uses a hybrid evidence strategy. Workflows A/B/C remain the primary real-device anchor, while the broader workload universe supports simulator calibration, workload-family characterization, continuous-stream studies, and ablations.

```latex
\begin{figure}[!t]
\centering
\includegraphics[width=0.95\columnwidth]{figures/workload_universe_coverage.png}
\caption{Checkpoint-pinned workload-universe coverage across deployed workflows, model-family studies, continuous streams, and stress cases.}
\label{fig:workload_coverage}
\end{figure}
```

The checkpoint manifest prevents evidence drift. Every figure, table, and paragraph in the paper is regenerated from the pinned checkpoint `\texttt{{{CHECKPOINT_MANIFEST_REF}}}`, and the broader revision boundary is `\texttt{{{TRUTH_SOURCE_REF}}}`.

```latex
{{WORKFLOW_RESULTS_TABLE_TEX}}
```

```latex
{{WORKLOAD_TABLE_TEX}}
```
