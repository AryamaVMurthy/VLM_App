# Baselines and Ablations

All comparisons run inside the same simulator/runtime environment. The paper uses internal baselines, explicit ablation baselines, and faithful method-class proxy baselines instead of mixing incomparable published numbers from other hardware.

```latex
\begin{figure}[!t]
\centering
\includegraphics[width=0.96\columnwidth]{figures/baseline_comparison.png}
\caption{Internal baseline deltas versus GraphPilot on the informative checkpoint-pinned workloads.}
\label{fig:baseline_comparison}
\end{figure}
```

```latex
\begin{figure}[!t]
\centering
\includegraphics[width=0.96\columnwidth]{figures/proxy_baseline_comparison.png}
\caption{Faithful method-class proxy deltas versus GraphPilot under the same calibrated environment.}
\label{fig:proxy_baseline_comparison}
\end{figure}
```

The informative ablations are the ones that actually move the checkpointed workloads: single-backend collapse, pipeline removal on streaming-sensitive paths, retrieval backend swaps, and thermal-plan switching under sustained load. Where GraphPilot ties static baselines, the paper reports that tie directly.

```latex
{{BASELINE_TABLE_TEX}}
```

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.9\columnwidth]{figures/ablation_breakdown.png}
\caption{Selected ablation deltas on the informative checkpoint-pinned workloads. The non-zero cases identify where the current support-safe deployment actually benefits from GraphPilot's control logic.}
\label{fig:ablation_breakdown}
\end{figure}
```

```latex
{{ABLATION_MATRIX_TABLE_TEX}}
```

```latex
\FloatBarrier
```
