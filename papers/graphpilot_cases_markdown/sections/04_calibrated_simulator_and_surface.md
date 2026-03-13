# Calibrated Simulator and Support-Safe Surface

The simulator is anchored to actual checkpointed CPU/GPU/NPU statistics rather than undocumented vendor throughput claims. Surrogate parameters capture launch overhead, contention, orchestration bias, and workload-family residuals so that the simulator is useful for ranking and what-if analysis without pretending to be an oracle.

```latex
The simulator is tied to the same mobile hardware and LiteRT deployment surface described by the checkpointed Qualcomm and LiteRT evidence paths~\cite{qualcomm2025,litert2026}.
```

```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=0.97\textwidth]{figures/calibration_overview.png}
\caption{Calibration overview from the canonical checkpoint. The simulator is tuned against measured workflow deltas, family-level residuals, and fitted backend surrogate terms for CPU/GPU/NPU resources.}
\label{fig:calibration_overview}
\end{figure*}
```

```latex
\begin{figure*}[!t]
\centering
\begin{minipage}[t]{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{figures/backend_affinity_matrix.png}
\caption{Family/backend mean latency matrix used for ranking support-safe resource affinity.}
\label{fig:backend_affinity}
\end{minipage}\hfill
\begin{minipage}[t]{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{figures/support_safe_feasibility.png}
\caption{Support-safe feasibility matrix from the checkpointed backend registry. Unsupported paths remain explicit.}
\label{fig:support_safe_feasibility}
\end{minipage}
\end{figure*}
```

The calibrated evidence surface therefore supports three concrete uses: workload-family ranking, sensitivity analysis under fallback and thermal drift, and honest method-class baseline comparison inside the same simulator/runtime environment.

```latex
{{CALIBRATION_TABLE_TEX}}
```
