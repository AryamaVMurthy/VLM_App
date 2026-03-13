# Detailed Sensitivity Panels

The checkpointed CASES story depends on a few causal mechanisms: fallback-aware costing, streaming-aware pipelining, thermal plan-bank behavior, and objective-weight sensitivity. The detailed panels below expose those mechanisms directly instead of leaving them implicit.

```latex
\begin{figure*}[!t]
\centering
\begin{minipage}[t]{0.48\textwidth}
\centering
\textbf{(a) Fallback penalty}\\[2pt]
\includegraphics[width=\textwidth]{figures/fallback_penalty.png}
\end{minipage}\hfill
\begin{minipage}[t]{0.48\textwidth}
\centering
\textbf{(b) Selected ablation deltas}\\[2pt]
\includegraphics[width=\textwidth]{figures/ablation_breakdown.png}
\end{minipage}

\vspace{6pt}

\begin{minipage}[t]{0.48\textwidth}
\centering
\textbf{(c) Thermal plan-bank curves}\\[2pt]
\includegraphics[width=\textwidth]{figures/thermal_plan_bank.png}
\end{minipage}\hfill
\begin{minipage}[t]{0.48\textwidth}
\centering
\textbf{(d) Objective sensitivity}\\[2pt]
\includegraphics[width=\textwidth]{figures/objective_sensitivity.png}
\end{minipage}
\caption{Detailed sensitivity panels used to explain GraphPilot's ranking and control behavior. The plots remain checkpoint-pinned and use the same canonical evidence paths as the primary paper figures.}
\label{fig:sensitivity_detail_panels}
\end{figure*}
```
