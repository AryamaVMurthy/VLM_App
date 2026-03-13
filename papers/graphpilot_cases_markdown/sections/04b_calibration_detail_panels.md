# Detailed Calibration Panels

The checkpointed simulator is tuned to measured CPU/GPU/NPU behavior through explicit workflow deltas, family residuals, and fitted backend surrogate terms. The detailed panels below show the actual statistics used for that tuning rather than only the summarized composite.

```latex
\begin{figure*}[!t]
\centering
\begin{minipage}[t]{0.48\textwidth}
\centering
\textbf{(a) Sim-to-real deltas}\\[2pt]
\includegraphics[width=\textwidth]{figures/sim_real_calibration.png}
\end{minipage}\hfill
\begin{minipage}[t]{0.48\textwidth}
\centering
\textbf{(b) Family residual MAE}\\[2pt]
\includegraphics[width=\textwidth]{figures/calibration_family_mae.png}
\end{minipage}

\vspace{6pt}

\begin{minipage}[t]{0.48\textwidth}
\centering
\textbf{(c) Backend launch overhead}\\[2pt]
\includegraphics[width=\textwidth]{figures/backend_launch_overhead.png}
\end{minipage}\hfill
\begin{minipage}[t]{0.48\textwidth}
\centering
\textbf{(d) Backend contention scale}\\[2pt]
\includegraphics[width=\textwidth]{figures/backend_contention_scale.png}
\end{minipage}
\caption{Detailed simulator-calibration panels from the checkpoint. These plots expose the actual CPU/GPU/NPU-tuned surrogate terms used by GraphPilot rather than hiding them inside a single overview graphic.}
\label{fig:calibration_detail_panels}
\end{figure*}
```
