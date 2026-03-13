# Primary Results and Sim-to-Real Behavior

The deployed runtime keeps the support-safe CPU text stack and the working FastVLM NPU path. Workflow A exposes live responder-to-TTS streaming, while workflows B and C keep the VLM on the LiteRT NPU and preserve CPU retrieval where it is faster than the measured GPU alternative.

```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=0.97\textwidth]{figures/evaluation_overview.png}
\caption{Evaluation overview combining sim-to-real deltas, primary workflow latencies, continuous-stream scores, and selected baseline comparison slices from the canonical checkpoint.}
\label{fig:evaluation_overview}
\end{figure*}
```

The calibrated model is intentionally practical rather than mystical. It is tuned to measured CPU/GPU/NPU behavior and reports residuals explicitly, so the simulator can justify ranking and sensitivity analysis without claiming exact prediction for every workload family.

```latex
\begin{figure*}[!t]
\centering
\begin{minipage}[t]{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{figures/workflow_primary_results.png}
\caption{Warm latency of the three deployed workflows under the checkpointed support-safe plan.}
\label{fig:workflow_primary_results}
\end{minipage}\hfill
\begin{minipage}[t]{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{figures/continuous_stream_results.png}
\caption{Representative continuous-stream GraphPilot scores used to evaluate queue-aware behavior.}
\label{fig:continuous_stream_results}
\end{minipage}
\end{figure*}
```

The checkpoint also includes a sustained on-device run so the paper can show whether the simulator-facing thermal model reflects actual long-run drift rather than only one-shot latency. The detailed sustained panels and sim-to-real detail are broken out in the next section so the primary results page stays focused on the deployed workflows themselves.

```latex
\FloatBarrier
```
