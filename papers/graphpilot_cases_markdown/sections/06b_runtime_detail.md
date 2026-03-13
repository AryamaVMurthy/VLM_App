# Detailed Runtime and Sustained Behavior

Two pieces of evidence matter after the primary workflow latencies: whether the simulator remains anchored to the actual deployed workflows, and whether the runtime stays stable under a sustained device run. Both are checkpoint-pinned here instead of being left implicit.

```latex
\begin{figure}[!t]
\centering
\includegraphics[width=0.96\columnwidth]{figures/sim_real_calibration.png}
\caption{Checkpoint-pinned sim-to-real deltas for the deployed workflows. The simulator is used for ranking and sensitivity analysis, not as an exact oracle.}
\label{fig:sim_real_calibration_detail}
\end{figure}
```

```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=0.92\textwidth,height=0.76\textheight,keepaspectratio]{figures/sustained_detail.png}
\caption{Detailed sustained runtime panels from the checkpointed device run. The deployed runtime stays within a mild thermal envelope, but the measured latency drift still justifies thermal-aware queueing and plan-bank switching.}
\label{fig:sustained_detail}
\end{figure*}
```

The sustained run keeps the claim honest. GraphPilot is not just a best-of-one-shot latency story; it is evaluated against long-run traces and real TTFS-sensitive behavior on the same device class as the deployed runtime.
