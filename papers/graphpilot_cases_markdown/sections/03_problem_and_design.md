# Problem Formulation and Design

GraphPilot treats each request as a DAG whose nodes are stage or macro-region tasks and whose edges carry full outputs, chunks, or token streams. The runtime optimizes the checkpoint-pinned objective

```latex
\begin{equation}
\begin{aligned}
J(\Pi)={}&\alpha P95(T_{e2e}) + \beta P95(T_{TFS}) + \gamma \bar{E} + \delta M_{peak}\\
&+ \eta B_{copy} + \zeta Q_{loss} + \xi P95(T_{queue}) + \psi R_{miss}.
\end{aligned}
\end{equation}
```

Every assigned region must be support-safe, memory must remain within budget, and quality loss must stay within the checkpointed revision bounds. Responder prefill and responder decode remain separate throughout the runtime and simulator.

```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=0.97\textwidth]{figures/offline_online_split.png}
\caption{GraphPilot splits into an offline brain that profiles, calibrates, ranks, and banks plans, and an online brain that dispatches support-safe work with explicit streaming and memory/KV control.}
\label{fig:offline_online_split}
\end{figure*}
```

The calibrated simulator keeps the same causality structure as the deployed runtime. Execution cost combines launch overhead, compute-vs-memory bottlenecks, batching, thermal slowdown, contention, transfer penalties, and explicit fallback partitions:

```latex
\begin{equation}
T^{exec}_{u,h}=L_h + \max(T^{ops}_{u,h}(B), T^{mem}_{u,h})\,\rho_h(\theta)\,\kappa_h(q)
\end{equation}
```

```latex
\begin{equation}
T_{xfer}(S,h,h') =
\begin{cases}
0 & \text{compatible same-space handoff} \\
T_{map}(S) & \text{shared-buffer path} \\
\tau_0^{h,h'} + S/BW_{h,h'} + \tau^{h,h'}_{layout} & \text{otherwise.}
\end{cases}
\end{equation}
```

Online dispatch uses queue-aware priority with critical-path pressure, first-output bias, copy cost, memory/KV risk, and thermal penalties. Unsupported paths do not silently degrade into success; they remain simulator-only or fail fast.
