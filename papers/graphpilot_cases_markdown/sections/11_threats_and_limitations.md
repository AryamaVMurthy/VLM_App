# Threats and Limitations

The paper stays inside the measured system boundary.

- Backend feasibility is still narrow outside the FastVLM NPU path, so the strongest claim is support-safe scheduling rather than universal heterogeneous execution.
- Simulator residuals vary by family, so the simulator is appropriate for ranking and sensitivity analysis rather than exact oracle prediction.
- The broader workload universe is intentionally larger than the current deployed runtime surface, which means some policies remain calibrated what-if studies rather than online runtime behavior.
- Faithful method-class proxies are comparison surfaces inside the same environment, not line-by-line reimplementations of external systems.

These limits strengthen the paper rather than weaken it: they keep the claim aligned to the checkpointed evidence instead of pretending broader support than the artifact can defend.
