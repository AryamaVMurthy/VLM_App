# Conclusion

GraphPilot-Edge demonstrates a support-safe runtime and a calibrated simulator for continuous multimodal assistant DAGs on SM8750-class mobile hardware. The deployed runtime keeps FastVLM on the LiteRT NPU, preserves explicit streaming and memory/KV control, and refuses to turn infeasible accelerator paths into fake wins. The simulator extends that runtime with queue-aware, fallback-aware what-if analysis across a broader workload universe.

The strongest surviving claim is narrower than universal heterogeneous execution but stronger than a design sketch: GraphPilot is a measured mobile runtime plus a calibrated simulator, both rooted in one checkpointed evidence surface and one revision-bounded CASES artifact package.
