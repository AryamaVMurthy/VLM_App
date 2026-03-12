package com.qidk.fastvlm.core.graphpilot

data class GraphPilotThermalPlanBankConfig(
  val warmSlowdownThreshold: Double = 1.10,
  val hotSlowdownThreshold: Double = 1.25,
  val lowMemoryFractionThreshold: Double = 0.20,
  val hysteresisWindows: Int = 2,
) {
  init {
    require(warmSlowdownThreshold >= 1.0) { "warmSlowdownThreshold must be >= 1.0" }
    require(hotSlowdownThreshold >= warmSlowdownThreshold) { "hotSlowdownThreshold must be >= warmSlowdownThreshold" }
    require(lowMemoryFractionThreshold in 0.0..1.0) { "lowMemoryFractionThreshold must be in [0, 1]" }
    require(hysteresisWindows >= 1) { "hysteresisWindows must be >= 1" }
  }
}

data class GraphPilotPlanBankDecision(
  val stateId: String,
  val reason: String,
  val slowdownFactor: Double,
  val freeBytes: Long,
  val usableBytes: Long,
)

class GraphPilotThermalPlanBank(
  private val config: GraphPilotThermalPlanBankConfig = GraphPilotThermalPlanBankConfig(),
) {
  private var currentStateId: String = "cool"
  private var warmWindowCount: Int = 0
  private var hotWindowCount: Int = 0

  fun selectState(
    slowdownFactor: Double,
    freeBytes: Long,
    usableBytes: Long,
  ): GraphPilotPlanBankDecision {
    require(slowdownFactor > 0.0) { "slowdownFactor must be > 0" }
    require(usableBytes > 0L) { "usableBytes must be > 0" }
    require(freeBytes >= 0L) { "freeBytes must be >= 0" }

    val freeFraction = freeBytes.toDouble() / usableBytes.toDouble()
    if (freeFraction <= config.lowMemoryFractionThreshold) {
      warmWindowCount = 0
      hotWindowCount = 0
      currentStateId = "lowmem"
      return GraphPilotPlanBankDecision(
        stateId = currentStateId,
        reason = "free memory dropped below low-memory threshold",
        slowdownFactor = slowdownFactor,
        freeBytes = freeBytes,
        usableBytes = usableBytes,
      )
    }

    if (slowdownFactor >= config.hotSlowdownThreshold) {
      hotWindowCount += 1
      warmWindowCount = 0
      if (hotWindowCount >= config.hysteresisWindows || currentStateId == "hot") {
        currentStateId = "hot"
      }
      return GraphPilotPlanBankDecision(
        stateId = currentStateId,
        reason = "thermal slowdown reached hot threshold",
        slowdownFactor = slowdownFactor,
        freeBytes = freeBytes,
        usableBytes = usableBytes,
      )
    }

    if (slowdownFactor >= config.warmSlowdownThreshold) {
      warmWindowCount += 1
      hotWindowCount = 0
      if (warmWindowCount >= config.hysteresisWindows || currentStateId == "warm") {
        currentStateId = "warm"
      }
      return GraphPilotPlanBankDecision(
        stateId = currentStateId,
        reason = "thermal slowdown reached warm threshold",
        slowdownFactor = slowdownFactor,
        freeBytes = freeBytes,
        usableBytes = usableBytes,
      )
    }

    warmWindowCount = 0
    hotWindowCount = 0
    currentStateId = "cool"
    return GraphPilotPlanBankDecision(
      stateId = currentStateId,
      reason = "thermal slowdown and free memory are in the cool operating range",
      slowdownFactor = slowdownFactor,
      freeBytes = freeBytes,
      usableBytes = usableBytes,
    )
  }
}
