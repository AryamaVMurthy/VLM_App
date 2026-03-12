package com.qidk.fastvlm.core.graphpilot

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class GraphPilotThermalPlanBankTest {
  @Test
  fun selectsLowmemWhenFreeMemoryDropsBelowThreshold() {
    val bank = GraphPilotThermalPlanBank()

    val decision =
      bank.selectState(
        slowdownFactor = 1.0,
        freeBytes = 10L,
        usableBytes = 100L,
      )

    assertThat(decision.stateId).isEqualTo("lowmem")
    assertThat(decision.reason).contains("low-memory")
  }

  @Test
  fun requiresHysteresisBeforeSwitchingToHot() {
    val bank = GraphPilotThermalPlanBank(GraphPilotThermalPlanBankConfig(hysteresisWindows = 2))

    val first =
      bank.selectState(
        slowdownFactor = 1.3,
        freeBytes = 90L,
        usableBytes = 100L,
      )
    val second =
      bank.selectState(
        slowdownFactor = 1.3,
        freeBytes = 90L,
        usableBytes = 100L,
      )

    assertThat(first.stateId).isEqualTo("cool")
    assertThat(second.stateId).isEqualTo("hot")
  }

  @Test
  fun returnsToCoolWhenPressureClears() {
    val bank = GraphPilotThermalPlanBank(GraphPilotThermalPlanBankConfig(hysteresisWindows = 1))

    bank.selectState(
      slowdownFactor = 1.3,
      freeBytes = 90L,
      usableBytes = 100L,
    )
    val cool =
      bank.selectState(
        slowdownFactor = 1.0,
        freeBytes = 90L,
        usableBytes = 100L,
      )

    assertThat(cool.stateId).isEqualTo("cool")
  }
}
