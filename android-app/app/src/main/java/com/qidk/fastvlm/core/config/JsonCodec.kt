package com.qidk.fastvlm.core.config

import kotlinx.serialization.json.Json

object JsonCodec {
  val instance: Json =
    Json {
      ignoreUnknownKeys = true
      encodeDefaults = true
      explicitNulls = false
      prettyPrint = false
    }
}
