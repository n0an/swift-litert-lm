// LiteRTDemo — text decode benchmark probe.
//
// Isolates text decode speed to explain the gap to the model's headline tok/s.
// Three measurements, all GPU, warm:
//   1. The official fixed harness (256 prefill / 256 decode, speculative off).
//   2. A real generation with the MTP speculative drafter OFF.
//   3. The same with the drafter ON (the lever behind Gemma 4's headline speed).
//
// Launch with LITERT_BENCH=1. Lines are tagged "BENCH:" for devicectl polling.

import Foundation
import LiteRTFoundation
import os

enum BenchSelfTest {
  private static let logger = Logger(subsystem: "com.example.litertdemo", category: "BENCH")

  static var isRequested: Bool { ProcessInfo.processInfo.environment["LITERT_BENCH"] != nil }

  static func log(_ message: String) {
    logger.log("\(message, privacy: .public)")
    print("BENCH: \(message)")
    fflush(stdout)
  }

  private static func mb(_ bytes: Int64) -> String {
    String(format: "%.0f MB", Double(bytes) / 1_048_576)
  }

  static func run() async {
    log("start — device=\(ProcessInfo.processInfo.operatingSystemVersionString)")

    let model = LiteRTModel.gemma4_E2B
    let path: String
    do {
      path = try await LiteRTChat.ensureModel(model)
    } catch {
      log("FATAL could not obtain model: \(error.localizedDescription)")
      log("DONE")
      return
    }

    // The official fixed harness (256 prefill / 256 decode, GPU) is the clean
    // apples-to-apples number. Run it with the MTP speculative drafter OFF then
    // ON — the drafter is the lever behind Gemma 4's headline tokens/sec.
    ExperimentalFlags.optIntoExperimentalAPIs()
    for spec in [false, true] {
      ExperimentalFlags.enableSpeculativeDecoding = spec
      do {
        let bi = try await benchmark(
          modelPath: path, backend: .gpu, prefillTokens: 256, decodeTokens: 256)
        log(String(
          format: "HARNESS spec=%@: decode %.1f tok/s · prefill %.1f tok/s · init %.1fs · %@",
          spec ? "ON " : "off", bi.lastDecodeTokensPerSecond, bi.lastPrefillTokensPerSecond,
          bi.initTimeInSecond, mb(LiteRTChat.memoryFootprintBytes())))
      } catch {
        log("HARNESS spec=\(spec) FAILED: \(error.localizedDescription)")
      }
      try? await Task.sleep(nanoseconds: 800_000_000)
    }
    ExperimentalFlags.enableSpeculativeDecoding = false

    // Real autoregressive generation, wall-clock, benchmark OFF (benchmark mode
    // + respond crashes with output_buffer_dup on this build). This is the true
    // test of whether the MTP drafter speeds up real decoding. words/s ≈ 0.75 ×
    // tokens/s, so a ~2× jump with spec ON would mean MTP is working.
    let prompt = "Write a detailed explanation of how a bicycle works, in about 150 words."
    for spec in [false, true] {
      do {
        let chat = try await LiteRTChat(
          model, modalities: [] as Modality, speculativeDecoding: spec)
        _ = try? await chat.respond("Hi.")  // warm up
        let start = Date()
        let out = try await chat.respond(prompt)
        let wall = Date().timeIntervalSince(start)
        let words = out.split(whereSeparator: { $0 == " " || $0 == "\n" }).count
        log(String(
          format: "GEN spec=%@: %.1f words/s · ~%d words · wall %.1fs · %@",
          spec ? "ON " : "off", Double(words) / max(wall, 0.001), words, wall,
          mb(LiteRTChat.memoryFootprintBytes())))
        try? await Task.sleep(nanoseconds: 800_000_000)
      } catch {
        log("GEN spec=\(spec) FAILED: \(error.localizedDescription)")
      }
    }

    log("DONE")
  }
}
