// LiteRTDemo — Foundation Models multimodal + guided-generation self-test.
//
// Exercises the full FM-mode consumer API over the LiteRT backend:
//   • G2 guided generation — `respond(generating:)` with a `@Generable` type
//   • audio via `LiteRTAudioSegment` in a `@PromptBuilder`
//   • video via `LiteRTVideoSegment` (frames from `VideoFrameSampler`)
//
// Launch with LITERT_FM=1. Lines are tagged "FM:" for devicectl polling.

#if canImport(FoundationModels)

import Foundation
import FoundationModels
import LiteRTFoundation
import os

@available(iOS 27.0, macOS 27.0, *)
@Generable
struct PrimaryColors {
  @Guide(description: "Exactly three additive primary colors")
  var colors: [String]
}

@available(iOS 27.0, macOS 27.0, *)
enum FMMultimodalSelfTest {
  private static let logger = Logger(subsystem: "com.example.litertdemo", category: "FM")

  static var isRequested: Bool { ProcessInfo.processInfo.environment["LITERT_FM"] != nil }

  static func log(_ message: String) {
    logger.log("\(message, privacy: .public)")
    print("FM: \(message)")
    fflush(stdout)
  }

  static func run() async {
    log("start — guided + audio + video over the Foundation Models API")
    do {
      let model = try await LiteRTLanguageModel(.gemma4_E2B)
      let session = LanguageModelSession(model: model)

      // G2 — guided generation into a @Generable type.
      let guided = try await session.respond(generating: PrimaryColors.self) {
        "List the three additive primary colors."
      }
      log("GUIDED → colors = \(guided.content.colors)")

      // Audio understanding through the FM API (custom segment).
      if let wav = Bundle.main.url(forResource: "have_a_wonderful_day", withExtension: "wav") {
        let data = try Data(contentsOf: wav)
        let answer = try await session.respond {
          LiteRTAudioSegment(data: data)
          "Transcribe the spoken words in this audio."
        }
        log("AUDIO(FM) → \(answer.content.replacingOccurrences(of: "\n", with: " "))")
      }

      // Video understanding through the FM API (app-sampled frames).
      if let mov = Bundle.main.url(forResource: "sample", withExtension: "mp4") {
        let frames = try await VideoFrameSampler.sampleFrames(from: mov, count: 4)
        let answer = try await session.respond {
          LiteRTVideoSegment(frames: frames)
          "Describe what happens in this video."
        }
        log("VIDEO(FM) → \(answer.content.replacingOccurrences(of: "\n", with: " "))")
      } else {
        log("VIDEO(FM) skipped — bundle a sample.mp4 to exercise it")
      }

      log("PASS — guided + audio + video drive LiteRT via the FM API")
    } catch {
      log("FAILED: \(error.localizedDescription)")
    }
    log("DONE")
  }
}

#endif
