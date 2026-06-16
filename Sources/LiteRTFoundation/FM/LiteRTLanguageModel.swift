// swift-litert-lm — Foundation Models backend (FM mode)
//
// `LiteRTLanguageModel` makes LiteRT-LM a first-class Apple Foundation Models
// backend, alongside Apple's own `CoreAILanguageModel` and `MLXLanguageModel`:
//
//   let model   = try await LiteRTLanguageModel(.gemma4_E2B)
//   let session = LanguageModelSession(model: model)          // Apple's exact API
//   let answer  = try await session.respond(to: "Hi")          // streaming / tools / @Generable
//
// The FM API is transcript-based (each turn hands the executor the full
// conversation), while LiteRT-LM is stateful (a `Conversation` accumulates its
// own KV cache). We bridge by rebuilding a fresh LiteRT `Conversation` from the
// transcript on each turn — correct and simple; an incremental fast-path is a
// later optimization.

#if canImport(FoundationModels)

import Foundation
import FoundationModels
import CoreGraphics
import LiteRTLM

/// A LiteRT-LM model exposed as an Apple Foundation Models backend.
@available(iOS 27.0, macOS 27.0, *)
public struct LiteRTLanguageModel: LanguageModel {
  public typealias Executor = LiteRTExecutor

  public let capabilities: LanguageModelCapabilities
  public let executorConfiguration: LiteRTExecutor.Configuration

  /// Create the backend, downloading the model on first use.
  ///
  /// - Parameters:
  ///   - model: Which catalog model to run.
  ///   - storageDirectory: Where to keep the downloaded model (defaults to
  ///     Application Support/LiteRTModels).
  ///   - onDownloadProgress: Called on first run while the model downloads.
  public init(
    _ model: LiteRTModel,
    storageDirectory: URL? = nil,
    onDownloadProgress: (@Sendable (ModelDownloader.Progress) -> Void)? = nil
  ) async throws {
    let path = try await LiteRTChat.ensureModel(
      model, storageDirectory: storageDirectory, onProgress: onDownloadProgress)
    self.executorConfiguration = LiteRTExecutor.Configuration(model: model, modelPath: path)
    // Declared capabilities: guided generation (best-effort schema-in-prompt; see
    // the executor) and vision (gates image attachments). Audio/video ride the
    // custom-segment hook and are not capability-gated.
    var capabilities: [LanguageModelCapabilities.Capability] = [.guidedGeneration]
    if model.supportedModalities.contains(.vision) { capabilities.append(.vision) }
    self.capabilities = LanguageModelCapabilities(capabilities: capabilities)
  }
}

/// Drives generation for `LiteRTLanguageModel` over the FM executor protocol.
@available(iOS 27.0, macOS 27.0, *)
public final class LiteRTExecutor: LanguageModelExecutor {
  public typealias Model = LiteRTLanguageModel

  /// Lightweight, `Hashable` description of what engine to build. The actual
  /// (async-initialized) engine is created lazily by the executor.
  public struct Configuration: Hashable, Sendable {
    public let model: LiteRTModel
    public let modelPath: String
    public init(model: LiteRTModel, modelPath: String) {
      self.model = model
      self.modelPath = modelPath
    }
  }

  private let engine: LazyEngine

  public init(configuration: Configuration) throws {
    self.engine = LazyEngine(configuration: configuration)
  }

  public func prewarm(model: Model, transcript: Transcript) {
    // Kick off engine creation + a tiny warmup so the first real turn is fast.
    Task { try? await engine.prewarmed() }
  }

  public func respond(
    to request: LanguageModelExecutorGenerationRequest,
    model: Model,
    streamingInto channel: LanguageModelExecutorGenerationChannel
  ) async throws {
    let engine = try await self.engine.ready()
    // Guided generation (G2): if the request carries a schema, encode it to JSON
    // and steer the model toward it via the prompt (schema-in-prompt). FM parses
    // the model's JSON into the @Generable type. This is soft guidance; hard
    // constrained decoding (llguidance) is a follow-up.
    let schemaJSON = request.schema.flatMap { try? Self.encodeSchema($0) }
    let plan = try Self.plan(from: request.transcript, schemaJSON: schemaJSON)

    let conversation = try await engine.createConversation(
      with: ConversationConfig(
        systemMessage: plan.systemMessage,
        initialMessages: plan.history,
        samplerConfig: try? SamplerConfig(topK: 40, topP: 0.95, temperature: 0.8)))

    for try await chunk in conversation.sendMessageStream(plan.prompt) {
      let delta = chunk.toString
      if !delta.isEmpty {
        await channel.send(.response(action: .appendText(delta, tokenCount: 1)))
      }
    }
  }

  // MARK: Transcript → LiteRT messages

  private struct Plan {
    let systemMessage: Message?
    let history: [Message]
    let prompt: Message
  }

  /// Split the FM transcript into a system message, prior turns (history), and
  /// the final user prompt that this turn should answer. When `schemaJSON` is
  /// given, schema guidance is appended to the final prompt.
  private static func plan(from transcript: Transcript, schemaJSON: String?) throws -> Plan {
    let entries = Array(transcript)
    guard
      let lastPromptIndex = entries.lastIndex(where: {
        if case .prompt = $0 { return true } else { return false }
      })
    else {
      throw LiteRTFMError.noPrompt
    }

    var systemText: [String] = []
    var history: [Message] = []
    var prompt: Message?

    for (i, entry) in entries.enumerated() {
      switch entry {
      case .instructions(let instructions):
        systemText.append(text(of: instructions.segments))
      case .prompt(let p):
        if i == lastPromptIndex {
          var c = contents(of: p.segments)
          if let schemaJSON, !schemaJSON.isEmpty {
            c.append(
              .text(
                "\n\nRespond with ONLY a JSON object that conforms to this JSON schema. "
                  + "Output valid JSON and nothing else:\n\(schemaJSON)"))
          }
          prompt = Message(contents: c, role: .user)
        } else {
          history.append(Message(contents: contents(of: p.segments), role: .user))
        }
      case .response(let r):
        history.append(Message(contents: [.text(text(of: r.segments))], role: .model))
      case .toolCalls, .toolOutput, .reasoning:
        break  // not handled in this phase
      @unknown default:
        break
      }
    }

    let system = systemText.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    return Plan(
      systemMessage: system.isEmpty ? nil : Message(system, role: .system),
      history: history,
      prompt: prompt!  // guaranteed by lastPromptIndex
    )
  }

  /// Concatenate the text of a segment list (non-text segments ignored for now).
  private static func text(of segments: [Transcript.Segment]) -> String {
    segments.compactMap { segment in
      if case .text(let t) = segment { return t.content } else { return nil }
    }.joined(separator: " ")
  }

  /// Encode an FM `GenerationSchema` to a JSON Schema string (it's `Codable`).
  private static func encodeSchema(_ schema: GenerationSchema) throws -> String {
    let data = try JSONEncoder().encode(schema)
    return String(data: data, encoding: .utf8) ?? ""
  }

  /// Map FM segments to LiteRT content: text, image attachments, and audio via
  /// the `LiteRTAudioSegment` custom segment.
  private static func contents(of segments: [Transcript.Segment]) -> [Content] {
    var out: [Content] = []
    for segment in segments {
      switch segment {
      case .text(let t):
        if !t.content.isEmpty { out.append(.text(t.content)) }
      case .attachment(let attachment):
        if case .image(let image) = attachment.content, let png = pngData(from: image.cgImage) {
          out.append(.imageData(png))
        }
      case .custom(let custom):
        if let audio = custom as? LiteRTAudioSegment {
          out.append(.audioData(audio.content.data))
        } else if let video = custom as? LiteRTVideoSegment {
          out.append(contentsOf: video.content.frames.map { Content.imageData($0) })
        }
      case .structure:
        break  // structured (guided-generation) content — a later phase
      @unknown default:
        break
      }
    }
    return out.isEmpty ? [.text("")] : out
  }
}

/// Errors specific to the Foundation Models bridge.
@available(iOS 27.0, macOS 27.0, *)
public enum LiteRTFMError: Error, LocalizedError {
  case noPrompt

  public var errorDescription: String? {
    switch self {
    case .noPrompt: return "The transcript contains no prompt to respond to."
    }
  }
}

/// Lazily creates and caches the LiteRT engine. The FM executor's `init` is
/// synchronous but engine initialization is async, so we defer it to the first
/// `respond` (which is async) and memoize the result.
@available(iOS 27.0, macOS 27.0, *)
private actor LazyEngine {
  private let configuration: LiteRTExecutor.Configuration
  private var engine: Engine?
  private var warmed = false

  init(configuration: LiteRTExecutor.Configuration) {
    self.configuration = configuration
  }

  func ready() async throws -> Engine {
    if let engine { return engine }
    let model = configuration.model
    // Bring up the vision + audio towers so image attachments and audio custom
    // segments work through the FM API. Backends come from the catalog (Gemma 4
    // E2B: both CPU — vision Metal fails STABLEHLO_COMPOSITE, audio is CPU-only).
    ExperimentalFlags.optIntoExperimentalAPIs()
    if let budget = model.defaultVisualTokenBudget { ExperimentalFlags.visualTokenBudget = budget }
    let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
    let config = try EngineConfig(
      modelPath: configuration.modelPath, backend: .gpu,
      visionBackend: model.supportedModalities.contains(.vision) ? model.visionBackend : nil,
      audioBackend: model.supportedModalities.contains(.audio) ? model.audioBackend : nil,
      maxNumTokens: model.defaultMaxTokens, cacheDir: caches?.path)
    let created = Engine(engineConfig: config)
    try await created.initialize()
    engine = created
    return created
  }

  func prewarmed() async throws {
    let engine = try await ready()
    if warmed { return }
    warmed = true
    let warmup = try await engine.createConversation()
    for try await _ in warmup.sendMessageStream(Message("Hi")) {}
  }
}

#endif
