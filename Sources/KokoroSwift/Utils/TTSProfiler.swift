//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import os

/// Per-stage wall-clock profiling for `KokoroTTS.generateAudio`.
///
/// MLX builds a lazy graph, so bracketing a stage with timestamps alone would
/// measure graph construction, not execution. When `enabled` is true, each
/// stage boundary forces `MLX.eval` on that stage's outputs so the reported
/// times are real. Forcing evaluation serializes the graph, which adds some
/// overhead — totals with the profiler on can run slightly higher than
/// production, so keep it off outside benchmarking sessions.
///
/// Timing summaries are emitted through `os.Logger` at notice level
/// (subsystem `com.binaryteaparty.kokoro`), which persists in Release builds:
/// `log stream --predicate 'subsystem == "com.binaryteaparty.kokoro"'`
public enum TTSProfiler {
  /// Enables per-stage evaluation and timing summaries. Off by default.
  /// Set once at app startup, before any generation runs — not synchronized.
  nonisolated(unsafe) public static var enabled = false

  static let signposter = OSSignposter(subsystem: "com.binaryteaparty.kokoro", category: "TTSStages")
  static let log = Logger(subsystem: "com.binaryteaparty.kokoro", category: "TTSTiming")
}

/// Accumulates stage timings for a single `generateAudio` call.
/// All methods are no-ops when `TTSProfiler.enabled` is false.
final class TTSStageClock {
  private var entries: [(name: String, ms: Double)] = []
  private let callStart = ContinuousClock.now
  private var lastMark = ContinuousClock.now
  private let signpostID: OSSignpostID
  private var callInterval: OSSignpostIntervalState?

  init() {
    signpostID = TTSProfiler.signposter.makeSignpostID()
    if TTSProfiler.enabled {
      callInterval = TTSProfiler.signposter.beginInterval("generateAudio", id: signpostID)
    }
  }

  /// Closes the stage that ran since the previous mark, forcing evaluation of
  /// its outputs so the elapsed time reflects actual GPU/CPU work.
  func mark(_ name: StaticString, eval arrays: [MLXArray] = []) {
    guard TTSProfiler.enabled else { return }
    if !arrays.isEmpty {
      MLX.eval(arrays)
    }
    let now = ContinuousClock.now
    entries.append((name: "\(name)", ms: ms(from: lastMark, to: now)))
    TTSProfiler.signposter.emitEvent(name, id: signpostID)
    lastMark = now
  }

  /// Emits the one-line summary for this call and ends the signpost interval.
  func finish(tokenCount: Int, frameCount: Int, sampleCount: Int) {
    guard TTSProfiler.enabled else { return }
    if let callInterval {
      TTSProfiler.signposter.endInterval("generateAudio", callInterval)
    }
    let total = ms(from: callStart, to: ContinuousClock.now)
    let audioSeconds = Double(sampleCount) / Double(KokoroTTS.Constants.samplingRate)
    let stages = entries.map { String(format: "%@=%.0fms", $0.name, $0.ms) }.joined(separator: " ")
    let summary = String(
      format: "tokens=%d frames=%d audio=%.2fs total=%.0fms rtf=%.2f %@",
      tokenCount, frameCount, audioSeconds, total,
      audioSeconds > 0 ? (total / 1000.0) / audioSeconds : 0,
      stages
    )
    TTSProfiler.log.notice("chunk \(summary, privacy: .public)")
  }

  private func ms(from: ContinuousClock.Instant, to: ContinuousClock.Instant) -> Double {
    let duration = to - from
    let (seconds, attoseconds) = duration.components
    return Double(seconds) * 1000.0 + Double(attoseconds) / 1e15
  }
}
