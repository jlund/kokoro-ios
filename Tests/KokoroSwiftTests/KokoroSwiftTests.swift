import MLX
import Testing
@testable import KokoroSwift

/// Reference implementation of the alignment matrix: the original
/// per-frame CPU loop that `createAlignmentTarget` replaced. Kept here so
/// the vectorized version is checked against known-good output.
private func referenceAlignmentTarget(durations: MLXArray, batchSize: Int) -> MLXArray {
  let indices = MLX.concatenated(
    durations.enumerated().map { index, duration in
      let frameCount: Int = duration.item()
      return MLX.repeated(MLXArray([index]), count: frameCount)
    }
  )

  let totalFrames = indices.shape[0]
  var alignmentArray = [Float](repeating: 0.0, count: totalFrames * batchSize)

  for frame in 0 ..< totalFrames {
    let phonemeIndex: Int = indices[frame].item()
    alignmentArray[phonemeIndex * totalFrames + frame] = 1.0
  }

  let alignmentTarget = MLXArray(alignmentArray).reshaped([batchSize, totalFrames])
  return alignmentTarget.expandedDimensions(axis: 0)
}

@Test(arguments: [
  [1, 3, 2, 5, 1],
  [1],
  [1, 1, 1, 1],
  [12, 1, 7, 30, 2, 2, 9],
])
func alignmentTargetMatchesReference(durationValues: [Int]) {
  let durations = MLXArray(durationValues.map { Int32($0) })
  let batchSize = durationValues.count

  let vectorized = KokoroTTS.createAlignmentTarget(durations: durations, batchSize: batchSize)
  let reference = referenceAlignmentTarget(durations: durations, batchSize: batchSize)

  #expect(vectorized.shape == reference.shape)
  #expect(MLX.allClose(vectorized, reference).item(Bool.self))
}

@Test func alignmentTargetLargeRandomDurations() {
  // Frame counts comparable to a real 500-char chunk (thousands of frames).
  var seed: UInt64 = 0x9E3779B97F4A7C15
  func nextDuration() -> Int32 {
    seed = seed &* 6364136223846793005 &+ 1442695040888963407
    return Int32(seed % 18) + 1
  }
  let durationValues = (0 ..< 400).map { _ in nextDuration() }
  let durations = MLXArray(durationValues)

  let vectorized = KokoroTTS.createAlignmentTarget(durations: durations, batchSize: durationValues.count)
  let reference = referenceAlignmentTarget(durations: durations, batchSize: durationValues.count)

  #expect(vectorized.shape == reference.shape)
  #expect(MLX.allClose(vectorized, reference).item(Bool.self))
}
