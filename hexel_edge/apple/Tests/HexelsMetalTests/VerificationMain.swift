import Darwin
import Foundation
import Metal

struct VerificationFailure: Error, CustomStringConvertible {
    let description: String
}

struct VerificationSkip: Error {
    let reason: String
    init(_ reason: String) { self.reason = reason }
}

func expect(_ condition: Bool, _ message: String = "Expected true", file: String = #filePath, line: Int = #line) throws {
    if !condition { throw VerificationFailure(description: "\(file):\(line): \(message)") }
}

func expectEqual<T: Equatable>(_ actual: T, _ expected: T, _ message: String = "", file: String = #filePath, line: Int = #line) throws {
    try expect(actual == expected, "\(message) expected \(expected), got \(actual)", file: file, line: line)
}

func expectEqual<T: BinaryFloatingPoint>(
    _ actual: T, _ expected: T, accuracy: T, _ message: String = "",
    file: String = #filePath, line: Int = #line
) throws {
    try expect(actual.isFinite && expected.isFinite && abs(actual - expected) <= accuracy,
               "\(message) expected \(expected) ± \(accuracy), got \(actual)", file: file, line: line)
}

func expectGreaterThan<T: Comparable>(_ actual: T, _ expected: T, _ message: String = "", file: String = #filePath, line: Int = #line) throws {
    try expect(actual > expected, "\(message) expected \(actual) > \(expected)", file: file, line: line)
}

func expectLessThan<T: Comparable>(_ actual: T, _ expected: T, _ message: String = "", file: String = #filePath, line: Int = #line) throws {
    try expect(actual < expected, "\(message) expected \(actual) < \(expected)", file: file, line: line)
}

func requireValue<T>(_ value: T?, file: String = #filePath, line: Int = #line) throws -> T {
    guard let value else { throw VerificationFailure(description: "\(file):\(line): Required value is nil") }
    return value
}

func expectThrows<T>(_ expression: @autoclosure () throws -> T, file: String = #filePath, line: Int = #line) throws {
    do { _ = try expression() } catch { return }
    throw VerificationFailure(description: "\(file):\(line): Expected an error, but operation succeeded")
}

func expectNoThrow<T>(_ expression: @autoclosure () throws -> T) throws {
    _ = try expression()
}

/// Executable checks work with Command Line Tools, which need not ship XCTest.
@main
struct VerificationMain {
    static func main() {
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments.contains("--help") {
            print("hexels-tests [--require-metal] [--filter substring]")
            exit(0)
        }
        var filter: String?
        var requireMetal = false
        var index = 0
        while index < arguments.count {
            switch arguments[index] {
            case "--require-metal": requireMetal = true
            case "--filter":
                index += 1
                guard index < arguments.count else { print("ERROR: --filter requires a substring"); exit(2) }
                filter = arguments[index]
            default: print("ERROR: unknown argument \(arguments[index])"); exit(2)
            }
            index += 1
        }
        let renderer = RendererTests()
        let bundle = SceneBundleTests()
        let capture = CaptureGeometryTests()
        let tests: [(String, () throws -> Void)] = [
            ("reference-center", renderer.testIndependentReferenceAtFrontFacingCenter),
            ("metal-reference-occlusion", renderer.testMetalMatchesIndependentSpatialColorAndOcclusionReference),
            ("all-20-gradients", renderer.testAllTwentyAnalyticGradientsMatchIndependentFiniteDifferences),
            ("occluded-plane-gradients", renderer.testOccludedPlaneGradientsMatchIndependentFiniteDifferences),
            ("partial-edge-tile-gradients", renderer.testPartialEdgeTilesMatchReferenceAndAllOccludedGradients),
            ("unclamped-color", renderer.testRawSpatialColorRemainsUnclamped),
            ("near-clipping", renderer.testBackgroundAndNearClipping),
            ("early-termination", renderer.testEarlyTerminationSkipsThresholdCrossingPrimitive),
            ("alpha-softness-branches", renderer.testAlphaSaturationAndSoftnessFloorStopTheirGradients),
            ("alpha-cutoff-tile-coverage", renderer.testThresholdEqualityHasConservativeTileCoverage),
            ("metal-allocation-lifetime", renderer.testRepeatedTrainingDoesNotRetainPerStepMetalAllocations),
            ("optimizer-checkpoint-resume", renderer.testOptimizerReducesLossAndCheckpointPreservesNextUpdate),
            ("invalid-parameters-camera", renderer.testInvalidParametersAndCameraAreRejected),
            ("resource-budgets", renderer.testResourceBudgetsRejectBeforeUnboundedWorkspaceAllocation),
            ("bundle-image-orientation", bundle.testBundleRoundTripAndCalibratedImageOrientation),
            ("bundle-schema-truncation", bundle.testUnsupportedSchemaAndTruncatedParametersAreRejected),
            ("bundle-path-symlink-escape", bundle.testEscapingAssetPathsAndSymlinksAreRejected),
            ("bundle-heldout-view-order", bundle.testHeldOutAndAbsentViewOrderEntriesAreRejected),
            ("bundle-image-size", bundle.testImageSizeMismatchIsRejectedBeforeTraining),
            ("bundle-calibration", bundle.testNonRigidAndInvalidCalibrationAreRejected),
            ("bundle-recipe-identity", bundle.testRecipeHintsAreValidatedAndBoundToTheBundle),
            ("capture-pose-pixel-convention", capture.testARCameraAxesAndHalfPixelResize),
            ("capture-camera-interpolation", capture.testNovelCameraInterpolationEndpointsShortestArcAndAdmission),
            ("capture-depth-plane", capture.testRegisteredDepthPlaneNormalsAndColor),
            ("capture-depth-confidence-bounds", capture.testDepthHolesConfidenceAndSamplingBounds),
            ("capture-sparse-observed-color", capture.testSparsePointsUseObservedColorsAndFacingPrior),
            ("capture-voxel-seed-bounds", capture.testVoxelSeedsBoundedFiniteAndNormalAligned),
            ("capture-hexel-hard-surface", capture.testHexelSurfaceMatchesHardSupportAndGradientColor),
            ("capture-mesh-stride-indices", capture.testMeshStrideOffsetsTransformsAndIndexAdmission),
            ("capture-surface-ply-obj", capture.testPLYOBJRoundTripCountsIndicesNormalsAndColors),
        ].filter { filter == nil || $0.0.contains(filter!) }
        guard !tests.isEmpty else { print("ERROR: no checks match the requested filter"); exit(2) }
        var passed = 0, failed = 0, skipped = 0
        let start = Date()
        for (name, run) in tests {
            do { try run(); passed += 1; print("PASS \(name)") }
            catch let skip as VerificationSkip { skipped += 1; print("SKIP \(name): \(skip.reason)") }
            catch { failed += 1; print("FAIL \(name): \(error)") }
        }
        if requireMetal && MTLCreateSystemDefaultDevice() == nil {
            failed += 1
            print("FAIL required Metal device is unavailable; GPU checks may not be counted as passing")
        }
        print(String(format: "Checks: %d passed, %d failed, %d skipped (%.3f s)",
                     passed, failed, skipped, Date().timeIntervalSince(start)))
        exit(failed == 0 ? 0 : 1)
    }
}
