import Foundation
import Metal
import HexelsMetal

final class RendererTests {
    private func requireMetal() throws {
        if MTLCreateSystemDefaultDevice() == nil {
            throw VerificationSkip("A physical Metal device is required for GPU verification")
        }
    }

    private func camera(_ reference: ReferenceCamera) -> HexelCamera {
        HexelCamera(width: reference.width, height: reference.height,
                    intrinsics: reference.intrinsics, cameraToWorld: reference.cameraToWorld,
                    near: Float(reference.near))
    }

    private let background = SIMD3<Float>(0.07, 0.11, 0.17)

    func testIndependentReferenceAtFrontFacingCenter() throws {
        var p = IndependentReference.fixture()
        p[0] = 0; p[1] = 0; p[5] = 1; p[6] = 0; p[7] = 0; p[8] = 0
        var c = ReferenceCamera()
        c.origin = .zero
        let center = IndependentReference.render(parameters: p, camera: c)[c.height / 2 * c.width + c.width / 2]
        let alpha = IndependentReference.sigmoid(p[18]) * IndependentReference.sigmoid(exp(p[19]))
        try expectEqual(center.w, alpha, accuracy: 1e-12)
        try expectEqual(center.x, alpha * p[9] + (1 - alpha) * 0.07, accuracy: 1e-12)
        try expectEqual(center.y, alpha * p[10] + (1 - alpha) * 0.11, accuracy: 1e-12)
        try expectEqual(center.z, alpha * p[11] + (1 - alpha) * 0.17, accuracy: 1e-12)
    }

    func testMetalMatchesIndependentSpatialColorAndOcclusionReference() throws {
        try requireMetal()
        let c = ReferenceCamera()
        var back = IndependentReference.fixture(z: 3.1)
        back[0] = -0.13; back[9] = 0.17; back[10] = 0.6; back[11] = 0.8
        // Deliberately reverse input order: visibility is determined by center depth.
        let p = back + IndependentReference.fixture()
        let engine = try MetalHexelTrainer(parameters: p.map(Float.init))
        let actual = try engine.render(camera: camera(c), background: background)
        let expected = IndependentReference.render(parameters: p, camera: c)
        try expectEqual(actual.count, c.width * c.height)
        for (pixel, values) in zip(actual, expected).enumerated() {
            for channel in 0..<4 {
                try expectEqual(Double(values.0[channel]), values.1[channel], accuracy: 3e-5,
                               "pixel \(pixel), channel \(channel)")
            }
        }
        var rotated = c
        let angle = 0.17
        rotated.right = SIMD3(cos(angle), 0, -sin(angle))
        rotated.forward = SIMD3(sin(angle), 0, cos(angle))
        let rotatedActual = try engine.render(camera: camera(rotated), background: background)
        let rotatedExpected = IndependentReference.render(parameters: p, camera: rotated)
        for (pixel, values) in zip(rotatedActual, rotatedExpected).enumerated() {
            for channel in 0..<4 {
                try expectEqual(Double(values.0[channel]), values.1[channel], accuracy: 3e-5,
                                "rotated camera pixel \(pixel), channel \(channel)")
            }
        }
    }

    func testAllTwentyAnalyticGradientsMatchIndependentFiniteDifferences() throws {
        try requireMetal()
        let c = ReferenceCamera()
        let p = IndependentReference.fixture()
        let target = IndependentReference.target(camera: c)
        let engine = try MetalHexelTrainer(parameters: p.map(Float.init))
        let actual = try engine.lossAndGradients(
            camera: camera(c), target: target.map { SIMD4<Float>($0) }, background: background)
        let expectedLoss = IndependentReference.loss(
            IndependentReference.render(parameters: p, camera: c), target: target)
        try expectEqual(Double(actual.loss), expectedLoss, accuracy: 2e-6)
        try expectEqual(actual.gradients.count, 20)
        for index in 0..<20 {
            let numerical = IndependentReference.finiteDifference(
                parameters: p, parameter: index, camera: c, target: target)
            try expect(numerical.isFinite)
            try expectGreaterThan(abs(numerical), 1e-8, "Fixture must exercise parameter \(index)")
            try expectEqual(Double(actual.gradients[index]), numerical,
                           accuracy: max(2e-5, abs(numerical) * 0.025), "parameter \(index)")
        }
    }

    func testOccludedPlaneGradientsMatchIndependentFiniteDifferences() throws {
        try requireMetal()
        let c = ReferenceCamera()
        var back = IndependentReference.fixture(z: 3.1)
        back[0] = -0.13; back[9] = 0.17; back[10] = 0.6; back[11] = 0.8
        let p = back + IndependentReference.fixture()
        let target = IndependentReference.target(camera: c)
        let engine = try MetalHexelTrainer(parameters: p.map(Float.init))
        let actual = try engine.lossAndGradients(
            camera: camera(c), target: target.map { SIMD4<Float>($0) }, background: background)
        try expectEqual(actual.gradients.count, 40)
        for index in 0..<40 {
            let numerical = IndependentReference.finiteDifference(
                parameters: p, parameter: index, camera: c, target: target)
            try expectEqual(Double(actual.gradients[index]), numerical,
                            accuracy: max(3e-5, abs(numerical) * 0.03), "occluded parameter \(index)")
        }
    }

    func testPartialEdgeTilesMatchReferenceAndAllOccludedGradients() throws {
        try requireMetal()
        var c = ReferenceCamera()
        c.width = 33; c.height = 19
        c.fx = 27; c.fy = 25; c.cx = 15.7; c.cy = 8.6
        var front = IndependentReference.fixture()
        front[3] = log(1.1); front[4] = log(0.7)
        var edge = IndependentReference.fixture(z: 2.9)
        edge[0] = 1.6; edge[1] = 0.86
        edge[3] = log(0.65); edge[4] = log(0.55)
        edge[9] = 0.17; edge[10] = 0.6; edge[11] = 0.8
        // The right/bottom primitive reaches the 1-column and 3-row partial
        // threadgroups. Its left edge overlaps the larger foreground patch.
        let p = edge + front
        let expected = IndependentReference.render(parameters: p, camera: c)
        try expectGreaterThan(expected[(c.height - 1) * c.width + c.width - 1].w, 0.01,
                              "The partial bottom-right group must contain visible work")
        try expect(expected.contains(where: { $0.w == 0 }), "Fixture must also exercise padded/background lanes")
        let engine = try MetalHexelTrainer(parameters: p.map(Float.init))
        let actual = try engine.render(camera: camera(c), background: background)
        for (pixel, values) in zip(actual, expected).enumerated() {
            for channel in 0..<4 {
                try expectEqual(Double(values.0[channel]), values.1[channel], accuracy: 4e-5,
                                "partial-tile pixel \(pixel), channel \(channel)")
            }
        }
        let target = IndependentReference.target(camera: c)
        let result = try engine.lossAndGradients(
            camera: camera(c), target: target.map { SIMD4<Float>($0) }, background: background)
        try expectEqual(result.gradients.count, 40)
        for index in 0..<40 {
            let numerical = IndependentReference.finiteDifference(
                parameters: p, parameter: index, camera: c, target: target)
            try expectEqual(Double(result.gradients[index]), numerical,
                            accuracy: max(1.5e-5, abs(numerical) * 0.03), "partial-tile parameter \(index)")
        }
    }

    func testRawSpatialColorRemainsUnclamped() throws {
        try requireMetal()
        var c = ReferenceCamera()
        c.width = 1; c.height = 1; c.cx = 0; c.cy = 0; c.origin = .zero
        var p = IndependentReference.fixture()
        p[0] = 0; p[1] = 0; p[5] = 1; p[6] = 0; p[7] = 0; p[8] = 0
        p[9] = 1.8; p[10] = -0.4
        let engine = try MetalHexelTrainer(parameters: p.map(Float.init))
        let pixel = try engine.render(camera: camera(c), background: background)[0]
        try expectGreaterThan(pixel.x, 1, "Display clamping must not change the training renderer")
        try expectLessThan(pixel.y, 0)
    }

    func testBackgroundAndNearClipping() throws {
        try requireMetal()
        let c = ReferenceCamera()
        var p = IndependentReference.fixture(z: 0.1)
        p[5] = 1; p[6] = 0; p[7] = 0; p[8] = 0
        let engine = try MetalHexelTrainer(parameters: p.map(Float.init))
        let actual = try engine.render(camera: camera(c), background: background)
        for pixel in actual {
            try expectEqual(pixel.x, background.x, accuracy: 1e-7)
            try expectEqual(pixel.y, background.y, accuracy: 1e-7)
            try expectEqual(pixel.z, background.z, accuracy: 1e-7)
            try expectEqual(pixel.w, 0, accuracy: 1e-7)
        }
    }

    func testEarlyTerminationSkipsThresholdCrossingPrimitive() throws {
        try requireMetal()
        var c = ReferenceCamera()
        c.width = 1; c.height = 1; c.cx = 0; c.cy = 0; c.origin = .zero
        var p: [Double] = []
        for index in 0..<3 {
            var plane = IndependentReference.fixture(z: 2 + Double(index) * 0.5)
            plane[0] = 0; plane[1] = 0
            plane[5] = 1; plane[6] = 0; plane[7] = 0; plane[8] = 0
            plane[9] = Double(index + 1) * 0.2
            plane[18] = log(0.985 / 0.015); plane[19] = log(30)
            p += plane
        }
        let engine = try MetalHexelTrainer(parameters: p.map(Float.init))
        let actual = try engine.render(camera: camera(c), background: background)[0]
        let expected = IndependentReference.render(parameters: p, camera: c)[0]
        for channel in 0..<4 { try expectEqual(Double(actual[channel]), expected[channel], accuracy: 3e-6) }
        try expectEqual(Double(actual.w), 1 - 0.015 * 0.015, accuracy: 3e-6)
    }

    func testAlphaSaturationAndSoftnessFloorStopTheirGradients() throws {
        try requireMetal()
        var c = ReferenceCamera()
        c.width = 1; c.height = 1; c.cx = 0; c.cy = 0; c.origin = .zero
        var p = IndependentReference.fixture()
        p[0] = 0; p[1] = 0; p[5] = 1; p[6] = 0; p[7] = 0; p[8] = 0
        p[18] = 20; p[19] = log(30)
        let saturated = try MetalHexelTrainer(parameters: p.map(Float.init))
        let capped = try saturated.lossAndGradients(camera: camera(c), target: [.zero], background: background)
        try expectEqual(try saturated.render(camera: camera(c), background: background)[0].w, 0.99, accuracy: 1e-6)
        try expectEqual(capped.gradients[18], 0, accuracy: 1e-8)
        try expectEqual(capped.gradients[19], 0, accuracy: 1e-8)
        p[18] = 0; p[19] = -20
        let floored = try MetalHexelTrainer(parameters: p.map(Float.init))
        let gradient = try floored.lossAndGradients(camera: camera(c), target: [.zero], background: background)
        try expectEqual(gradient.gradients[19], 0, accuracy: 1e-8)
        try expectGreaterThan(abs(gradient.gradients[18]), 1e-6)
    }

    func testThresholdEqualityHasConservativeTileCoverage() throws {
        try requireMetal()
        var c = ReferenceCamera()
        c.width = 1; c.height = 1; c.cx = 0; c.cy = 0; c.origin = .zero
        var p = IndependentReference.fixture()
        p[0] = 0; p[1] = 0; p[5] = 1; p[6] = 0; p[7] = 0; p[8] = 0
        p[18] = -5.399971961975098
        p[19] = 0.6524151563644409
        let engine = try MetalHexelTrainer(parameters: p.map(Float.init))
        _ = try engine.render(camera: camera(c), background: background)
        try expectEqual(engine.lastIntersectionCount, 1,
                        "Conservative binning must retain cutoff-adjacent support despite CPU/GPU exp rounding")
        // This exact boundary can round below cutoff on Metal and at cutoff on
        // the CPU. Two Float ULPs above it must be visible, without redefining
        // the renderer's discontinuous alpha acceptance rule in the test.
        p[18] = Double(Float(p[18]).nextUp.nextUp)
        let visible = try MetalHexelTrainer(parameters: p.map(Float.init))
        let pixel = try visible.render(camera: camera(c), background: background)[0]
        try expectGreaterThan(pixel.w, 0, "An accepted near-cutoff sample must survive tile binning")
        try expectEqual(pixel.w, 1 / Float(255), accuracy: 2e-7)
    }

    func testRepeatedTrainingDoesNotRetainPerStepMetalAllocations() throws {
        try requireMetal()
        let c = ReferenceCamera()
        let p = IndependentReference.fixture()
        let target = IndependentReference.target(camera: c).map { SIMD4<Float>($0) }
        let engine = try MetalHexelTrainer(parameters: p.map(Float.init))
        for _ in 0..<10 { try engine.step(camera: camera(c), target: target, background: background) }
        let warmBytes = engine.allocatedGPUBytes
        for _ in 0..<100 {
            try engine.step(camera: camera(c), target: target, background: background)
            _ = try engine.render(camera: camera(c), background: background)
        }
        try expectLessThan(engine.allocatedGPUBytes - warmBytes, 4 * 1024 * 1024,
                           "Repeated steps/renders must release transient buffers after warmup")
    }

    func testOptimizerReducesLossAndCheckpointPreservesNextUpdate() throws {
        try requireMetal()
        let c = ReferenceCamera()
        let truth = IndependentReference.fixture()
        let target = IndependentReference.render(parameters: truth, camera: c).map { SIMD4<Float>($0) }
        var initial = truth
        initial[9] = 0.16; initial[10] = 0.55; initial[11] = 0.38
        var rates = Array(repeating: Float(0.0001), count: 20)
        for index in 9...11 { rates[index] = 0.025 }
        let configuration = TrainingConfiguration(learningRates: rates)
        let engine = try MetalHexelTrainer(parameters: initial.map(Float.init), configuration: configuration)
        let before = try engine.lossAndGradients(camera: camera(c), target: target, background: background).loss
        for _ in 0..<30 { try engine.step(camera: camera(c), target: target, background: background) }
        let after = try engine.lossAndGradients(camera: camera(c), target: target, background: background).loss
        try expectLessThan(after, before * 0.15)
        try expectEqual(engine.stepCount, 30)

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let checkpoint = directory.appendingPathComponent("checkpoint.json")
        engine.checkpointContext = "independent-reference-fixture-v1"
        try engine.saveCheckpoint(to: checkpoint)
        let restored = try MetalHexelTrainer.loadCheckpoint(from: checkpoint, configuration: configuration)
        try expectEqual(restored.parameters, engine.parameters)
        try expectEqual(restored.stepCount, engine.stepCount)
        try expectEqual(restored.checkpointContext, engine.checkpointContext)
        let savedRender = try engine.render(camera: camera(c), background: background)
        let loadedRender = try restored.render(camera: camera(c), background: background)
        try expectEqual(savedRender, loadedRender)
        try engine.step(camera: camera(c), target: target, background: background)
        try restored.step(camera: camera(c), target: target, background: background)
        for (a, b) in zip(engine.parameters, restored.parameters) {
            try expectEqual(a, b, accuracy: 1e-7, "Adam moments/step must survive save and reload")
        }
    }

    func testInvalidParametersAndCameraAreRejected() throws {
        try requireMetal()
        try expectThrows(try MetalHexelTrainer(parameters: [0]))
        var p = IndependentReference.fixture().map(Float.init)
        p[0] = .nan
        try expectThrows(try MetalHexelTrainer(parameters: p))
        p = IndependentReference.fixture().map(Float.init)
        for index in 5...8 { p[index] = 0 }
        try expectThrows(try MetalHexelTrainer(parameters: p))
        let c = ReferenceCamera()
        let engine = try MetalHexelTrainer(parameters: IndependentReference.fixture().map(Float.init))
        let invalid = HexelCamera(width: 0, height: c.height, intrinsics: c.intrinsics,
                                  cameraToWorld: c.cameraToWorld)
        try expectThrows(try engine.render(camera: invalid, background: background))
        try expectThrows(try engine.lossAndGradients(camera: camera(c), target: [], background: background))
    }

    func testResourceBudgetsRejectBeforeUnboundedWorkspaceAllocation() throws {
        try requireMetal()
        let p = IndependentReference.fixture().map(Float.init)
        try expectThrows(try MetalHexelTrainer(
            parameters: p, configuration: TrainingConfiguration(maxWorkingSetBytes: 1)))
        try expectThrows(try MetalHexelTrainer(
            parameters: p, configuration: TrainingConfiguration(maxIntersections: 0)))
        let c = ReferenceCamera()
        let engine = try MetalHexelTrainer(parameters: p + p,
                                          configuration: TrainingConfiguration(maxIntersections: 1))
        try expectThrows(try engine.render(camera: camera(c), background: background))
    }
}
