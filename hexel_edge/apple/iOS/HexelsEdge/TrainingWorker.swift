import CryptoKit
import Foundation
import HexelsMetal

struct TrainingPause: Error, LocalizedError {
    let reason: String
    var errorDescription: String? { reason }
}

private struct LocalSession: Codable {
    var settings: SceneSettings
    var directory: String
}

/// One serialized trainer, bounded image cache, and local benchmark recorder.
actor TrainingWorker {
    struct Snapshot: Sendable {
        let frameName: String
        let frameCount: Int
        let primitiveCount: Int
        let width: Int
        let height: Int
        let pixels: [SIMD4<Float>]
        let step: Int
        let loss: Float?
        let checkpoint: URL?
        let settings: SceneSettings
        let capabilities: DeviceCapabilities
        let availableTrainingViewCount: Int
        let selectedTrainingViewCount: Int
        let sourceImageLongEdge: Int
        let effectiveImageLongEdge: Int
        let telemetry: DeviceTelemetry?
        let heldOutPSNR: Double?
        let reportURL: URL?
        let capturedSurfaceURL: URL?
        let hexelSurfaceURL: URL?
        let showingReference: Bool
        let isNovelView: Bool
        let previewRevision: Int
    }

    private var bundle: SceneBundle?
    private var trainer: MetalHexelTrainer?
    private var sceneDirectory: URL?
    private var sceneRoot: URL?
    private var capturedSurface: URL?
    private var previewIndex = 0
    private var novelCamera: HexelCamera?
    private var lastLoss: Float?
    private var decodedImages: [String: [SIMD4<Float>]] = [:]
    private var cacheOrder: [String] = []
    private var settings = SceneSettings()
    private var capabilities: DeviceCapabilities?
    private var selectedTraining: [Int] = []
    private var viewOrder: [Int] = []
    private var edge = 640
    private var showingReference = false
    private var telemetry: DeviceTelemetry?
    private var heldOutPSNR: Double?
    private var report: BenchmarkReport?
    private var reportDirectory: URL?
    private var reportActive = false
    private var requireUnplugged = false
    private var lastReportWrite = Date.distantPast
    private var lastMemorySample = Date.distantPast
    private var lastPreviewUpdate = Date.distantPast
    private var previewPixels: [SIMD4<Float>] = []
    private var previewRevision = 0
    private var measuredSteps = 0
    private var measuredSeconds = 0.0

    private var storage: URL { URL.applicationSupportDirectory.appendingPathComponent("HexelsEdge", isDirectory: true) }

    func restoreLastScene() async throws -> Snapshot? {
        try prepareSharedDocuments()
        guard let path = UserDefaults.standard.string(forKey: "lastLocalScene") else { return nil }
        let root = storage.appendingPathComponent(path, isDirectory: true)
        guard FileManager.default.fileExists(atPath: root.appendingPathComponent("manifest.json").path) else { return nil }
        let metadataURL = root.appendingPathComponent("active_session.json")
        if FileManager.default.fileExists(atPath: metadataURL.path) {
            let metadata = try JSONDecoder().decode(LocalSession.self, from: Data(contentsOf: metadataURL))
            guard metadata.directory == "." || (metadata.directory.hasPrefix(".sessions/") && !metadata.directory.contains("..")) else {
                throw HexelsError.invalid("Saved session path is invalid")
            }
            return try await open(root: root, requested: metadata.settings,
                                  session: root.appendingPathComponent(metadata.directory, isDirectory: true))
        }
        // The previous app always used every training view at original resolution.
        let legacy = try SceneBundle(url: root)
        let legacySettings = SceneSettings(preset: .high,
            imageLongEdge: legacy.frames.map { max($0.width, $0.height) }.max() ?? 640,
            trainingViewLimit: legacy.frames.filter { $0.split == "train" }.count, additionalSteps: 10_000)
        return try await open(root: root, requested: legacySettings, session: root)
    }

    func importScene(from source: URL, settings: SceneSettings = SceneSettings()) async throws -> Snapshot {
        try prepareSharedDocuments()
        let accessible = source.startAccessingSecurityScopedResource()
        defer { if accessible { source.stopAccessingSecurityScopedResource() } }
        try validateImportSize(source)
        var directory = storage.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: storage, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: source, to: directory)
        do {
            var values = URLResourceValues(); values.isExcludedFromBackup = true
            try directory.setResourceValues(values)
            try JSONEncoder().encode(LocalSession(settings: settings, directory: "."))
                .write(to: directory.appendingPathComponent("active_session.json"), options: .atomic)
            let result = try await open(root: directory, requested: settings, session: directory)
            UserDefaults.standard.set(directory.lastPathComponent, forKey: "lastLocalScene")
            return result
        } catch { try? FileManager.default.removeItem(at: directory); throw error }
    }

    func applySettings(_ requested: SceneSettings) async throws -> Snapshot {
        guard let root = sceneRoot else { throw HexelsError.invalid("Import a scene first") }
        _ = try saveCheckpoint()
        let relative = ".sessions/\(UUID().uuidString)"
        let directory = root.appendingPathComponent(relative, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // Stage the durable selection before committing the live worker. Roll it back on a failed admission.
        let metadataURL = root.appendingPathComponent("active_session.json")
        let previousMetadata = try? Data(contentsOf: metadataURL)
        do {
            try JSONEncoder().encode(LocalSession(settings: requested, directory: relative)).write(to: metadataURL, options: .atomic)
            return try await open(root: root, requested: requested, session: directory)
        } catch {
            if let previousMetadata { try? previousMetadata.write(to: metadataURL, options: .atomic) }
            else { try? FileManager.default.removeItem(at: metadataURL) }
            try? FileManager.default.removeItem(at: directory); throw error
        }
    }

    /// Duration/preset metadata does not change the fitted views or calibrated pixels.
    func updateRuntimeSettings(_ requested: SceneSettings) throws {
        guard let root = sceneRoot, let directory = sceneDirectory else { return }
        var effective = settings
        effective.preset = requested.preset
        effective.additionalSteps = max(1, min(requested.additionalSteps, 100_000))
        let relative = directory == root ? "." : ".sessions/\(directory.lastPathComponent)"
        try JSONEncoder().encode(LocalSession(settings: effective, directory: relative))
            .write(to: root.appendingPathComponent("active_session.json"), options: .atomic)
        settings = effective
    }

    private func open(root: URL, requested: SceneSettings, session: URL) async throws -> Snapshot {
        let loaded = try SceneBundle(url: root)
        let loadedSurface = try Self.capturedSurfaceFile(in: root)
        let caps = await PlatformDeviceState.capabilities()
        guard caps.supportsMetal, caps.maxWorkspaceMiB >= 32,
              loaded.initialParameters.count / 20 <= caps.maxPrimitiveCount,
              loaded.frames.count <= 256, loaded.frames.allSatisfy({ max($0.width, $0.height) <= 2048 }) else {
            throw HexelsError.resourceLimit("Prepared scene exceeds this device's conservative limits; use fewer seeds or smaller source images.")
        }
        let training = loaded.frames.indices.filter { loaded.frames[$0].split == "train" }
        let sourceEdge = loaded.frames.map { max($0.width, $0.height) }.max() ?? 0
        var effective = requested
        effective.imageLongEdge = min(sourceEdge, max(32, min(requested.imageLongEdge, caps.maxImageLongEdge)))
        effective.trainingViewLimit = min(training.count, max(1, min(requested.trainingViewLimit, caps.maxTrainingViews)))
        effective.additionalSteps = max(1, min(requested.additionalSteps, 100_000))
        let selected = DevicePolicy.selectTrainingIndices(training, limit: effective.trainingViewLimit)
        guard !selected.isEmpty else { throw HexelsError.invalid("Scene has no training views") }
        let selectedSet = Set(selected)
        var order = loaded.viewOrder.map(Int.init).filter { selectedSet.contains($0) }
        if Set(order) != selectedSet { order = selected }
        let dimensions = loaded.frames.map { frame -> [Int] in
            let d = DevicePolicy.resizedDimensions(width: frame.width, height: frame.height, longEdge: effective.imageLongEdge)
            return [d.width, d.height]
        }
        let selection = try JSONSerialization.data(withJSONObject: ["training": selected, "dimensions": dimensions], options: [.sortedKeys])
        let digest = SHA256.hash(data: selection).map { String(format: "%02x", $0) }.joined()
        let context = "edge2:\(loaded.fingerprint):\(digest)"
        let configuration = TrainingConfiguration(learningRates: loaded.recipeHint == "detail" ? TrainingConfiguration.detailRates : TrainingConfiguration.defaultRates,
            maxIntersections: 4_000_000,
            maxWorkingSetBytes: caps.maxWorkspaceMiB * 1_048_576)
        let checkpoint = session.appendingPathComponent("checkpoint.hexstate")
        let engine: MetalHexelTrainer
        if FileManager.default.fileExists(atPath: checkpoint.path) {
            engine = try MetalHexelTrainer.loadCheckpoint(from: checkpoint, configuration: configuration)
            let unchanged = selected == training && zip(loaded.frames, dimensions).allSatisfy { $0.0.width == $0.1[0] && $0.0.height == $0.1[1] }
            guard engine.parameters.count == loaded.initialParameters.count,
                  engine.learningRates == configuration.learningRates,
                  engine.checkpointContext == context || (unchanged && engine.checkpointContext == loaded.fingerprint) else {
                throw HexelsError.invalid("Checkpoint settings differ from the selected views/resolution. Restore the original settings or import a fresh bundle.")
            }
        } else { engine = try MetalHexelTrainer(parameters: loaded.initialParameters, configuration: configuration) }
        engine.checkpointContext = context
        try Task.checkCancellation()
        let first = loaded.frames.firstIndex(where: { $0.split == "test" }) ?? selected[0]
        let camera = Self.camera(loaded.frames[first], longEdge: effective.imageLongEdge)
        let firstPixels = try engine.render(camera: camera, background: loaded.background)
        try Task.checkCancellation()
        bundle = loaded; trainer = engine; sceneRoot = root; sceneDirectory = session
        capturedSurface = loadedSurface
        settings = effective; capabilities = caps; selectedTraining = selected; viewOrder = order
        edge = effective.imageLongEdge; previewIndex = first; lastLoss = nil; heldOutPSNR = nil
        decodedImages = [:]; cacheOrder = []; showingReference = false; novelCamera = nil
        report = nil; reportDirectory = nil; reportActive = false; measuredSteps = 0; measuredSeconds = 0
        previewPixels = firstPixels; lastPreviewUpdate = Date(); previewRevision += 1
        return try await snapshot(at: previewIndex, reference: false, refresh: false)
    }

    static func camera(_ frame: SceneFrame, longEdge: Int) -> HexelCamera {
        let d = DevicePolicy.resizedDimensions(width: frame.width, height: frame.height, longEdge: longEdge)
        return HexelCamera(width: d.width, height: d.height,
            intrinsics: DevicePolicy.resizedIntrinsics(frame.intrinsics, oldWidth: frame.width, oldHeight: frame.height,
                newWidth: d.width, newHeight: d.height), cameraToWorld: frame.cameraToWorld)
    }

    private func image(for frame: SceneFrame, bundle: SceneBundle, longEdge: Int? = nil) throws -> [SIMD4<Float>] {
        let resizeEdge = longEdge ?? edge
        let key = "\(frame.name):\(resizeEdge)"
        if let cached = decodedImages[key] { return cached }
        let original = try bundle.image(for: frame, maxDecodedBytes: max(16, capabilities?.maxWorkspaceMiB ?? 64) * 1_048_576 / 3)
        let d = DevicePolicy.resizedDimensions(width: frame.width, height: frame.height, longEdge: resizeEdge)
        let pixels: [SIMD4<Float>]
        if d.width == frame.width && d.height == frame.height { pixels = original }
        else {
            pixels = (0..<(d.width * d.height)).map { index in
                let x = (Double(index % d.width) + 0.5) * Double(frame.width) / Double(d.width) - 0.5
                let y = (Double(index / d.width) + 0.5) * Double(frame.height) / Double(d.height) - 0.5
                let x0 = max(0, min(frame.width - 1, Int(floor(x)))), x1 = min(frame.width - 1, max(0, Int(floor(x)) + 1))
                let y0 = max(0, min(frame.height - 1, Int(floor(y)))), y1 = min(frame.height - 1, max(0, Int(floor(y)) + 1))
                let a = Float(x - floor(x)), b = Float(y - floor(y))
                return (1-b) * ((1-a)*original[y0*frame.width+x0] + a*original[y0*frame.width+x1])
                    + b * ((1-a)*original[y1*frame.width+x0] + a*original[y1*frame.width+x1])
            }
        }
        if cacheOrder.count >= 3 { decodedImages.removeValue(forKey: cacheOrder.removeFirst()) }
        decodedImages[key] = pixels; cacheOrder.append(key)
        return pixels
    }

    func trainBatch(steps: Int) async throws -> Snapshot {
        try Task.checkCancellation()
        guard let bundle, let trainer else { throw HexelsError.invalid("Import a scene first") }
        let before = await sampleTelemetry()
        if before.availableMemoryBytes < 32 * 1_048_576 {
            throw TrainingPause(reason: "Paused for low advisory memory headroom (\(before.availableMemoryBytes / 1_048_576) MiB available). Close other apps before resuming.")
        }
        if let reason = DevicePolicy.pauseReason(thermal: before.thermalState, batteryFraction: before.batteryFraction,
            batteryState: before.batteryState, lowPowerMode: before.lowPowerMode, requireUnplugged: requireUnplugged) {
            throw TrainingPause(reason: reason)
        }
        let start = ProcessInfo.processInfo.systemUptime
        var completed = 0
        do {
            for _ in 0..<steps {
                try Task.checkCancellation()
                let frame = bundle.frames[viewOrder[trainer.stepCount % viewOrder.count]]
                let target = try image(for: frame, bundle: bundle)
                try Task.checkCancellation()
                lastLoss = try trainer.step(camera: Self.camera(frame, longEdge: edge),
                    target: target, background: bundle.background)
                completed += 1
                if trainer.stepCount % 100 == 0 { _ = try saveCheckpoint() }
            }
        } catch {
            accountTraining(seconds: ProcessInfo.processInfo.systemUptime - start, steps: completed)
            _ = await sampleTelemetry(); throw error
        }
        accountTraining(seconds: ProcessInfo.processInfo.systemUptime - start, steps: completed)
        _ = await sampleTelemetry()
        try Task.checkCancellation()
        return try await snapshot(at: previewIndex, reference: showingReference, refresh: Date().timeIntervalSince(lastPreviewUpdate) >= 0.5)
    }

    private func accountTraining(seconds: Double, steps: Int) {
        measuredSeconds += seconds; measuredSteps += steps
        if reportActive { report!.training_seconds += seconds }
    }

    func sampleTelemetry(force: Bool = false) async -> DeviceTelemetry {
        let elapsed = reportActive ? Date().timeIntervalSince(report!.started_at) : measuredSeconds
        let detailed = force || Date().timeIntervalSince(lastMemorySample) >= 5
        let sample = await PlatformDeviceState.capture(gpuBytes: trainer?.allocatedGPUBytes ?? 0,
            stepMilliseconds: measuredSteps >= 20 ? measuredSeconds / Double(measuredSteps) * 1000 : nil,
            elapsedSeconds: elapsed, previous: telemetry, includeMemory: detailed)
        let transition = telemetry?.batteryState != sample.batteryState || telemetry?.thermalState != sample.thermalState || telemetry?.lowPowerMode != sample.lowPowerMode
        if detailed { lastMemorySample = Date() }
        telemetry = sample
        if reportActive, let directory = reportDirectory {
            report!.observe(sample, step: trainer?.stepCount ?? 0)
            if detailed || transition {
                do {
                    try BenchmarkReport.appendCSV(sample, step: trainer?.stepCount ?? 0, in: directory)
                    try report!.persist(in: directory); lastReportWrite = Date()
                } catch { report!.invalidateBattery("Telemetry persistence error: \(error.localizedDescription)") }
            }
        }
        return sample
    }

    func beginRun(targetStep: Int, warmupSteps: Int, requireUnplugged: Bool) async throws {
        guard let bundle, let trainer else { throw HexelsError.invalid("Import a scene first") }
        let sample = await sampleTelemetry(force: true)
        if let reason = DevicePolicy.pauseReason(thermal: sample.thermalState, batteryFraction: sample.batteryFraction,
            batteryState: sample.batteryState, lowPowerMode: sample.lowPowerMode, requireUnplugged: requireUnplugged) {
            throw TrainingPause(reason: reason)
        }
        let id = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-") + "_" + UUID().uuidString.prefix(8)
        // New camera scenes keep evaluation photos and geometry private too.
        let captureScene = sceneRoot.map { FileManager.default.fileExists(atPath: $0.appendingPathComponent("capture_summary.json").path) } ?? false
        let reportRoot = captureScene ? sceneRoot!.appendingPathComponent("reports", isDirectory: true)
                                      : URL.documentsDirectory.appendingPathComponent("Reports", isDirectory: true)
        var directory = reportRoot.appendingPathComponent(id, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var values = URLResourceValues(); values.isExcludedFromBackup = true; try directory.setResourceValues(values)
        self.requireUnplugged = requireUnplugged; reportDirectory = directory
        measuredSteps = 0; measuredSeconds = 0
        report = BenchmarkReport(run_id: id, initial_step: trainer.stepCount, target_step: targetStep,
            last_step: trainer.stepCount, warmup_steps: warmupSteps,
            selected_training_frames: selectedTraining.map { bundle.frames[$0].name },
            held_out_frames: bundle.frames.filter { $0.split == "test" }.map(\.name),
            image_long_edge: edge, evaluation_long_edge: evaluationEdge(bundle), primitive_count: trainer.parameters.count / 20,
            settings: settings, bundle_fingerprint: bundle.fingerprint, checkpoint_context: trainer.checkpointContext ?? "")
        lastReportWrite = .distantPast
        reportActive = true
        _ = await sampleTelemetry(force: true)
        try report!.persist(in: directory)
    }

    func finishRun(status: String, reason: String? = nil) async throws {
        defer { requireUnplugged = false; reportActive = false }
        _ = try saveCheckpoint()
        _ = await sampleTelemetry(force: true)
        if reportActive, let directory = reportDirectory {
            report!.status = status; report!.stop_reason = reason; report!.ended_at = Date()
            if status == "backgrounded" { report!.invalidateBattery("Run interrupted by backgrounding") }
            if status == "memory_pressure" { report!.invalidateBattery("Run interrupted by an OS memory warning") }
            try report!.persist(in: directory)
            try trainer?.saveCheckpoint(to: directory.appendingPathComponent("final.hexstate"))
        }
    }

    func evaluateQuality() async throws -> QualityEvaluation {
        try Task.checkCancellation()
        guard let bundle, let trainer else { throw HexelsError.invalid("Import a scene first") }
        let tests = bundle.frames.filter { $0.split == "test" }
        guard !tests.isEmpty else { throw HexelsError.invalid("This bundle has no held-out views") }
        let fixedEdge = evaluationEdge(bundle)
        let directory = reportActive ? reportDirectory?.appendingPathComponent(String(format: "eval_%06d", trainer.stepCount), isDirectory: true) : nil
        if let directory { try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true) }
        var squared = 0.0, samples = 0, perView: [Double] = []
        for (index, frame) in tests.enumerated() {
            try Task.checkCancellation()
            let camera = Self.camera(frame, longEdge: fixedEdge)
            let target = try image(for: frame, bundle: bundle, longEdge: fixedEdge)
            try Task.checkCancellation()
            let prediction = try trainer.render(camera: camera, background: bundle.background)
            var viewSquared = 0.0
            for (a,b) in zip(prediction, target) {
                let d = a - b; viewSquared += Double(d.x*d.x + d.y*d.y + d.z*d.z)
            }
            squared += viewSquared; samples += target.count * 3
            perView.append(-10 * log10(max(viewSquared / Double(target.count * 3), 1e-20)))
            if let directory {
                try BenchmarkImages.writePNG(prediction, width: camera.width, height: camera.height,
                    to: directory.appendingPathComponent("render_\(index).png"))
                try BenchmarkImages.writePNG(target, width: camera.width, height: camera.height,
                    to: directory.appendingPathComponent("target_\(index).png"))
            }
        }
        let mse = squared / Double(samples)
        let result = QualityEvaluation(step: trainer.stepCount, elapsed_seconds: report.map { Date().timeIntervalSince($0.started_at) } ?? 0,
            pooled_rgb_mse: mse, psnr_db: -10 * log10(max(mse, 1e-20)),
            mean_view_psnr_db: perView.reduce(0,+) / Double(perView.count), per_view_psnr_db: perView, evaluation_long_edge: fixedEdge)
        heldOutPSNR = result.psnr_db
        if reportActive, let directory = reportDirectory {
            report!.evaluations.append(result)
            if report!.best_psnr_db == nil || result.psnr_db > report!.best_psnr_db! {
                report!.best_psnr_db = result.psnr_db; report!.best_step = trainer.stepCount
                try trainer.saveCheckpoint(to: directory.appendingPathComponent("best.hexstate"))
            }
            try report!.persist(in: directory)
        }
        return result
    }

    private func evaluationEdge(_ bundle: SceneBundle) -> Int {
        min(640, bundle.frames.map { max($0.width,$0.height) }.max() ?? 640)
    }

    func nextView() async throws -> Snapshot {
        guard let bundle else { throw HexelsError.invalid("Import a scene first") }
        let candidate = (previewIndex + 1) % bundle.frames.count
        let result = try await snapshot(at: candidate, reference: showingReference, nativeCamera: true)
        previewIndex = candidate; novelCamera = nil; return result
    }
    func setReference(_ reference: Bool) async throws -> Snapshot {
        let result = try await snapshot(at: previewIndex, reference: reference, nativeCamera: true)
        showingReference = reference; novelCamera = nil; return result
    }
    /// Render from a genuinely new camera; optimization and quality still use recorded views.
    func novelView() async throws -> Snapshot {
        try Task.checkCancellation()
        guard let bundle, let trainer, bundle.frames.count > 1 else { throw HexelsError.invalid("Open a scene with at least two cameras first") }
        let next = (previewIndex + 1) % bundle.frames.count
        let camera = try CaptureGeometry.interpolatedCamera(Self.camera(bundle.frames[previewIndex], longEdge: edge),
                                                           Self.camera(bundle.frames[next], longEdge: edge))
        let pixels = try trainer.render(camera: camera, background: bundle.background)
        try Task.checkCancellation()
        guard pixels.count == camera.width * camera.height,
              pixels.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite }) else {
            throw HexelsError.invalid("Synthesized view contains invalid pixels")
        }
        if let sceneDirectory {
            let receipt: [String: Any] = ["schema_version": 1, "optimizer_step": trainer.stepCount,
                "source_camera_index": previewIndex, "next_camera_index": next, "interpolation_fraction": 0.5,
                "width": camera.width, "height": camera.height, "rendered_pixels": pixels.count,
                "all_pixels_finite": true, "has_reference_photo": false,
                "used_for_training_or_quality_metrics": false]
            try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys, .prettyPrinted])
                .write(to: sceneDirectory.appendingPathComponent("novel_view_check.json"), options: .atomic)
        }
        novelCamera = camera; showingReference = false
        previewPixels = pixels; lastPreviewUpdate = Date(); previewRevision += 1
        return try await snapshot(at: previewIndex, reference: false, refresh: false)
    }
    func currentStepCount() throws -> Int {
        guard let trainer else { throw HexelsError.invalid("Import a scene first") }; return trainer.stepCount
    }
    func hasHeldOutViews() -> Bool { bundle?.frames.contains { $0.split == "test" } ?? false }
    func currentReportURL() -> URL? { reportDirectory?.appendingPathComponent("report.json") }

    /// Export learned hard-support surface patches; these do not fuse into a watertight mesh.
    func exportHexelSurface() throws -> URL {
        try Task.checkCancellation()
        guard let trainer, let sceneDirectory, let bundle else { throw HexelsError.invalid("Open or capture a scene first") }
        let destination = sceneDirectory.appendingPathComponent("hexel_surface_step_\(trainer.stepCount).ply")
        if FileManager.default.fileExists(atPath: destination.path) { return destination }
        let parameters = trainer.parameters
        var visible: [Float] = []
        visible.reserveCapacity(parameters.count)
        for offset in stride(from: 0, to: parameters.count, by: 20) {
            let logit = parameters[offset + 18]
            // Cells whose maximum optical opacity is under 1% add little useful surface geometry.
            if logit >= log(Float(0.01) / Float(0.99)) {
                visible.append(contentsOf: parameters[offset..<(offset + 20)])
            }
        }
        guard !visible.isEmpty else { throw HexelsError.invalid("No visible Hexels to export; refine the scene first") }
        let surface = try SurfaceExport.hexelPatches(parameters: visible)
        try Task.checkCancellation()
        let description: [String: Any] = [
            "schema_version": 1, "surface_kind": "independent_hexagonal_patches_not_fused_or_watertight",
            "optimizer_step": trainer.stepCount, "bundle_fingerprint": bundle.fingerprint,
            "checkpoint_context": trainer.checkpointContext ?? "", "exported_cells": visible.count / 20,
            "vertices": surface.vertices.count, "triangles": surface.triangles.count,
            "minimum_cell_opacity": 0.01, "optical_alpha_and_soft_tails_exported": false,
            "color": "encoded sRGB vertex colors from trained spatial RGB; clipped to bytes",
        ]
        let metadata = try JSONSerialization.data(withJSONObject: description, options: [.sortedKeys, .prettyPrinted])
        try SurfaceExport.writePLY(surface, to: destination)
        do { try metadata.write(to: destination.appendingPathExtension("json"), options: .atomic) }
        catch { try? FileManager.default.removeItem(at: destination); throw error }
        return destination
    }

    /// Optional capture provenance must refer to an actual bounded local file inside this scene.
    private static func capturedSurfaceFile(in root: URL) throws -> URL? {
        let metadata = try Data(contentsOf: root.appendingPathComponent("manifest.json"))
        let manifest = try JSONSerialization.jsonObject(with: metadata) as? [String: Any]
        guard let value = manifest?["captured_surface_file"] else { return nil }
        guard let relative = value as? String, !relative.isEmpty, !relative.hasPrefix("/"),
              !relative.split(separator: "/").contains("..") else {
            throw HexelsError.invalid("Captured surface path must stay inside its scene")
        }
        let parent = root.standardizedFileURL.resolvingSymlinksInPath()
        let candidate = parent.appendingPathComponent(relative).standardizedFileURL.resolvingSymlinksInPath()
        guard candidate.path.hasPrefix(parent.path + "/"), ["obj", "ply"].contains(candidate.pathExtension.lowercased()) else {
            throw HexelsError.invalid("Captured surface must be a local OBJ or PLY inside its scene")
        }
        let properties = try candidate.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
        guard properties.isRegularFile == true, properties.isSymbolicLink != true,
              let size = properties.fileSize, size > 0, size <= 128 * 1024 * 1024 else {
            throw HexelsError.resourceLimit("Captured surface is absent or exceeds its 128 MiB file limit")
        }
        return candidate
    }
    @discardableResult func saveCheckpoint() throws -> URL {
        guard let trainer, let sceneDirectory else { throw HexelsError.invalid("Import a scene first") }
        let url = sceneDirectory.appendingPathComponent("checkpoint.hexstate")
        try trainer.saveCheckpoint(to: url); return url
    }
    func snapshot() async throws -> Snapshot { try await snapshot(at: previewIndex, reference: showingReference) }
    private func snapshot(at index: Int, reference: Bool, refresh: Bool = true, nativeCamera: Bool = false) async throws -> Snapshot {
        // The cached initial snapshot follows the transactional commit and cannot cancel it.
        if refresh || previewPixels.isEmpty { try Task.checkCancellation() }
        guard let bundle, let trainer, let sceneDirectory, let capabilities else { throw HexelsError.invalid("Import a scene first") }
        let frame = bundle.frames[index]
        let isNovel = !nativeCamera && !reference && index == previewIndex && novelCamera != nil
        let camera = isNovel ? novelCamera! : Self.camera(frame, longEdge: edge)
        if refresh || previewPixels.isEmpty {
            previewPixels = reference ? try image(for: frame, bundle: bundle) : try trainer.render(camera: camera, background: bundle.background)
            lastPreviewUpdate = Date(); previewRevision += 1
        }
        let checkpoint = sceneDirectory.appendingPathComponent("checkpoint.hexstate")
        let patchSurface = sceneDirectory.appendingPathComponent("hexel_surface_step_\(trainer.stepCount).ply")
        return Snapshot(frameName: isNovel ? "Between views · synthesized viewpoint" : "\(frame.name) (\(frame.split))", frameCount: bundle.frames.count,
            primitiveCount: trainer.parameters.count / 20, width: camera.width, height: camera.height,
            pixels: previewPixels, step: trainer.stepCount, loss: lastLoss,
            checkpoint: FileManager.default.fileExists(atPath: checkpoint.path) ? checkpoint : nil,
            settings: settings, capabilities: capabilities,
            availableTrainingViewCount: bundle.frames.filter { $0.split == "train" }.count,
            selectedTrainingViewCount: selectedTraining.count,
            sourceImageLongEdge: bundle.frames.map { max($0.width,$0.height) }.max() ?? 0,
            effectiveImageLongEdge: edge, telemetry: telemetry, heldOutPSNR: heldOutPSNR,
            reportURL: reportDirectory?.appendingPathComponent("report.json"),
            capturedSurfaceURL: capturedSurface,
            hexelSurfaceURL: FileManager.default.fileExists(atPath: patchSurface.path) ? patchSurface : nil,
            showingReference: reference, isNovelView: isNovel, previewRevision: previewRevision)
    }

    private func prepareSharedDocuments() throws {
        var documents = URL.documentsDirectory
        try FileManager.default.createDirectory(at: documents, withIntermediateDirectories: true)
        var values = URLResourceValues(); values.isExcludedFromBackup = true; try documents.setResourceValues(values)
        let directory = documents.appendingPathComponent("Benchmarks", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }
    private func validateImportSize(_ source: URL) throws {
        let values = try source.resourceValues(forKeys: [.isDirectoryKey,.isSymbolicLinkKey])
        guard values.isDirectory == true, values.isSymbolicLink != true else { throw HexelsError.invalid("Select a prepared scene folder") }
        let keys: Set<URLResourceKey> = [.fileSizeKey,.isSymbolicLinkKey]
        guard let files = FileManager.default.enumerator(at: source, includingPropertiesForKeys: Array(keys)) else { throw HexelsError.invalid("Cannot read scene folder") }
        var bytes = 0, count = 0
        for case let file as URL in files {
            let values = try file.resourceValues(forKeys: keys)
            guard values.isSymbolicLink != true else { throw HexelsError.invalid("Scene imports cannot contain symbolic links") }
            bytes += values.fileSize ?? 0; count += 1
            guard bytes <= 512*1024*1024, count <= 2_000 else { throw HexelsError.resourceLimit("Import exceeds 512 MiB / 2,000-file limit") }
        }
    }
}
