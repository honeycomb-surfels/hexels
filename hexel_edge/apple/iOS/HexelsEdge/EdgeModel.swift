import Foundation
import Combine
import SwiftUI
import UIKit

@MainActor
final class EdgeModel: ObservableObject {
    @Published private(set) var image: UIImage?
    @Published private(set) var isBusy = false
    @Published private(set) var isTraining = false
    @Published private(set) var status = "Capture a scene or open a prepared local folder to begin."
    @Published private(set) var frameName = ""
    @Published private(set) var step = 0
    @Published private(set) var loss: Float?
    @Published private(set) var checkpoint: URL?
    @Published var error: String?
    @Published var settings = SceneSettings() {
        didSet {
            guard !isBusy, image != nil,
                  settings.preset != oldValue.preset || settings.additionalSteps != oldValue.additionalSteps else { return }
            let requested = settings
            Task {
                do { try await worker.updateRuntimeSettings(requested) }
                catch { self.error = error.localizedDescription }
            }
        }
    }
    @Published private(set) var capabilities = PlatformDeviceState.capabilities()
    @Published private(set) var telemetry: DeviceTelemetry?
    @Published private(set) var heldOutPSNR: Double?
    @Published private(set) var reportURL: URL?
    @Published private(set) var capturedSurfaceURL: URL?
    @Published private(set) var hexelSurfaceURL: URL?
    @Published private(set) var benchmarkStatus = "Ready"
    @Published private(set) var benchmarkProgress = 0.0
    @Published private(set) var isBenchmarking = false
    @Published private(set) var isEvaluating = false
    @Published private(set) var availableTrainingViewCount = 0
    @Published private(set) var selectedTrainingViewCount = 0
    @Published private(set) var sourceImageLongEdge = 0
    @Published private(set) var effectiveImageLongEdge = 0
    @Published private(set) var primitiveCount = 0
    @Published private(set) var trainingGoalStep: Int?
    @Published private(set) var trainingProgress = 0.0
    @Published private(set) var estimatedRemainingSeconds: Double?
    @Published private(set) var isShowingReference = false
    @Published private(set) var isNovelView = false
    private let worker = TrainingWorker()
    private var task: Task<Void, Never>?
    private var didHandleLaunch = false
    private var runStartStep = 0
    private var backgroundPause = false
    private var previewRevision = -1
    private var memoryWarning: AnyCancellable?
    private var memoryPause = false

    init() {
        memoryWarning = NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)
            .sink { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self, self.isBusy else { return }
                    self.memoryPause = true
                    self.task?.cancel()
                }
            }
    }

    func restore() {
        guard !isBusy, !didHandleLaunch else { return }
        didHandleLaunch = true
        let arguments = ProcessInfo.processInfo.arguments
        if let folder = argument("--benchmark-folder", in: arguments) {
            launchBenchmark(folder: folder, arguments: arguments); return
        }
        isBusy = true
        task = Task {
            defer { isBusy = false }
            do { if let snapshot = try await worker.restoreLastScene() { try apply(snapshot, resetSettings: true) } }
            catch { self.error = error.localizedDescription }
        }
    }

    func selectPreset(_ preset: QualityPreset) {
        var proposed = DevicePolicy.settings(for: preset, capabilities: capabilities)
        if sourceImageLongEdge > 0 { proposed.imageLongEdge = min(proposed.imageLongEdge, sourceImageLongEdge) }
        if availableTrainingViewCount > 0 { proposed.trainingViewLimit = min(proposed.trainingViewLimit, availableTrainingViewCount) }
        settings = proposed
    }

    func importScene(_ url: URL) {
        guard !isBusy else { return }
        isBusy = true; status = "Copying and validating the local scene…"
        let requested = settings
        task = Task {
            defer { isBusy = false }
            do { try apply(try await worker.importScene(from: url, settings: requested), resetSettings: true) }
            catch { self.error = error.localizedDescription; status = "Scene import failed." }
        }
    }

    func applySettings() {
        guard !isBusy, image != nil else { return }
        isBusy = true
        let requested = settings
        task = Task {
            defer { isBusy = false }
            do {
                try apply(try await worker.applySettings(requested), resetSettings: true)
                status = "New fit from prepared seeds. The previous checkpoint is retained."
            } catch { self.error = error.localizedDescription }
        }
    }

    func train(additionalSteps: Int) {
        run(additionalSteps: additionalSteps, benchmark: false, warmupDelay: 0,
            requireUnplugged: false, minimumSeconds: 0, maximumSeconds: 1800)
    }

    func startBenchmark(additionalSteps: Int = 30_000, warmupDelay: Int = 30, requireUnplugged: Bool = true) {
        run(additionalSteps: additionalSteps, benchmark: true, warmupDelay: warmupDelay,
            requireUnplugged: requireUnplugged, minimumSeconds: 1200, maximumSeconds: 1800)
    }

    private func run(additionalSteps: Int, benchmark: Bool, warmupDelay: Int,
                     requireUnplugged: Bool, minimumSeconds: Double, maximumSeconds: Double) {
        guard !isBusy, image != nil else { return }
        isBusy = true; isTraining = true; isBenchmarking = benchmark; backgroundPause = false; memoryPause = false
        runStartStep = step
        let requestedSteps = max(1, min(additionalSteps, 100_000))
        settings.additionalSteps = requestedSteps
        let requestedSettings = settings
        var requestedGoal = step + requestedSteps
        trainingGoalStep = requestedGoal; trainingProgress = 0; benchmarkProgress = 0
        task = Task {
            var began = false
            let previousIdleTimerSetting = UIApplication.shared.isIdleTimerDisabled
            UIApplication.shared.isIdleTimerDisabled = true
            defer {
                UIApplication.shared.isIdleTimerDisabled = previousIdleTimerSetting
                isBusy = false; isTraining = false; isBenchmarking = false; trainingGoalStep = nil
            }
            do {
                try await worker.updateRuntimeSettings(requestedSettings)
                let canEvaluate = await worker.hasHeldOutViews()
                if benchmark && !canEvaluate { throw TrainingPause(reason: "A benchmark needs a prepared scene with fixed held-out views.") }
                let warmup = benchmark ? min(20, requestedSteps) : 0
                if warmup > 0 {
                    benchmarkStatus = "Warming up: excluded from battery measurement"
                    try apply(try await worker.trainBatch(steps: warmup))
                }
                // Warmup updates remain real saved work, but are outside the measured step budget.
                runStartStep = step; requestedGoal = step + requestedSteps; trainingGoalStep = requestedGoal
                if benchmark {
                    for seconds in stride(from: max(0,warmupDelay), through: 1, by: -1) {
                        try Task.checkCancellation()
                        benchmarkStatus = "Unplug the cable · starts in \(seconds)s"
                        telemetry = await worker.sampleTelemetry()
                        try await Task.sleep(for: .seconds(1))
                    }
                }
                try await worker.beginRun(targetStep: requestedGoal, warmupSteps: warmup, requireUnplugged: requireUnplugged)
                began = true; benchmarkStatus = benchmark ? "Measuring on this device" : "Training"
                if canEvaluate { heldOutPSNR = try await worker.evaluateQuality().psnr_db }
                let started = Date()
                var currentStep = step
                var nextEvaluation = currentStep + 1000
                var lastUIUpdate = Date.distantPast
                while true {
                    try Task.checkCancellation()
                    let elapsed = Date().timeIntervalSince(started)
                    if (currentStep >= requestedGoal && elapsed >= minimumSeconds) || elapsed >= maximumSeconds || currentStep - runStartStep >= 100_000 { break }
                    let remaining = currentStep < requestedGoal ? requestedGoal - currentStep : 10
                    let snapshot = try await worker.trainBatch(steps: min(10, remaining, 100_000 - (currentStep - runStartStep)))
                    currentStep = snapshot.step
                    if Date().timeIntervalSince(lastUIUpdate) >= 0.5 || currentStep == requestedGoal {
                        try apply(snapshot); lastUIUpdate = Date()
                        trainingProgress = min(1, Double(currentStep - runStartStep) / Double(requestedSteps))
                        let timeProgress = minimumSeconds > 0 ? min(1, elapsed / minimumSeconds) : 1
                        benchmarkProgress = benchmark ? min(trainingProgress,timeProgress) : trainingProgress
                        if let milliseconds = telemetry?.stepMilliseconds {
                            estimatedRemainingSeconds = max(Double(max(0,requestedGoal - currentStep)) * milliseconds / 1000,
                                                            max(0,minimumSeconds - elapsed))
                        }
                    }
                    if canEvaluate && currentStep >= nextEvaluation {
                        isEvaluating = true
                        heldOutPSNR = try await worker.evaluateQuality().psnr_db
                        isEvaluating = false; nextEvaluation = currentStep + 1000
                    }
                }
                if canEvaluate {
                    isEvaluating = true
                    heldOutPSNR = try await worker.evaluateQuality().psnr_db
                    isEvaluating = false
                }
                let complete = currentStep >= requestedGoal && Date().timeIntervalSince(started) >= minimumSeconds
                try await worker.finishRun(status: complete ? "complete" : "bounded_limit")
                try apply(try await worker.snapshot())
                benchmarkStatus = complete ? "Complete · report saved" : "Stopped at time/step limit · report saved"
                status = "Saved locally at step \(step)."
                if complete { trainingProgress = 1; benchmarkProgress = 1 }
            } catch {
                isEvaluating = false
                do {
                    if began { try await worker.finishRun(status: backgroundPause ? "backgrounded" : (memoryPause ? "memory_pressure" : "paused"), reason: error.localizedDescription) }
                    checkpoint = try await worker.saveCheckpoint()
                    step = try await worker.currentStepCount()
                    reportURL = await worker.currentReportURL()
                    if !Task.isCancelled && UIApplication.shared.applicationState == .active { try apply(try await worker.snapshot()) }
                    if error is CancellationError {
                        status = memoryPause ? "Paused after an OS memory warning. Saved at step \(step)." : "Paused and saved at step \(step)."
                        benchmarkStatus = status
                    } else if let pause = error as? TrainingPause {
                        status = pause.reason; benchmarkStatus = pause.reason
                    } else { self.error = error.localizedDescription; benchmarkStatus = "Stopped · checkpoint saved" }
                } catch { self.error = error.localizedDescription }
            }
        }
    }

    func stop() {
        backgroundPause = UIApplication.shared.applicationState != .active
        task?.cancel()
    }

    func save() {
        guard !isBusy, image != nil else { return }
        isBusy = true
        task = Task {
            defer { isBusy = false }
            do { checkpoint = try await worker.saveCheckpoint(); status = "Checkpoint saved locally." }
            catch { self.error = error.localizedDescription }
        }
    }

    func exportHexelSurface() {
        guard !isBusy, image != nil else { return }
        isBusy = true
        task = Task {
            defer { isBusy = false }
            do {
                let surface = try await worker.exportHexelSurface()
                try Task.checkCancellation()
                hexelSurfaceURL = surface
                status = "Surface patches saved locally. They are separate hexagons, not a fused or watertight mesh."
            } catch is CancellationError { status = "Surface export paused." }
            catch { self.error = error.localizedDescription }
        }
    }

    func novelView() { performPreview { try await self.worker.novelView() } }
    func nextView() { performPreview { try await self.worker.nextView() } }
    func showReference() { performPreview { try await self.worker.setReference(true) } }
    func showReconstruction() { performPreview { try await self.worker.setReference(false) } }
    func refreshPreview() { performPreview { try await self.worker.snapshot() } }

    private func performPreview(_ operation: @escaping @MainActor () async throws -> TrainingWorker.Snapshot) {
        guard !isBusy, image != nil else { return }
        isBusy = true
        task = Task {
            defer { isBusy = false }
            do { try apply(try await operation()) }
            catch is CancellationError { status = "Preview paused." }
            catch { self.error = error.localizedDescription }
        }
    }

    func evaluateQuality() {
        guard !isBusy, image != nil else { return }
        isBusy = true; isEvaluating = true; memoryPause = false; backgroundPause = false
        task = Task {
            defer { isBusy = false; isEvaluating = false }
            var began = false
            do {
                try await worker.beginRun(targetStep: step, warmupSteps: 0, requireUnplugged: false)
                began = true
                heldOutPSNR = try await worker.evaluateQuality().psnr_db
                try await worker.finishRun(status: "evaluation")
                try apply(try await worker.snapshot())
            } catch {
                if began { try? await worker.finishRun(status: backgroundPause ? "backgrounded" : (memoryPause ? "memory_pressure" : "paused"), reason: error.localizedDescription) }
                reportURL = await worker.currentReportURL()
                if error is CancellationError { status = "Evaluation paused; local checkpoint saved." }
                else { self.error = error.localizedDescription }
            }
        }
    }

    private func launchBenchmark(folder: String, arguments: [String]) {
        var relative = folder
        if relative.hasPrefix("Documents/") { relative = String(relative.dropFirst(10)) }
        let root = URL.documentsDirectory.standardizedFileURL
        let url = root.appendingPathComponent(relative, isDirectory: true).standardizedFileURL
        guard !relative.hasPrefix("/"), url.path.hasPrefix(root.path + "/") else { error = "Benchmark folder must be inside Documents"; return }
        let preset = argument("--benchmark-preset", in: arguments).flatMap(QualityPreset.init(rawValue:)) ?? .high
        selectPreset(preset)
        if let edge = argument("--benchmark-image-long-edge", in: arguments).flatMap(Int.init) { settings.imageLongEdge = edge }
        if let views = argument("--benchmark-views", in: arguments).flatMap(Int.init) { settings.trainingViewLimit = views }
        let steps = argument("--benchmark-steps", in: arguments).flatMap(Int.init) ?? 30_000
        let delay = argument("--benchmark-warmup-delay", in: arguments).flatMap(Int.init) ?? 30
        let minimum = argument("--benchmark-min-seconds", in: arguments).flatMap(Double.init) ?? 1200
        let maximum = argument("--benchmark-max-seconds", in: arguments).flatMap(Double.init) ?? 1800
        isBusy = true
        let requested = settings
        task = Task {
            do {
                try apply(try await worker.importScene(from: url, settings: requested), resetSettings: true)
                isBusy = false
                run(additionalSteps: steps, benchmark: true, warmupDelay: max(0,min(delay,300)),
                    requireUnplugged: !arguments.contains("--benchmark-allow-charging"),
                    minimumSeconds: max(0,min(minimum,1800)), maximumSeconds: max(1,min(maximum,3600)))
            } catch { isBusy = false; self.error = error.localizedDescription }
        }
    }

    private func argument(_ name: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: name), index + 1 < arguments.count else { return nil }
        return arguments[index + 1]
    }

    private func apply(_ snapshot: TrainingWorker.Snapshot, resetSettings: Bool = false) throws {
        if snapshot.previewRevision != previewRevision {
        var bytes = [UInt8](); bytes.reserveCapacity(snapshot.pixels.count * 4)
        for pixel in snapshot.pixels {
            for channel in 0..<3 {
                let value = pixel[channel].isFinite ? min(1,max(0,pixel[channel])) : 0
                bytes.append(UInt8((value * 255).rounded()))
            }
            bytes.append(255)
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData), let space = CGColorSpace(name: CGColorSpace.sRGB),
              let cgImage = CGImage(width: snapshot.width, height: snapshot.height, bitsPerComponent: 8,
                bitsPerPixel: 32, bytesPerRow: snapshot.width * 4, space: space,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider,
                decode: nil, shouldInterpolate: true, intent: .defaultIntent) else { throw CocoaError(.fileReadCorruptFile) }
        image = UIImage(cgImage: cgImage); previewRevision = snapshot.previewRevision
        }
        frameName = snapshot.frameName; step = snapshot.step
        loss = snapshot.loss; checkpoint = snapshot.checkpoint
        if resetSettings { settings = snapshot.settings }
        capabilities = snapshot.capabilities; telemetry = snapshot.telemetry
        heldOutPSNR = snapshot.heldOutPSNR; reportURL = snapshot.reportURL
        capturedSurfaceURL = snapshot.capturedSurfaceURL; hexelSurfaceURL = snapshot.hexelSurfaceURL
        isNovelView = snapshot.isNovelView
        availableTrainingViewCount = snapshot.availableTrainingViewCount
        selectedTrainingViewCount = snapshot.selectedTrainingViewCount
        sourceImageLongEdge = snapshot.sourceImageLongEdge; effectiveImageLongEdge = snapshot.effectiveImageLongEdge
        primitiveCount = snapshot.primitiveCount; isShowingReference = snapshot.showingReference
        status = "\(snapshot.primitiveCount) Hexels · \(snapshot.selectedTrainingViewCount) training images · step \(snapshot.step)"
    }
}
