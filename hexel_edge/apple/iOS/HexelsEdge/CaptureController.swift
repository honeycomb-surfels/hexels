@preconcurrency import ARKit
import AVFoundation
import Combine
import Foundation
import HexelsMetal
import UIKit
import simd

@MainActor
final class CaptureController: ObservableObject {
    let session = ARSession()
    @Published private(set) var isCapturing = false
    @Published private(set) var isProcessing = false
    @Published private(set) var status = "Scan a small, static, well-lit scene."
    @Published private(set) var tracking = "Not running"
    @Published private(set) var permissionStatus = "Not requested"
    @Published private(set) var guidance = "Move sideways and around the scene; avoid people, reflections and pure rotation."
    @Published private(set) var frameCount = 0
    @Published private(set) var pointCount = 0
    @Published private(set) var supportsDepth = false
    @Published private(set) var preparedBundleURL: URL?
    @Published private(set) var capturedMeshURL: URL?
    @Published private(set) var error: String?
    @Published private(set) var settings = CaptureSettings()
    @Published private(set) var translationSpan: Float = 0
    var maxFrames: Int { settings.maxFrames }
    var acceptedFrameCount: Int { frameCount }
    var depthAvailable: Bool { supportsDepth }
    var isRecording: Bool { isCapturing }
    var isFinishing: Bool { isProcessing }
    var canFinish: Bool { isCapturing && !isProcessing && frameCount >= 12 && pointCount >= 64 && translationSpan >= 0.12 }
    private var bridge: CaptureSessionBridge?
    private var generation = UUID()
    private var observers: [AnyCancellable] = []
    private var previousIdleTimer: Bool?
    private var isStarting = false

    init() {
        observers.append(NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification).sink { [weak self] _ in
            Task { @MainActor [weak self] in self?.interrupt("Capture cancelled when the app entered the background. Start a new scan.") }
        })
        observers.append(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification).sink { [weak self] _ in
            Task { @MainActor [weak self] in self?.interrupt("Capture cancelled after an OS memory warning. Use fewer views or 640 pixels.") }
        })
    }

    func start(settings requested: CaptureSettings = CaptureSettings()) async {
        guard !isCapturing, !isProcessing, !isStarting else { return }
        let token = UUID(); generation = token; error = nil
        isStarting = true
        defer { if generation == token { isStarting = false } }
        do {
            settings = try requested.validated()
            guard ARWorldTrackingConfiguration.isSupported else { throw HexelsError.unavailable("AR world tracking is unavailable on this device") }
            let authorization = AVCaptureDevice.authorizationStatus(for: .video)
            let allowed: Bool
            if authorization == .notDetermined { allowed = await AVCaptureDevice.requestAccess(for:.video) }
            else { allowed = authorization == .authorized }
            guard generation == token else { return }
            permissionStatus = allowed ? "Granted" : "Denied"
            guard allowed else { throw HexelsError.unavailable("Camera access is required for guided capture. Enable it in Settings to scan.") }
            guard UIApplication.shared.applicationState == .active else { throw HexelsError.unavailable("Return to the app before starting camera capture") }
            guard PlatformDeviceState.capabilities().availableMemoryMiB >= 256 else { throw HexelsError.resourceLimit("Camera capture needs more advisory memory headroom. Close other apps before scanning.") }
            let config = ARWorldTrackingConfiguration()
            config.worldAlignment = .gravity
            config.isAutoFocusEnabled = true
            config.videoHDRAllowed = false
            supportsDepth = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
            if supportsDepth { config.frameSemantics.insert(.sceneDepth) }
            if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) { config.sceneReconstruction = .mesh }
            // Prefer30fps and the smallest adequate raw image. Stored images have a separate640/1280cap.
            let formats = ARWorldTrackingConfiguration.supportedVideoFormats.filter {
                guard $0.framesPerSecond == 30 && max($0.imageResolution.width,$0.imageResolution.height) <= 2048 else { return false }
                if #available(iOS 26.0, *) { return $0.defaultColorSpace == .sRGB }
                return true
            }
            if let format = formats.sorted(by: { $0.imageResolution.width*$0.imageResolution.height < $1.imageResolution.width*$1.imageResolution.height }).first {
                config.videoFormat = format
            }
            let recorder = try CaptureSessionBridge(settings:settings, progress: { [weak self] progress in
                Task { @MainActor [weak self] in
                    guard let self, self.generation == token else { return }
                    self.frameCount = progress.frames; self.pointCount = progress.points; self.translationSpan = progress.span
                    self.tracking = progress.tracking; self.guidance = progress.guidance
                    if !self.isProcessing { self.status = progress.frames >= self.settings.maxFrames ? "View limit reached. Finish to prepare your scene." : "\(progress.frames)/\(self.settings.maxFrames) views saved privately" }
                }
            }, failure: { [weak self] message in
                Task { @MainActor [weak self] in
                    guard let self, self.generation == token else { return }
                    self.interrupt(message)
                }
            })
            bridge = recorder; session.delegate = recorder; session.delegateQueue = recorder.delegateQueue
            frameCount = 0; pointCount = 0; translationSpan = 0; preparedBundleURL = nil; capturedMeshURL = nil
            previousIdleTimer = UIApplication.shared.isIdleTimerDisabled; UIApplication.shared.isIdleTimerDisabled = true
            isCapturing = true; tracking = "Initializing"
            status = "Finding tracking features…"
            guidance = supportsDepth ? "Move slowly around a small static scene. Keep overlap between views." : "Sparse feature fallback: use textured surfaces and sideways motion. No dense captured mesh is promised."
            session.run(config,options:[.resetTracking,.removeExistingAnchors])
        } catch { self.error = error.localizedDescription; status = error.localizedDescription }
    }

    func finish() async throws -> URL {
        guard canFinish, let bridge else { throw HexelsError.invalid("Capture at least 12 translated views and 64 valid seed cells before finishing") }
        let token = generation
        isProcessing = true; isCapturing = false; session.pause(); bridge.stopAccepting()
        status = "Preparing calibrated images, geometry and surface locally…"
        defer { if generation == token { isProcessing = false }; restoreIdleTimer() }
        do {
            let result = try await withTaskCancellationHandler(operation: { try await bridge.finish() }, onCancel: { bridge.cancel() })
            guard generation == token, !Task.isCancelled else { bridge.cancel(); throw CancellationError() }
            self.bridge = nil
            preparedBundleURL = result.bundle; capturedMeshURL = result.mesh
            status = "Local scene ready. Captured ARKit surface and trained Hexels are separate representations."
            return result.bundle
        } catch {
            bridge.cancel()
            if generation == token { self.bridge = nil; self.error = error.localizedDescription; status = error.localizedDescription }
            throw error
        }
    }

    func cancel() {
        generation = UUID(); session.pause(); bridge?.cancel(); bridge = nil
        isCapturing = false; isProcessing = false; isStarting = false; tracking = "Stopped"
        status = "Capture cancelled. Incomplete local capture data is removed."
        restoreIdleTimer()
    }
    private func interrupt(_ message: String) {
        guard isCapturing || isProcessing || isStarting else { return }
        cancel(); error = message; status = message
    }
    private func restoreIdleTimer() {
        if let previousIdleTimer { UIApplication.shared.isIdleTimerDisabled = previousIdleTimer; self.previousIdleTimer = nil }
    }
}

private struct CaptureProgress: Sendable {
    let frames: Int
    let points: Int
    let span: Float
    let tracking: String
    let guidance: String
}

/// A quick AR callback queue admits at most one payload to one processing queue.
/// Lock protects admission/cancellation only; recorder work never runs under it.
private final class CaptureSessionBridge: NSObject, ARSessionDelegate, @unchecked Sendable {
    let delegateQueue = DispatchQueue(label:"HexelsEdge.capture.delegate",qos:.userInitiated)
    private let processingQueue = DispatchQueue(label:"HexelsEdge.capture.processing",qos:.userInitiated)
    private let lock = NSLock()
    private let recorder: CaptureBundleRecorder
    private let progress: @Sendable (CaptureProgress) -> Void
    private let failure: @Sendable (String) -> Void
    private var accepting = true, cancelled = false, busy = false
    private var count = 0, points = 0
    private var span: Float = 0
    private var lastTimestamp = -Double.infinity
    private var lastPose: simd_float4x4?
    private var lastGuidance = -Double.infinity
    private var anchors: [UUID: ARMeshAnchor] = [:] // Delegate queue only; bounded below.
    private var droppedMeshes = 0

    init(settings: CaptureSettings, progress: @escaping @Sendable (CaptureProgress) -> Void,
         failure: @escaping @Sendable (String) -> Void) throws {
        recorder = try CaptureBundleRecorder(settings:settings)
        self.progress = progress; self.failure = failure
        super.init()
    }
    func stopAccepting() { lock.withLock { accepting = false } }
    func cancel() {
        lock.withLock { accepting = false; cancelled = true }
        processingQueue.async { self.recorder.cancel() }
    }
    private var isCancelled: Bool { lock.withLock { cancelled } }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let tracking: String
        switch frame.camera.trackingState {
        case .normal: tracking = "Normal"
        case .notAvailable: tracking = "Unavailable"
        case .limited(let reason):
            switch reason {
            case .initializing: tracking = "Initializing"
            case .excessiveMotion: tracking = "Too much motion"
            case .insufficientFeatures: tracking = "Too few features"
            case .relocalizing: tracking = "Relocalizing"
            @unknown default: tracking = "Limited"
            }
        }
        let pose = frame.camera.transform
        let decision: (accept: Bool, note: String, publish: Bool) = lock.withLock {
            guard accepting, !cancelled, count < recorder.settings.maxFrames else { return (false,"Finish when ready.",false) }
            let announce = frame.timestamp-lastGuidance >= 0.75
            if announce { lastGuidance = frame.timestamp }
            guard tracking == "Normal" else { return (false,"Wait for normal tracking. Add light/texture and move slowly.",announce) }
            guard !busy, frame.timestamp-lastTimestamp >= 0.45 else { return (false,"Keep the scene static and move slowly.",false) }
            if let previous = lastPose {
                let delta = SIMD3(pose.columns.3.x-previous.columns.3.x,pose.columns.3.y-previous.columns.3.y,pose.columns.3.z-previous.columns.3.z)
                guard simd_length(delta) >= 0.04 else { return (false,"Move sideways at least 4 cm; turning in place adds little geometry.",announce) }
                let a = SIMD3(pose.columns.2.x,pose.columns.2.y,pose.columns.2.z), b = SIMD3(previous.columns.2.x,previous.columns.2.y,previous.columns.2.z)
                guard simd_dot(a,b) >= 0.82 else { return (false,"Return toward the last view, then move slowly with more overlap.",announce) }
            }
            busy = true; lastTimestamp = frame.timestamp; lastPose = pose
            return (true,"Saving a calibrated view…",true)
        }
        if decision.publish { publish(tracking:tracking,guidance:decision.note) }
        guard decision.accept else { return }
        let inverseView = frame.camera.viewMatrix(for:.landscapeRight).inverse
        var orientationError: Float = 0
        for column in 0..<4 { for row in 0..<4 { orientationError = max(orientationError,abs(inverseView[column][row]-pose[column][row])) } }
        let payload = CapturePayload(color:frame.capturedImage,depth:frame.sceneDepth?.depthMap,confidence:frame.sceneDepth?.confidenceMap,
            intrinsics:frame.camera.intrinsics,arCameraToWorld:pose,timestamp:frame.timestamp,
            featurePoints:frame.sceneDepth == nil ? Array(frame.rawFeaturePoints?.points.prefix(20_000) ?? []) : [],landscapePoseError:orientationError)
        processingQueue.async {
            guard !self.isCancelled else { return }
            do {
                try autoreleasepool { try self.recorder.ingest(payload) }
                self.lock.withLock { self.busy = false; self.count = self.recorder.frameCount; self.points = self.recorder.pointCount; self.span = self.recorder.translationSpan }
                self.publish(tracking:"Normal",guidance:self.recorder.canFinish ? "Good coverage so far. Continue around the scene or Finish." : "Keep moving around the scene with overlapping views.")
            } catch { self.lock.withLock { self.busy = false; self.accepting = false }; self.failure(error.localizedDescription) }
        }
    }
    private func publish(tracking: String, guidance: String) {
        let value = lock.withLock { CaptureProgress(frames:count,points:points,span:span,tracking:tracking,guidance:guidance) }
        progress(value)
    }
    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) { updateMeshes(anchors) }
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) { updateMeshes(anchors) }
    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) { for anchor in anchors { self.anchors.removeValue(forKey:anchor.identifier) } }
    private func updateMeshes(_ values: [ARAnchor]) {
        guard lock.withLock({ accepting && !cancelled }) else { return }
        for case let anchor as ARMeshAnchor in values {
            let geometry = anchor.geometry
            guard geometry.vertices.format == .float3, geometry.faces.indexCountPerPrimitive == 3,
                  [2,4].contains(geometry.faces.bytesPerIndex), geometry.vertices.count > 0, geometry.faces.count > 0,
                  geometry.vertices.count <= 250_000, geometry.faces.count <= 500_000 else { droppedMeshes += 1; continue }
            let bytes = geometry.vertices.buffer.length+geometry.faces.buffer.length
            let others = anchors.filter { $0.key != anchor.identifier }.values
            let otherBytes = others.reduce(0) { $0+$1.geometry.vertices.buffer.length+$1.geometry.faces.buffer.length }
            let otherVertices = others.reduce(0) { $0+$1.geometry.vertices.count }
            let otherFaces = others.reduce(0) { $0+$1.geometry.faces.count }
            guard anchors[anchor.identifier] != nil || anchors.count < 96, bytes+otherBytes <= 32*1_048_576,
                  geometry.vertices.count+otherVertices <= 250_000, geometry.faces.count+otherFaces <= 500_000 else { droppedMeshes += 1; continue }
            anchors[anchor.identifier] = anchor
        }
    }
    func session(_ session: ARSession, didFailWithError error: Error) { failure("AR capture failed: \(error.localizedDescription)") }
    func sessionWasInterrupted(_ session: ARSession) { failure("AR capture was interrupted. Start a new scan so all cameras share one reliable coordinate frame.") }

    func finish() async throws -> (bundle: URL, mesh: URL?) {
        try await withCheckedThrowingContinuation { continuation in
            delegateQueue.async {
                if self.isCancelled { continuation.resume(throwing:CancellationError()); return }
                let meshes = self.anchors.sorted { $0.key.uuidString < $1.key.uuidString }.map { _,anchor in
                    let geometry = anchor.geometry
                    return CapturedMeshBuffers(vertices:Data(bytes:geometry.vertices.buffer.contents(),count:geometry.vertices.buffer.length),
                        vertexCount:geometry.vertices.count,vertexStride:geometry.vertices.stride,vertexOffset:geometry.vertices.offset,
                        indices:Data(bytes:geometry.faces.buffer.contents(),count:geometry.faces.buffer.length),triangleCount:geometry.faces.count,
                        indexBytes:geometry.faces.bytesPerIndex,transform:anchor.transform)
                }
                let dropped = self.droppedMeshes
                self.processingQueue.async {
                    do {
                        guard !self.isCancelled else { throw CancellationError() }
                        let result = try autoreleasepool { try self.recorder.finish(meshes:meshes,droppedMeshes:dropped) }
                        guard !self.isCancelled else { self.recorder.cancel(); throw CancellationError() }
                        continuation.resume(returning:result)
                    } catch { self.recorder.cancel(); continuation.resume(throwing:error) }
                }
            }
        }
    }
}
