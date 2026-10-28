import ARKit
import AVFoundation
import SceneKit
import SwiftUI
import UIKit

/// Guided local capture. CaptureController owns camera configuration and data.
struct CaptureView: View {
    let maximumLongEdge: Int
    let maximumSeeds: Int
    let onComplete: (URL) -> Void
    @StateObject private var controller = CaptureController()
    @State private var targetFrames = 48
    @State private var longEdge = 640
    @State private var hasStarted = false
    @State private var completed = false
    @State private var isStarting = false
    @State private var isFinishing = false
    @State private var operation: Task<Void, Never>?
    @State private var errorMessage: String?
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.openURL) private var openURL
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @Environment(\.verticalSizeClass) private var verticalSizeClass

    private var permissionDenied: Bool {
        let authorization = AVCaptureDevice.authorizationStatus(for: .video)
        return authorization == .denied || authorization == .restricted
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            CaptureCameraView(session: controller.session).ignoresSafeArea()
            LinearGradient(colors: [.black.opacity(0.55), .clear, .black.opacity(0.35)],
                           startPoint: .top, endPoint: .bottom).ignoresSafeArea().allowsHitTesting(false)
            if hasStarted && controller.isCapturing {
                Image(systemName: "viewfinder").font(.system(size: 64, weight: .ultraLight))
                    .foregroundStyle(.white.opacity(0.65)).accessibilityHidden(true).allowsHitTesting(false)
            }
            VStack(spacing: 18) {
                topBar
                if dynamicTypeSize >= .xxLarge || verticalSizeClass == .compact {
                    ScrollView {
                        capturePanel.frame(maxWidth: 540).frame(maxWidth: .infinity)
                    }
                    .scrollBounceBehavior(.basedOnSize)
                } else {
                    Spacer(minLength: 20)
                    capturePanel.frame(maxWidth: 540).frame(maxWidth: .infinity)
                }
            }
            .padding(20)
            if controller.isProcessing || isFinishing {
                Color.black.opacity(0.65).ignoresSafeArea()
                VStack(spacing: 18) {
                    ProgressView().controlSize(.large).tint(.white)
                    Text("Preparing your scene").font(.title2.weight(.semibold))
                    Text(controller.status).font(.subheadline).multilineTextAlignment(.center)
                        .foregroundStyle(.white.opacity(0.8)).frame(maxWidth: 360)
                    Button("Cancel", role: .cancel, action: cancel).buttonStyle(.bordered)
                }
                .foregroundStyle(.white).padding(28)
            }
        }
        .environment(\.colorScheme, .dark)
        .tint(.indigo)
        .interactiveDismissDisabled(controller.isCapturing || controller.isProcessing)
        .alert("Capture needs attention", isPresented: Binding(
            get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
                Button("OK") { errorMessage = nil }
            } message: { Text(errorMessage ?? "") }
        .onChange(of: controller.error) { _, message in errorMessage = message }
        .onChange(of: scenePhase) { _, phase in
            // The initial camera permission sheet is inactive without a running
            // session. Do not dismiss that sheet before the user can answer it.
            if phase == .background || (phase == .inactive && (controller.isCapturing || controller.isProcessing)) {
                cancel()
            }
        }
        .onDisappear {
            operation?.cancel()
            if completed { controller.session.pause() }
            else { controller.cancel() }
        }
    }

    @ViewBuilder private var topBar: some View {
        if dynamicTypeSize.isAccessibilitySize {
            VStack(alignment: .leading, spacing: 10) {
                cancelButton
                capturePrivacyBadge
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            HStack { cancelButton; Spacer(); capturePrivacyBadge }
        }
    }

    private var cancelButton: some View {
        Button(action: cancel) { Label("Cancel", systemImage: "xmark") }
            .buttonStyle(.plain).foregroundStyle(.white)
            .padding(.horizontal, 16).padding(.vertical, 10).frame(minHeight: 44)
            .background(.black.opacity(0.65), in: Capsule())
    }

    private var capturePrivacyBadge: some View {
        Label("Private capture", systemImage: "lock.shield")
            .font(.subheadline.weight(.medium)).foregroundStyle(.white)
            .padding(.horizontal, 13).padding(.vertical, 10)
            .background(.black.opacity(0.65), in: Capsule())
    }

    private var capturePanel: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 7) {
                Text(hasStarted ? (controller.canFinish ? "Ready when you are" : "Move slowly around your scene") : "Start with a still subject")
                    .font(.title3.weight(.semibold))
                Text(hasStarted ? controller.guidance : "Keep nearby surfaces in view. Move sideways and around the subject, keeping plenty of overlap between views.")
                    .font(.subheadline).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Label("Hold the device in landscape for upright saved previews.", systemImage: "rotate.right")
                    .font(.caption.weight(.medium)).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if hasStarted {
                captureMetrics
                ProgressView(value: min(1, Double(controller.frameCount) / Double(max(1, controller.maxFrames))))
                    .tint(.indigo)
                    .accessibilityLabel("Useful views captured")
                    .accessibilityValue("\(controller.frameCount) of \(controller.maxFrames)")
                Text(controller.status).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Text("\(controller.pointCount.formatted()) scene points recorded")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                if controller.isCapturing {
                    Button(action: finish) {
                        Label("Build scene", systemImage: "cube.transparent").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent).controlSize(.large)
                .disabled(!controller.canFinish || controller.isProcessing || isFinishing)
                    if !controller.canFinish {
                        Text("Collect at least 12 useful views with sideways movement and enough scene detail before building.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                } else {
                    startButton
                }
            } else {
                captureSettings
                startButton
            }
            if permissionDenied, let settings = URL(string: UIApplication.openSettingsURLString) {
                Button { openURL(settings) } label: { Label("Allow camera in Settings", systemImage: "gearshape") }
                    .buttonStyle(.bordered).frame(maxWidth: .infinity)
            }
            Label("Images and geometry stay on this device.", systemImage: "lock")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(22)
        .background(.thickMaterial, in: RoundedRectangle(cornerRadius: 24))
        .overlay(RoundedRectangle(cornerRadius: 24).strokeBorder(.white.opacity(0.1)))
    }

    private var captureSettings: some View {
        VStack(spacing: 12) {
            Picker("Useful views", selection: $targetFrames) {
                Text("24 · quick").tag(24)
                Text("48 · balanced").tag(48)
                Text("64 · more coverage").tag(64)
            }
            Picker("Image detail", selection: $longEdge) {
                Text("640 px").tag(640)
                if maximumLongEdge >= 1280 { Text("1280 px").tag(1280) }
            }
        }
        .font(.subheadline)
    }

    private var captureMetrics: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 20) { metrics }
            VStack(alignment: .leading, spacing: 12) { metrics }
        }
    }

    @ViewBuilder private var metrics: some View {
        EdgeMetric(title: "Useful views", value: "\(controller.frameCount) / \(controller.maxFrames)")
        EdgeMetric(title: "Tracking", value: controller.tracking.capitalized)
        EdgeMetric(title: "Depth", value: controller.supportsDepth ? "Available" : "Tracked points")
    }

    private var startButton: some View {
        Button {
            guard !isStarting else { return }
            isStarting = true
            hasStarted = true
            let settings = CaptureSettings(maxFrames: targetFrames, longEdge: longEdge,
                                           maxSeeds: min(20_000, max(64, maximumSeeds)))
            operation = Task {
                defer { isStarting = false }
                await controller.start(settings: settings)
            }
        } label: {
            HStack {
                if isStarting { ProgressView().tint(.white) }
                Label(isStarting ? "Starting camera…" : (hasStarted ? "Try again" : "Start capture"), systemImage: "camera.fill")
            }
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent).controlSize(.large)
        .disabled(isStarting || controller.isProcessing || permissionDenied)
    }

    private func finish() {
        guard !isFinishing else { return }
        isFinishing = true
        operation = Task {
            defer { isFinishing = false }
            do {
                let url = try await controller.finish()
                try Task.checkCancellation()
                // finish pauses ARSession before returning. Training import
                // begins only after the camera no longer owns its resources.
                completed = true
                onComplete(url)
            } catch is CancellationError {
                // Cancellation closes capture without presenting a second alert.
            } catch { errorMessage = error.localizedDescription }
        }
    }

    private func cancel() {
        operation?.cancel()
        controller.cancel()
        dismiss()
    }
}

private struct CaptureCameraView: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.backgroundColor = .black
        view.session = session
        view.scene = SCNScene()
        view.preferredFramesPerSecond = 30
        view.automaticallyUpdatesLighting = true
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}

    static func dismantleUIView(_ uiView: ARSCNView, coordinator: ()) {
        uiView.session.pause()
        uiView.isPlaying = false
    }
}
