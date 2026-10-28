import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var model = EdgeModel()
    @State private var importing = false
    @State private var capturing = false
    @State private var surfaceToPreview: SurfacePreviewItem?
    @State private var fineTuning = false
    @State private var showingDeviceTest = false
    @State private var previewZoomed = false
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    private var hasScene: Bool { model.image != nil }
    private var isRunning: Bool { model.isTraining || model.isBenchmarking }
    private var usesActionBar: Bool { horizontalSizeClass == .compact || dynamicTypeSize.isAccessibilitySize }
    private var settingsNeedApply: Bool {
        hasScene && (model.settings.imageLongEdge != model.effectiveImageLongEdge
            || model.settings.trainingViewLimit != model.selectedTrainingViewCount)
    }

    var body: some View {
        NavigationStack {
            GeometryReader { geometry in
                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        heading
                        if geometry.size.width >= 900 && !dynamicTypeSize.isAccessibilitySize {
                            HStack(alignment: .top, spacing: 24) {
                                previewColumn.frame(maxWidth: .infinity)
                                inspector.frame(width: 328)
                            }
                        } else {
                            previewColumn
                            inspector
                        }
                        privacyNote
                    }
                    .frame(maxWidth: 1380)
                    .padding(.horizontal, geometry.size.width > 700 ? 32 : 20)
                    .padding(.top, 24)
                    .padding(.bottom, 32)
                    .frame(maxWidth: .infinity)
                }
                .scrollDisabled(previewZoomed)
                .background(Color(uiColor: .systemGroupedBackground))
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                if hasScene && usesActionBar {
                    trainingAction
                        .padding(.horizontal, 20).padding(.vertical, 12)
                        .background(.ultraThinMaterial)
                        .overlay(alignment: .top) { Divider() }
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    HStack(spacing: 9) {
                        HexelsMark().frame(width: 28, height: 28)
                        Text("Hexel").font(.headline)
                        Text("EDGE").font(.caption2.weight(.semibold)).tracking(1.8).foregroundStyle(.secondary)
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Hexel Edge")
                }
                ToolbarItem(placement: .topBarTrailing) {
                    HStack(spacing: 18) {
                        Button { capturing = true } label: { Label("Capture scene", systemImage: "camera") }
                        Button { importing = true } label: { Label("Open prepared folder", systemImage: "folder") }
                    }
                    .disabled(model.isBusy)
                }
            }
            .tint(.indigo)
            .fileImporter(isPresented: $importing, allowedContentTypes: [.folder]) { result in
                switch result {
                case .success(let url): model.importScene(url)
                case .failure(let error): model.error = error.localizedDescription
                }
            }
            .fullScreenCover(isPresented: $capturing) {
                CaptureView(maximumLongEdge: model.capabilities.maxImageLongEdge,
                            maximumSeeds: min(20_000, model.capabilities.maxPrimitiveCount)) { url in
                    // The capture controller has stopped its ARSession before
                    // returning the finished local bundle to the training model.
                    capturing = false
                    model.importScene(url)
                }
            }
            .sheet(item: $surfaceToPreview) { item in SurfacePreviewView(item: item) }
            .alert("Unable to complete this step", isPresented: Binding(
                get: { model.error != nil }, set: { if !$0 { model.error = nil } })) {
                    Button("OK") { model.error = nil }
                } message: { Text(model.error ?? "") }
            .task { model.restore() }
            .onChange(of: scenePhase) { _, phase in
                if phase != .active, model.isBusy { model.stop() }
                if phase == .active && !capturing { model.refreshPreview() }
            }
        }
    }

    private var heading: some View {
        VStack(alignment: .leading, spacing: 8) {
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .firstTextBaseline) { headingTitle; Spacer(minLength: 12); privacyBadge }
                VStack(alignment: .leading, spacing: 12) { headingTitle; privacyBadge }
            }
            Text(hasScene ? "Explore your scene and refine it at your own pace." : "Make room for detail. Keep your scene close.")
                .font(.subheadline).foregroundStyle(.secondary)
        }
    }

    private var headingTitle: some View {
        Text(hasScene ? "Scene studio" : "A new perspective.")
            .font(.largeTitle.weight(.bold)).tracking(-0.8)
    }

    private var privacyBadge: some View {
        Label("On-device", systemImage: "lock.shield")
            .font(.caption.weight(.medium)).foregroundStyle(.secondary)
            .padding(.horizontal, 11).padding(.vertical, 7)
            .background(.quaternary, in: Capsule())
    }

    @ViewBuilder private var previewColumn: some View {
        if let image = model.image {
            VStack(spacing: 20) {
                previewCard(image)
                progressCard
                qualityCheck
                surfaceExports
            }
        } else {
            emptyScene
        }
    }

    private func previewCard(_ image: UIImage) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Scene preview").font(.headline)
                    Text(model.frameName).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                    if model.isNovelView {
                        Text("Synthesized viewpoint").font(.caption).foregroundStyle(.secondary)
                    }
                }
                Spacer()
                statusPill
            }
            ZoomableSceneImage(image: image, isZoomed: $previewZoomed)
                .frame(maxWidth: .infinity, alignment: .center).id(model.frameName)
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 12) { comparisonPicker; Spacer(minLength: 8); cameraControls }
                VStack(alignment: .leading, spacing: 12) { comparisonPicker; cameraControls }
            }
            .disabled(model.isBusy)
        }
        .edgeCard()
    }

    private var statusPill: some View {
        HStack(spacing: 5) {
            Circle().fill(isRunning ? Color.orange : Color.green).frame(width: 6, height: 6)
            Text(isRunning ? "Refining" : (model.checkpoint != nil ? "Saved" : "Ready"))
                .font(.caption.weight(.medium))
        }
        .padding(.horizontal, 10).padding(.vertical, 7)
        .background(.quaternary, in: Capsule())
        .accessibilityElement(children: .combine)
    }

    private var comparisonPicker: some View {
        HStack(spacing: 3) {
            comparisonButton("Reconstruction", selected: !model.isShowingReference, action: model.showReconstruction)
            comparisonButton("Reference", selected: model.isShowingReference, action: model.showReference)
                .disabled(model.isNovelView)
        }
        .padding(4).background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
        .accessibilityLabel("Preview image")
    }

    private func comparisonButton(_ title: String, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title).font(.caption.weight(.semibold)).padding(.horizontal, 11).padding(.vertical, 9)
                .foregroundStyle(selected ? Color.primary : Color.secondary)
                .background(selected ? Color(uiColor: .secondarySystemGroupedBackground) : .clear,
                            in: RoundedRectangle(cornerRadius: 9))
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }

    private var nextViewButton: some View {
        Button(action: model.nextView) { Label("Next view", systemImage: "arrow.right") }
            .font(.subheadline.weight(.medium)).buttonStyle(.borderless)
            .accessibilityHint("Renders the next prepared camera view")
    }

    private var cameraControls: some View {
        HStack(spacing: 18) {
            Button(action: model.novelView) {
                Label("Between views", systemImage: "view.3d")
                    .font(.caption.weight(.semibold))
            }
            .buttonStyle(.borderless)
            .accessibilityHint("Renders a synthesized viewpoint halfway between two cameras")
            .accessibilityAddTraits(model.isNovelView ? .isSelected : [])
            nextViewButton
        }
    }

    private var progressCard: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(isRunning ? "Finding the finer details" : "Refine your scene").font(.headline)
                    Text(model.status).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 12)
                if model.isBusy && !isRunning { ProgressView().accessibilityLabel("Working") }
            }
            if isRunning {
                ProgressView(value: model.isBenchmarking ? model.benchmarkProgress : model.trainingProgress)
                    .tint(.indigo)
                    .accessibilityLabel(model.isBenchmarking ? "Device test progress" : "Training progress")
                HStack {
                    Text("Step \(model.step.formatted())").font(.caption.monospacedDigit())
                    Spacer()
                    if let seconds = model.estimatedRemainingSeconds {
                        Text("About \(shortDuration(seconds)) left").font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 20) { fitMetrics }
                VStack(alignment: .leading, spacing: 12) { fitMetrics }
            }
            if !usesActionBar { trainingAction }
            if !isRunning {
                HStack(spacing: 20) {
                    Button(action: model.save) { Label("Save", systemImage: "square.and.arrow.down") }
                        .disabled(model.isBusy)
                    Spacer()
                    if let checkpoint = model.checkpoint {
                        ShareLink(item: checkpoint) { Label("Export checkpoint", systemImage: "square.and.arrow.up") }
                            .disabled(model.isBusy)
                    }
                }
                .font(.subheadline)
                if settingsNeedApply {
                    Label("Apply your new image settings before refining.", systemImage: "info.circle")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .edgeCard()
    }

    @ViewBuilder private var trainingAction: some View {
        if isRunning {
            Button(action: model.stop) {
                Label("Pause & save", systemImage: "pause.fill").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).controlSize(.large)
        } else if settingsNeedApply {
            Button(action: model.applySettings) {
                Label("Apply settings & start fresh", systemImage: "arrow.counterclockwise")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).controlSize(.large).disabled(model.isBusy)
        } else {
            Button { model.train(additionalSteps: model.settings.additionalSteps) } label: {
                Label("Refine · \(model.settings.additionalSteps.formatted()) steps", systemImage: "sparkles")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).controlSize(.large).disabled(model.isBusy)
        }
    }

    @ViewBuilder private var fitMetrics: some View {
        EdgeMetric(title: "Training images", value: model.selectedTrainingViewCount.formatted())
        EdgeMetric(title: "Image detail", value: "\(model.effectiveImageLongEdge) px")
        EdgeMetric(title: "Steps completed", value: model.step.formatted())
    }

    private var qualityCheck: some View {
        HStack(alignment: .center, spacing: 16) {
            Image(systemName: "checkmark.viewfinder").font(.title2).foregroundStyle(.indigo)
                .frame(width: 44, height: 44).background(Color.indigo.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
            VStack(alignment: .leading, spacing: 4) {
                Text("Quality check").font(.subheadline.weight(.semibold))
                if let psnr = model.heldOutPSNR {
                    Text("\(psnr, format: .number.precision(.fractionLength(2))) dB · reserved views")
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                } else {
                    Text("Compare reserved views with their photos.").font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 0)
            Button(action: model.evaluateQuality) {
                if model.isEvaluating { ProgressView().accessibilityLabel("Checking quality") }
                else { Text("Check").font(.subheadline.weight(.semibold)) }
            }
            .buttonStyle(.bordered).disabled(model.isBusy)
        }
        .edgeCard()
    }

    private var inspector: some View {
        VStack(spacing: 20) {
            if hasScene { qualitySettings }
            deviceCard
            if hasScene { deviceTest }
        }
    }

    private var surfaceExports: some View {
        VStack(alignment: .leading, spacing: 18) {
            Label("Surfaces", systemImage: "cube.transparent").font(.headline)
            if let captured = model.capturedSurfaceURL {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Captured surface").font(.subheadline.weight(.semibold))
                    Text("Geometry recorded during camera capture · \(captured.pathExtension.uppercased())")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack(spacing: 20) {
                        Button { surfaceToPreview = SurfacePreviewItem(url: captured, title: "Captured surface") } label: {
                            Label("Inspect in 3D", systemImage: "rotate.3d")
                        }
                        ShareLink(item: captured) { Label("Export", systemImage: "square.and.arrow.up") }
                    }
                    .font(.subheadline).disabled(model.isBusy)
                }
                Divider()
            }
            VStack(alignment: .leading, spacing: 10) {
                Text("Surface patches").font(.subheadline.weight(.semibold))
                Text("Colored Hexel cells exported as PLY. The cells remain independent and may have gaps.")
                    .font(.caption).foregroundStyle(.secondary)
                HStack(spacing: 20) {
                    Button(action: model.exportHexelSurface) {
                        Label(model.hexelSurfaceURL == nil ? "Prepare export" : "Update export", systemImage: "cube.transparent")
                    }
                    if let patches = model.hexelSurfaceURL {
                        ShareLink(item: patches) { Label("Export", systemImage: "square.and.arrow.up") }
                    }
                }
                .font(.subheadline).disabled(model.isBusy)
            }
        }
        .edgeCard()
    }

    private var qualitySettings: some View {
        VStack(alignment: .leading, spacing: 18) {
            Label("Quality", systemImage: "slider.horizontal.3").font(.headline)
            VStack(spacing: 8) {
                ForEach(QualityPreset.allCases) { preset in
                    Button { model.selectPreset(preset) } label: {
                        HStack(alignment: .top, spacing: 10) {
                            Image(systemName: model.settings.preset == preset ? "checkmark.circle.fill" : "circle")
                                .foregroundStyle(model.settings.preset == preset ? Color.indigo : Color.secondary)
                                .font(.title3)
                            VStack(alignment: .leading, spacing: 4) {
                                Text(preset.title).font(.subheadline.weight(.semibold)).foregroundStyle(.primary)
                                Text(presetCaption(preset)).font(.caption).foregroundStyle(.secondary)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            Spacer(minLength: 0)
                        }
                        .padding(12)
                        .background(model.settings.preset == preset ? Color.indigo.opacity(0.08) : .clear,
                                    in: RoundedRectangle(cornerRadius: 14))
                        .contentShape(RoundedRectangle(cornerRadius: 14))
                    }
                    .buttonStyle(.plain)
                    .accessibilityAddTraits(model.settings.preset == preset ? .isSelected : [])
                }
            }
            .disabled(model.isBusy)
            DisclosureGroup("Fine-tune", isExpanded: $fineTuning) {
                VStack(alignment: .leading, spacing: 18) {
                    Stepper(value: $model.settings.trainingViewLimit, in: 1...maximumTrainingViews) {
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Training images").font(.subheadline)
                            Text(model.settings.trainingViewLimit.formatted()).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                        }
                    }
                    Picker("Image detail", selection: $model.settings.imageLongEdge) {
                        ForEach(imageSizes, id: \.self) { size in Text("\(size) px").tag(size) }
                    }
                    Picker("Additional steps", selection: $model.settings.additionalSteps) {
                        ForEach(stepBudgets, id: \.self) { count in Text(count.formatted()).tag(count) }
                    }
                    Text("Limits follow this device and the images in your scene. More steps take longer; more images and detail use more memory.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(.top, 14)
                .disabled(model.isBusy)
            }
            .font(.subheadline.weight(.medium))
            if settingsNeedApply {
                Button(action: model.applySettings) {
                    Text("Apply settings & start fresh").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered).controlSize(.large).disabled(model.isBusy)
                Text("Starts from the scene's initial model. Your current checkpoint is kept locally.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .edgeCard()
    }

    private var maximumTrainingViews: Int {
        max(1, min(model.availableTrainingViewCount, model.capabilities.maxTrainingViews))
    }

    private var imageSizes: [Int] {
        let maximum = max(1, min(model.sourceImageLongEdge, model.capabilities.maxImageLongEdge))
        return Array(Set([32, 64, 128, 256, 320, 480, 640, 960, 1280, maximum]
            .filter { $0 <= maximum })).sorted()
    }

    private var stepBudgets: [Int] {
        Array(Set([100, 1_000, 3_000, 5_000, 10_000, 20_000, 30_000, model.settings.additionalSteps])).sorted()
    }

    private func presetCaption(_ preset: QualityPreset) -> String {
        switch preset {
        case .automatic: "A starting point chosen for this device."
        case .quick: "A lighter preview with less waiting."
        case .balanced: "A moderate budget for everyday detail."
        case .high: "More images, finer pixels, and a longer fit."
        }
    }

    private var deviceCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("This device", systemImage: UIDevice.current.userInterfaceIdiom == .pad ? "ipad.landscape" : "iphone")
                .font(.headline)
            VStack(alignment: .leading, spacing: 4) {
                Text(model.capabilities.gpuName).font(.subheadline.weight(.medium))
                Text("\(model.capabilities.physicalMemoryGiB, format: .number.precision(.fractionLength(1))) GiB system memory")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let telemetry = model.telemetry {
                Divider()
                EdgeInfoRow(title: "Thermal state", value: telemetry.thermalState.capitalized,
                            symbol: "thermometer.medium", color: thermalColor(telemetry.thermalState))
                if let battery = telemetry.batteryFraction {
                    EdgeInfoRow(title: telemetry.isCharging ? "Battery · charging" : "Battery",
                                value: "\(Int((battery * 100).rounded()))%",
                                symbol: telemetry.isCharging ? "battery.100percent.bolt" : "battery.100percent")
                }
                if telemetry.footprintBytes > 0 {
                    EdgeInfoRow(title: "App memory", value: "\(Int(telemetry.footprintBytes / 1_048_576)) MiB", symbol: "memorychip")
                }
                if let milliseconds = telemetry.stepMilliseconds {
                    EdgeInfoRow(title: "Average step", value: "\(milliseconds.formatted(.number.precision(.fractionLength(0)))) ms", symbol: "timer")
                }
                if telemetry.elapsedSeconds > 0 {
                    EdgeInfoRow(title: "Elapsed", value: shortDuration(telemetry.elapsedSeconds), symbol: "clock")
                }
                if telemetry.lowPowerMode {
                    Label("Low Power Mode is on", systemImage: "leaf")
                        .font(.caption).foregroundStyle(.orange)
                }
            }
            Text("Automatic starts conservatively. Time estimates use measured steps, and refinement pauses if the device needs to cool.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .edgeCard()
    }

    private var deviceTest: some View {
        VStack(alignment: .leading, spacing: 14) {
            DisclosureGroup("Device test", isExpanded: $showingDeviceTest) {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Targets 30,000 steps and 20 minutes, with a 30-minute safety limit. Includes a short warm-up and a 30-second settling period. Keep the device unplugged for battery readings.")
                        .font(.caption).foregroundStyle(.secondary)
                    if !model.benchmarkStatus.isEmpty {
                        Text(model.benchmarkStatus).font(.caption).fixedSize(horizontal: false, vertical: true)
                    }
                    if model.isBenchmarking {
                        ProgressView(value: model.benchmarkProgress).tint(.indigo)
                    } else {
                        Button { model.startBenchmark() } label: {
                            Label("Run device test", systemImage: "waveform.path.ecg").frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered).disabled(model.isBusy || settingsNeedApply)
                    }
                    if let report = model.reportURL {
                        ShareLink(item: report) { Label("Export test report", systemImage: "square.and.arrow.up") }
                            .font(.subheadline).disabled(model.isBusy)
                    }
                }
                .padding(.top, 14)
            }
            .font(.subheadline.weight(.semibold))
        }
        .edgeCard()
    }

    private var emptyScene: some View {
        VStack(spacing: 24) {
            HexelsSceneIllustration().frame(width: 210, height: 170).padding(.top, 24)
            VStack(spacing: 10) {
                Text("Bring your scene into focus").font(.title2.weight(.semibold))
                Text("Capture a still subject from several angles, then explore and refine its 3D scene on this device.")
                    .font(.subheadline).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    .frame(maxWidth: 380)
            }
            Button { capturing = true } label: {
                Label("Capture scene", systemImage: "camera.fill").padding(.horizontal, 20)
            }
            .buttonStyle(.borderedProminent).controlSize(.large).disabled(model.isBusy)
            Button { importing = true } label: {
                Label("Open prepared folder", systemImage: "folder")
            }
            .buttonStyle(.borderless).disabled(model.isBusy)
            if model.isBusy { ProgressView(model.status).font(.caption) }
            Text("You can also open an existing Hexels scene folder with its images and camera positions.")
                .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
                .frame(maxWidth: 400).padding(.bottom, 24)
        }
        .frame(maxWidth: .infinity).padding(28)
        .background {
            RoundedRectangle(cornerRadius: 24)
                .fill(Color(uiColor: .secondarySystemGroupedBackground))
                .overlay(alignment: .top) {
                    Ellipse().fill(Color.indigo.opacity(0.07)).frame(height: 170).blur(radius: 60).clipped()
                }
        }
        .clipShape(RoundedRectangle(cornerRadius: 24))
        .overlay(RoundedRectangle(cornerRadius: 24).strokeBorder(.primary.opacity(0.05)))
    }

    private var privacyNote: some View {
        Label("Scenes and refinement stay on this device. Sharing is always your choice.", systemImage: "lock")
            .font(.caption).foregroundStyle(.secondary).frame(maxWidth: .infinity)
            .multilineTextAlignment(.center)
    }

    private func thermalColor(_ value: String) -> Color {
        switch value {
        case "nominal": .green
        case "fair": .orange
        case "serious", "critical": .red
        default: .secondary
        }
    }

    private func shortDuration(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "a moment" }
        if seconds < 60 { return "\(max(1, Int(seconds))) sec" }
        if seconds < 3600 { return "\(Int((seconds / 60).rounded())) min" }
        return "\(Int(seconds / 3600)) hr \(Int(seconds.truncatingRemainder(dividingBy: 3600) / 60)) min"
    }
}
