import Foundation

enum QualityPreset: String, CaseIterable, Codable, Identifiable, Sendable {
    case automatic, quick, balanced, high
    var id: String { rawValue }
    var title: String { switch self { case .automatic: "Automatic"; case .quick: "Quick"; case .balanced: "Balanced"; case .high: "High quality" } }
    var detail: String {
        switch self {
        case .automatic: "Conservative limits from available memory and Metal support."
        case .quick: "A small preview with fewer training views and pixels."
        case .balanced: "More detail with a moderate training budget."
        case .high: "Use more of the prepared images and a longer fit."
        }
    }
}

struct SceneSettings: Codable, Equatable, Sendable {
    var preset: QualityPreset = .automatic
    var imageLongEdge: Int = 960
    var trainingViewLimit: Int = 48
    var additionalSteps: Int = 10_000
}

struct DeviceCapabilities: Sendable {
    let gpuName: String
    let supportsMetal: Bool
    let physicalMemoryGiB: Double
    let availableMemoryMiB: Double
    let recommendedGPUWorkingSetMiB: Double
    let maxImageLongEdge: Int
    let maxTrainingViews: Int
    let maxPrimitiveCount: Int
    let maxWorkspaceMiB: Int
    var summary: String {
        "\(gpuName) · \(String(format: "%.1f", physicalMemoryGiB)) GB unified memory"
    }
    var recommendationNote: String {
        "Memory-based starting limits. A runtime estimate appears after 20 measured steps; GPU memory guidance is not a RAM entitlement."
    }
}

enum DevicePolicy {
    /// Advisory available memory is sampled again at admission, not cached as free RAM.
    static func capabilities(physicalBytes: UInt64, availableBytes: UInt64,
                             gpuWorkingSetBytes: UInt64, gpuName: String,
                             supportsMetal: Bool) -> DeviceCapabilities {
        let mib: UInt64 = 1_048_576
        let physicalBudget = physicalBytes / 10
        let availableBudget = availableBytes / 3
        let gpuBudget = gpuWorkingSetBytes > 0 ? gpuWorkingSetBytes / 4 : physicalBudget
        let budgetMiB = Int(min(UInt64(768) * mib, physicalBudget, availableBudget, gpuBudget) / mib)
        let boundedBudget = max(0, budgetMiB)
        let high = boundedBudget >= 384
        return DeviceCapabilities(gpuName: gpuName, supportsMetal: supportsMetal,
            physicalMemoryGiB: Double(physicalBytes) / 1_073_741_824,
            availableMemoryMiB: Double(availableBytes) / Double(mib),
            recommendedGPUWorkingSetMiB: Double(gpuWorkingSetBytes) / Double(mib),
            maxImageLongEdge: high ? 1280 : (boundedBudget >= 192 ? 960 : 640),
            maxTrainingViews: high ? 96 : (boundedBudget >= 192 ? 48 : 24),
            maxPrimitiveCount: boundedBudget >= 192 ? 50_000 : 25_000,
            maxWorkspaceMiB: boundedBudget)
    }

    static func settings(for preset: QualityPreset, capabilities: DeviceCapabilities) -> SceneSettings {
        let edge: Int, views: Int, steps: Int
        switch preset {
        case .quick: (edge, views, steps) = (320, 12, 1_000)
        case .balanced: (edge, views, steps) = (640, 24, 10_000)
        case .high: (edge, views, steps) = (1280, 96, 30_000)
        case .automatic: (edge, views, steps) = (960, 48, 10_000)
        }
        return SceneSettings(preset: preset, imageLongEdge: min(edge, capabilities.maxImageLongEdge),
                             trainingViewLimit: min(views, capabilities.maxTrainingViews), additionalSteps: steps)
    }

    /// Even coverage of the original training trajectory; held-out indices never enter this list.
    static func selectTrainingIndices(_ training: [Int], limit: Int) -> [Int] {
        guard !training.isEmpty, limit > 0 else { return [] }
        let count = min(training.count, limit)
        if count == training.count { return training }
        if count == 1 { return [training[training.count / 2]] }
        return (0..<count).map { training[Int((Double($0) * Double(training.count - 1) / Double(count - 1)).rounded())] }
    }

    static func resizedDimensions(width: Int, height: Int, longEdge: Int) -> (width: Int, height: Int) {
        let scale = min(1, Double(max(1, longEdge)) / Double(max(width, height)))
        return (max(1, Int((Double(width) * scale).rounded())), max(1, Int((Double(height) * scale).rounded())))
    }

    static func resizedIntrinsics(_ k: [Float], oldWidth: Int, oldHeight: Int,
                                  newWidth: Int, newHeight: Int) -> [Float] {
        let sx = Float(newWidth) / Float(oldWidth), sy = Float(newHeight) / Float(oldHeight)
        return [k[0] * sx, k[1] * sy, (k[2] + 0.5) * sx - 0.5, (k[3] + 0.5) * sy - 0.5]
    }

    static func pauseReason(thermal: String, batteryFraction: Double?, batteryState: String,
                            lowPowerMode: Bool, requireUnplugged: Bool) -> String? {
        if thermal == "serious" || thermal == "critical" { return "Paused for device thermal pressure. Resume after it cools." }
        if let batteryFraction, batteryFraction <= 0.15, batteryState == "unplugged" { return "Paused to preserve the last 15% of battery." }
        if lowPowerMode { return "Paused because Low Power Mode is enabled." }
        if requireUnplugged && batteryState != "unplugged" { return "Unplug this device to start a battery benchmark." }
        return nil
    }
}
