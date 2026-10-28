import Foundation

/// Standalone policy checks: swiftc DevicePolicy.swift DevicePolicyChecks.swift -o /tmp/edge-policy-checks.
@main
struct DevicePolicyChecks {
    static func require(_ condition: Bool, _ message: String) {
        guard condition else { fatalError(message) }
    }

    static func main() {
        let gib: UInt64 = 1_073_741_824
        let mib: UInt64 = 1_048_576
        let capable = DevicePolicy.capabilities(physicalBytes: 8*gib, availableBytes: 4*gib,
            gpuWorkingSetBytes: 6*gib, gpuName: "Test GPU", supportsMetal: true)
        require(capable.maxWorkspaceMiB <= 768, "Global workspace ceiling exceeded")
        for preset in QualityPreset.allCases {
            let settings = DevicePolicy.settings(for: preset, capabilities: capable)
            require(settings.imageLongEdge <= capable.maxImageLongEdge, "Preset escaped resolution cap")
            require(settings.trainingViewLimit <= capable.maxTrainingViews, "Preset escaped view cap")
            require((1...100_000).contains(settings.additionalSteps), "Preset escaped iteration cap")
        }
        let constrained = DevicePolicy.capabilities(physicalBytes: 8*gib, availableBytes: 12*mib,
            gpuWorkingSetBytes: 6*gib, gpuName: "Test GPU", supportsMetal: true)
        require(constrained.maxWorkspaceMiB <= 4, "Low app headroom must not be rounded up into a larger budget")
        let unsupported = DevicePolicy.capabilities(physicalBytes: 8*gib, availableBytes: 4*gib,
            gpuWorkingSetBytes: 0, gpuName: "Unsupported GPU", supportsMetal: false)
        require(!unsupported.supportsMetal, "Unsupported GPU was promoted by RAM size")

        let pool = [1, 2, 3, 5, 6]
        require(DevicePolicy.selectTrainingIndices(pool, limit: 3) == [1, 3, 6], "Trajectory coverage changed")
        require(DevicePolicy.selectTrainingIndices(pool, limit: 99) == pool, "Selection duplicates/exceeds available views")
        require(DevicePolicy.selectTrainingIndices(pool, limit: 0).isEmpty, "Zero view budget must not select input")
        require(DevicePolicy.selectTrainingIndices([], limit: 10).isEmpty, "Empty input gained views")
        let size = DevicePolicy.resizedDimensions(width: 1600, height: 1200, longEdge: 320)
        require(size.width == 320 && size.height == 240, "Aspect ratio or size changed")
        let native = DevicePolicy.resizedDimensions(width: 321, height: 123, longEdge: 640)
        require(native.width == 321 && native.height == 123, "Prepared inputs must not be upsampled")
        let k = DevicePolicy.resizedIntrinsics([800, 1000, 319.5, 239.5], oldWidth: 640, oldHeight: 480,
            newWidth: 320, newHeight: 240)
        require(k == [400, 500, 159.5, 119.5], "Half-pixel principal-point mapping changed")

        require(DevicePolicy.pauseReason(thermal: "fair", batteryFraction: 0.8, batteryState: "unplugged",
            lowPowerMode: false, requireUnplugged: true) == nil, "Normal unplugged run unexpectedly pauses")
        require(DevicePolicy.pauseReason(thermal: "serious", batteryFraction: 0.8, batteryState: "unplugged",
            lowPowerMode: false, requireUnplugged: false) != nil, "Serious thermal pressure must pause")
        require(DevicePolicy.pauseReason(thermal: "nominal", batteryFraction: 0.15, batteryState: "unplugged",
            lowPowerMode: false, requireUnplugged: false) != nil, "Low battery must pause")
        require(DevicePolicy.pauseReason(thermal: "nominal", batteryFraction: 0.8, batteryState: "charging",
            lowPowerMode: false, requireUnplugged: true) != nil, "Charging battery test must not begin")
        print("PASS device policy: resource caps, low headroom, presets, held-out-safe subsets, calibration and pause rules")
    }
}
