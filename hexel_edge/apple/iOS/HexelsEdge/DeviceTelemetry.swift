import Darwin
import Foundation
import Metal
import UIKit

struct DeviceTelemetry: Codable, Sendable {
    let timestamp: Date
    let batteryFraction: Double?
    let batteryState: String
    let thermalState: String
    let lowPowerMode: Bool
    let footprintBytes: UInt64
    let gpuBytes: Int
    let stepMilliseconds: Double?
    let elapsedSeconds: Double
    let availableMemoryBytes: UInt64
    var isCharging: Bool { batteryState == "charging" || batteryState == "full" }
}

enum PlatformDeviceState {
    @MainActor static func capabilities() -> DeviceCapabilities {
        let device = MTLCreateSystemDefaultDevice()
        return DevicePolicy.capabilities(physicalBytes: ProcessInfo.processInfo.physicalMemory,
            availableBytes: UInt64(os_proc_available_memory()),
            gpuWorkingSetBytes: device?.recommendedMaxWorkingSetSize ?? 0,
            gpuName: device?.name ?? "Metal unavailable", supportsMetal: device?.supportsFamily(.apple7) ?? false)
    }

    @MainActor static func capture(gpuBytes: Int, stepMilliseconds: Double?, elapsedSeconds: Double,
                                  previous: DeviceTelemetry? = nil, includeMemory: Bool = true) -> DeviceTelemetry {
        UIDevice.current.isBatteryMonitoringEnabled = true
        let level = UIDevice.current.batteryLevel
        let state: String
        switch UIDevice.current.batteryState {
        case .unplugged: state = "unplugged"
        case .charging: state = "charging"
        case .full: state = "full"
        default: state = "unknown"
        }
        let thermal: String
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: thermal = "nominal"
        case .fair: thermal = "fair"
        case .serious: thermal = "serious"
        case .critical: thermal = "critical"
        @unknown default: thermal = "unknown"
        }
        var footprint = previous?.footprintBytes ?? 0
        if includeMemory {
            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
            let status = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
                }
            }
            if status == KERN_SUCCESS { footprint = UInt64(info.phys_footprint) }
        }
        return DeviceTelemetry(timestamp: Date(), batteryFraction: level >= 0 ? Double(level) : nil,
            batteryState: state, thermalState: thermal, lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            footprintBytes: footprint,
            gpuBytes: gpuBytes, stepMilliseconds: stepMilliseconds, elapsedSeconds: elapsedSeconds,
            availableMemoryBytes: includeMemory ? UInt64(os_proc_available_memory()) : (previous?.availableMemoryBytes ?? 0))
    }
}
