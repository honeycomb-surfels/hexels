import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

struct QualityEvaluation: Codable, Sendable {
    var step: Int
    var elapsed_seconds: Double
    var pooled_rgb_mse: Double
    var psnr_db: Double
    var mean_view_psnr_db: Double
    var per_view_psnr_db: [Double]
    var evaluation_long_edge: Int
}

struct BenchmarkReport: Codable, Sendable {
    var schema_version = 1
    var run_id: String
    var status = "running"
    var stop_reason: String?
    var started_at = Date()
    var ended_at: Date?
    var initial_step: Int
    var target_step: Int
    var last_step: Int
    var warmup_steps = 0
    var training_seconds = 0.0
    var wall_seconds = 0.0
    var steps_per_second: Double?
    var selected_training_frames: [String]
    var held_out_frames: [String]
    var image_long_edge: Int
    var evaluation_long_edge: Int
    var primitive_count: Int
    var battery_measurement_valid = true
    var battery_invalid_reasons: [String] = []
    var battery_start_fraction: Double?
    var battery_end_fraction: Double?
    var battery_drop_percentage_points: Double?
    var best_psnr_db: Double?
    var best_step: Int?
    var evaluations: [QualityEvaluation] = []
    var latest_telemetry: DeviceTelemetry?
    var peak_footprint_bytes: UInt64 = 0
    var peak_gpu_bytes = 0
    var settings: SceneSettings
    var bundle_fingerprint: String
    var checkpoint_context: String
    var battery_note = "Battery level is coarse OS telemetry, not energy or power. Charging intervals invalidate drain comparisons. Thermal state is categorical, not a temperature."

    mutating func observe(_ sample: DeviceTelemetry, step: Int) {
        latest_telemetry = sample
        last_step = step
        wall_seconds = Date().timeIntervalSince(started_at)
        peak_footprint_bytes = max(peak_footprint_bytes, sample.footprintBytes)
        peak_gpu_bytes = max(peak_gpu_bytes, sample.gpuBytes)
        if battery_start_fraction == nil { battery_start_fraction = sample.batteryFraction }
        battery_end_fraction = sample.batteryFraction
        if sample.batteryFraction == nil { invalidateBattery("Battery level unavailable") }
        if sample.batteryState != "unplugged" { invalidateBattery("External power or unknown charging state observed") }
        if battery_measurement_valid, let start = battery_start_fraction, let end = battery_end_fraction {
            battery_drop_percentage_points = (start - end) * 100
        }
        let completed = last_step - initial_step
        steps_per_second = training_seconds > 0 ? Double(completed) / training_seconds : nil
    }

    mutating func invalidateBattery(_ reason: String) {
        battery_measurement_valid = false
        battery_drop_percentage_points = nil
        if !battery_invalid_reasons.contains(reason) { battery_invalid_reasons.append(reason) }
    }

    func persist(in directory: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: directory.appendingPathComponent("report.json"), options: .atomic)
    }

    static func appendCSV(_ sample: DeviceTelemetry, step: Int, in directory: URL) throws {
        let url = directory.appendingPathComponent("telemetry.csv")
        if !FileManager.default.fileExists(atPath: url.path) {
            try Data("timestamp,elapsed_seconds,step,thermal_state,battery_fraction,battery_state,low_power_mode,footprint_bytes,gpu_bytes,available_memory_bytes,step_seconds\n".utf8).write(to: url)
        }
        let row = [ISO8601DateFormatter().string(from: sample.timestamp), String(sample.elapsedSeconds), String(step),
                   sample.thermalState, sample.batteryFraction.map { String($0) } ?? "", sample.batteryState,
                   String(sample.lowPowerMode), String(sample.footprintBytes), String(sample.gpuBytes),
                   String(sample.availableMemoryBytes), sample.stepMilliseconds.map { String($0 / 1000) } ?? ""]
            .joined(separator: ",") + "\n"
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(row.utf8))
    }
}

enum BenchmarkImages {
    static func writePNG(_ pixels: [SIMD4<Float>], width: Int, height: Int, to url: URL) throws {
        var bytes = [UInt8](); bytes.reserveCapacity(pixels.count * 4)
        for pixel in pixels {
            for channel in 0..<3 {
                let value = pixel[channel].isFinite ? min(1, max(0, pixel[channel])) : 0
                bytes.append(UInt8((value * 255).rounded()))
            }
            bytes.append(255)
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData), let space = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4, space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
    }
}
