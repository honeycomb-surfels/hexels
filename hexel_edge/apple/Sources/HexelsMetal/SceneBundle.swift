import Foundation
import ImageIO
import CoreGraphics
import simd
import CryptoKit

public enum HexelsError: Error, LocalizedError {
    case invalid(String), unavailable(String), resourceLimit(String), gpu(String)
    public var errorDescription: String? {
        switch self {
        case .invalid(let message), .unavailable(let message), .resourceLimit(let message), .gpu(let message):
            return message
        }
    }
}

/// CV camera axes: right/down/forward. Matrices in the bundle are row-major.
public struct HexelCamera: Sendable {
    public let width: Int
    public let height: Int
    public let intrinsics: [Float]
    public let cameraToWorld: [Float]
    public let near: Float

    public init(width: Int, height: Int, intrinsics: [Float], cameraToWorld: [Float], near: Float = 0.2) {
        self.width = width; self.height = height; self.intrinsics = intrinsics
        self.cameraToWorld = cameraToWorld; self.near = near
    }

    func validate() throws {
        guard width > 0, height > 0, width <= 8192, height <= 8192,
              intrinsics.count == 4, intrinsics.allSatisfy(\.isFinite),
              intrinsics[0] > 0, intrinsics[1] > 0,
              cameraToWorld.count == 16, cameraToWorld.allSatisfy(\.isFinite), near.isFinite, near > 0 else {
            throw HexelsError.invalid("Invalid image dimensions, intrinsics, camera transform, or near plane")
        }
        let m = cameraToWorld
        guard abs(m[12]) < 1e-5, abs(m[13]) < 1e-5, abs(m[14]) < 1e-5, abs(m[15] - 1) < 1e-5 else {
            throw HexelsError.invalid("camera_to_world must be a homogeneous rigid transform")
        }
        let r = right, d = down, f = forward
        guard abs(simd_length(r) - 1) < 2e-3, abs(simd_length(d) - 1) < 2e-3,
              abs(simd_length(f) - 1) < 2e-3, abs(simd_dot(r, d)) < 2e-3,
              abs(simd_dot(r, f)) < 2e-3, abs(simd_dot(d, f)) < 2e-3,
              simd_dot(simd_cross(r, d), f) > 0.998 else {
            throw HexelsError.invalid("camera_to_world rotation must be right-handed and orthonormal")
        }
    }

    var origin: SIMD3<Float> { SIMD3(cameraToWorld[3], cameraToWorld[7], cameraToWorld[11]) }
    var right: SIMD3<Float> { SIMD3(cameraToWorld[0], cameraToWorld[4], cameraToWorld[8]) }
    var down: SIMD3<Float> { SIMD3(cameraToWorld[1], cameraToWorld[5], cameraToWorld[9]) }
    var forward: SIMD3<Float> { SIMD3(cameraToWorld[2], cameraToWorld[6], cameraToWorld[10]) }
    func cameraPoint(_ point: SIMD3<Float>) -> SIMD3<Float> {
        let delta = point - origin
        return SIMD3(simd_dot(delta, right), simd_dot(delta, down), simd_dot(delta, forward))
    }
}

public struct SceneFrame: Decodable, Sendable {
    public let name: String
    public let image: String
    public let width: Int
    public let height: Int
    public let intrinsics: [Float]
    public let cameraToWorld: [Float]
    public let split: String
    public var camera: HexelCamera {
        HexelCamera(width: width, height: height, intrinsics: intrinsics, cameraToWorld: cameraToWorld)
    }
    enum CodingKeys: String, CodingKey {
        case name, image, width, height, intrinsics, split
        case cameraToWorld = "camera_to_world"
    }
}

private struct BundleManifest: Decodable {
    let schemaVersion: Int
    let variant: String
    let parameterLayout: String
    let primitiveCount: Int
    let parametersFile: String
    let viewOrderFile: String
    let background: [Float]
    let frames: [SceneFrame]
    let recipeHint: String?
    enum CodingKeys: String, CodingKey {
        case variant, background, frames
        case schemaVersion = "schema_version", parameterLayout = "parameter_layout"
        case primitiveCount = "primitive_count", parametersFile = "parameters_file", viewOrderFile = "view_order_file"
        case recipeHint = "recipe_hint"
    }
}

/// Validated local input bundle. Images are decoded one at a time, never preloaded.
public struct SceneBundle {
    public let url: URL
    public let frames: [SceneFrame]
    public let initialParameters: [Float]
    public let viewOrder: [UInt32]
    public let background: SIMD3<Float>
    public let fingerprint: String
    public let recipeHint: String

    public init(url: URL) throws {
        let root = url.hasDirectoryPath || url.pathExtension != "json" ? url : url.deletingLastPathComponent()
        self.url = root.standardizedFileURL.resolvingSymlinksInPath()
        let manifestURL = try Self.localFile("manifest.json",root:self.url)
        try Self.checkFileSize(manifestURL, maximum:16*1024*1024)
        let metadata = try Data(contentsOf: manifestURL)
        guard metadata.count <= 16 * 1024 * 1024 else { throw HexelsError.resourceLimit("Manifest exceeds 16 MiB") }
        let manifest = try JSONDecoder().decode(BundleManifest.self, from: metadata)
        let hint = manifest.recipeHint ?? "standard"
        guard hint == "standard" || hint == "detail" else { throw HexelsError.invalid("Unsupported training recipe hint") }
        recipeHint = hint
        guard manifest.schemaVersion == 1, manifest.variant == "v1",
              manifest.parameterLayout == "hexels_v1_log_wxyz_20",
              manifest.primitiveCount > 0, manifest.primitiveCount <= 1_000_000,
              manifest.background.count == 3, manifest.background.allSatisfy(\.isFinite),
              !manifest.frames.isEmpty, manifest.frames.count <= 2048 else {
            throw HexelsError.invalid("Unsupported or invalid Hexels bundle schema/budgets")
        }
        frames = manifest.frames
        background = SIMD3(manifest.background[0], manifest.background[1], manifest.background[2])
        guard Set(frames.map(\.name)).count == frames.count else { throw HexelsError.invalid("Duplicate camera names") }
        for frame in frames {
            try frame.camera.validate()
            guard frame.split == "train" || frame.split == "test" else { throw HexelsError.invalid("Unknown camera split") }
            _ = try Self.localFile(frame.image, root: self.url)
        }
        let parameterURL = try Self.localFile(manifest.parametersFile, root: self.url)
        initialParameters = try Self.readFloats(parameterURL, count: manifest.primitiveCount * 20)
        let orderURL = try Self.localFile(manifest.viewOrderFile, root: self.url)
        try Self.checkFileSize(orderURL,maximum:4*1_000_000)
        let orderData = try Data(contentsOf: orderURL)
        guard !orderData.isEmpty, orderData.count % 4 == 0, orderData.count <= 4 * 1_000_000 else {
            throw HexelsError.invalid("Invalid or oversized view order")
        }
        viewOrder = stride(from: 0, to: orderData.count, by: 4).map { offset in
            orderData.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
        }
        guard viewOrder.allSatisfy({ Int($0) < manifest.frames.count && manifest.frames[Int($0)].split == "train" }) else {
            throw HexelsError.invalid("View order references an absent or held-out frame")
        }
        try Self.validateParameters(initialParameters)
        var digest = SHA256()
        digest.update(data:metadata)
        let assetRoot = self.url
        for file in [parameterURL, orderURL] + (try manifest.frames.map { try Self.localFile($0.image,root:assetRoot) }) {
            let input = try FileHandle(forReadingFrom:file); defer { try? input.close() }
            while let block = try input.read(upToCount:65536), !block.isEmpty { digest.update(data:block) }
        }
        fingerprint = digest.finalize().map { String(format:"%02x",$0) }.joined()
    }

    static func localFile(_ path: String, root: URL) throws -> URL {
        let candidate = root.appendingPathComponent(path).standardizedFileURL.resolvingSymlinksInPath()
        guard !path.isEmpty, !path.hasPrefix("/"), candidate.path.hasPrefix(root.path + "/") else {
            throw HexelsError.invalid("Bundle asset must remain within the local bundle: \(path)")
        }
        return candidate
    }

    static func readFloats(_ url: URL, count: Int) throws -> [Float] {
        try checkFileSize(url,maximum:count*4,exact:count*4)
        let data = try Data(contentsOf: url)
        guard data.count == count * 4 else { throw HexelsError.invalid("Unexpected Float32 file size: \(url.lastPathComponent)") }
        return stride(from: 0, to: data.count, by: 4).map { offset in
            data.withUnsafeBytes { Float(bitPattern: UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self))) }
        }
    }

    /// Admit file-backed allocations before Data loads their bytes into memory.
    static func checkFileSize(_ url: URL, maximum: Int, exact: Int? = nil) throws {
        let values = try url.resourceValues(forKeys:[.fileSizeKey,.isRegularFileKey])
        guard values.isRegularFile == true, let size = values.fileSize,
              size > 0, size <= maximum, exact == nil || size == exact else {
            throw HexelsError.resourceLimit("Unexpected or oversized bundle file: \(url.lastPathComponent)")
        }
    }

    static func validateParameters(_ parameters: [Float]) throws {
        guard !parameters.isEmpty, parameters.count % 20 == 0, parameters.allSatisfy(\.isFinite) else {
            throw HexelsError.invalid("Parameters must be finite contiguous V1 rows of 20 Float32 values")
        }
        for i in stride(from: 0, to: parameters.count, by: 20) {
            let q = SIMD4(parameters[i+5], parameters[i+6], parameters[i+7], parameters[i+8])
            guard simd_length(q).isFinite, simd_length(q) > 1e-8, abs(parameters[i+3]) < 40, abs(parameters[i+4]) < 40,
                  parameters[i+19] > -40, parameters[i+19] < 40 else {
                throw HexelsError.invalid("Invalid quaternion or unrepresentable log scale/softness")
            }
        }
    }

    public func image(for frame: SceneFrame, maxDecodedBytes: Int = 64 * 1024 * 1024) throws -> [SIMD4<Float>] {
        // Both the temporary RGBA8 decode and returned Float32 RGBA array are
        // live together. Admit before asking ImageIO to decompress the source.
        guard frame.width > 0, frame.height > 0, frame.width <= 8192, frame.height <= 8192,
              maxDecodedBytes > 0, frame.width * frame.height <= maxDecodedBytes / 20 else {
            throw HexelsError.resourceLimit("Decoded image exceeds its memory admission limit")
        }
        let file = try Self.localFile(frame.image, root: url)
        guard let source = CGImageSourceCreateWithURL(file as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              (properties[kCGImagePropertyPixelWidth] as? Int) == frame.width,
              (properties[kCGImagePropertyPixelHeight] as? Int) == frame.height,
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw HexelsError.invalid("Cannot decode calibrated image dimensions: \(frame.image)")
        }
        var bytes = [UInt8](repeating: 0, count: frame.width * frame.height * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let success = bytes.withUnsafeMutableBytes { storage -> Bool in
            guard let context = CGContext(data: storage.baseAddress, width: frame.width, height: frame.height,
                bitsPerComponent: 8, bytesPerRow: frame.width * 4, space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: frame.width, height: frame.height))
            return true
        }
        guard success else { throw HexelsError.invalid("Cannot create image decode context") }
        return stride(from: 0, to: bytes.count, by: 4).map { i in
            let a = Float(bytes[i+3]) / 255
            return SIMD4(Float(bytes[i])/255 + (1-a)*background.x,
                         Float(bytes[i+1])/255 + (1-a)*background.y,
                         Float(bytes[i+2])/255 + (1-a)*background.z, 1)
        }
    }
}
