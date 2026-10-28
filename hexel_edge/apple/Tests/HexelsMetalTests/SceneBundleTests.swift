import CoreGraphics
import Foundation
import ImageIO
import HexelsMetal

final class SceneBundleTests {
    private var directories: [URL] = []

    deinit {
        for directory in directories { try? FileManager.default.removeItem(at: directory) }
    }
    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        directories.append(url)
        return url
    }

    @discardableResult
    private func writeBundle(at root: URL, change: (inout [String: Any]) -> Void = { _ in }) throws -> URL {
        let images = root.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: images, withIntermediateDirectories: true)
        // Four distinct corners detect accidental image flips/transposes in decoding.
        let rgba: [UInt8] = [255, 0, 0, 255, 0, 255, 0, 255,
                             0, 0, 255, 255, 255, 255, 0, 255]
        let provider = try requireValue(CGDataProvider(data: Data(rgba) as CFData))
        let colorSpace = try requireValue(CGColorSpace(name: CGColorSpace.sRGB))
        let image = try requireValue(CGImage(width: 2, height: 2, bitsPerComponent: 8,
            bitsPerPixel: 32, bytesPerRow: 8, space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try requireValue(CGImageDestinationCreateWithURL(
            images.appendingPathComponent("00000.png") as CFURL, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        try expect(CGImageDestinationFinalize(destination))
        var parameters = Data()
        for number in IndependentReference.fixture().map(Float.init) {
            var value = number.bitPattern.littleEndian
            withUnsafeBytes(of: &value) { parameters.append(contentsOf: $0) }
        }
        try parameters.write(to: root.appendingPathComponent("parameters.f32"))
        try Data([0, 0, 0, 0]).write(to: root.appendingPathComponent("view_order.u32"))
        var manifest: [String: Any] = [
            "schema_version": 1, "variant": "v1", "parameter_layout": "hexels_v1_log_wxyz_20",
            "primitive_count": 1, "parameters_file": "parameters.f32",
            "view_order_file": "view_order.u32", "background": [0.0, 0.0, 0.0],
            "frames": [["name": "00000.png", "image": "images/00000.png", "width": 2, "height": 2,
                        "intrinsics": [2.0, 2.0, 0.5, 0.5],
                        "camera_to_world": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                        "split": "train"]]
        ]
        change(&manifest)
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: root.appendingPathComponent("manifest.json"))
        return root
    }

    func testBundleRoundTripAndCalibratedImageOrientation() throws {
        let root = try writeBundle(at: temporaryDirectory())
        let bundle = try SceneBundle(url: root)
        try expectEqual(bundle.initialParameters, IndependentReference.fixture().map(Float.init))
        try expectEqual(bundle.viewOrder, [0])
        try expectEqual(bundle.frames.count, 1)
        let pixels = try bundle.image(for: bundle.frames[0])
        try expectEqual(pixels, [SIMD4<Float>(1, 0, 0, 1), SIMD4<Float>(0, 1, 0, 1),
                                SIMD4<Float>(0, 0, 1, 1), SIMD4<Float>(1, 1, 0, 1)])
    }

    func testUnsupportedSchemaAndTruncatedParametersAreRejected() throws {
        let root = try writeBundle(at: temporaryDirectory()) { $0["schema_version"] = 2 }
        try expectThrows(try SceneBundle(url: root))
        try writeBundle(at: root)
        try Data([0, 0, 0, 0]).write(to: root.appendingPathComponent("parameters.f32"))
        try expectThrows(try SceneBundle(url: root))
    }

    func testEscapingAssetPathsAndSymlinksAreRejected() throws {
        let root = try writeBundle(at: temporaryDirectory()) { $0["parameters_file"] = "../outside.f32" }
        try expectThrows(try SceneBundle(url: root))
        try writeBundle(at: root) { $0["parameters_file"] = "/tmp/outside.f32" }
        try expectThrows(try SceneBundle(url: root))
        try writeBundle(at: root)
        let outside = try temporaryDirectory().appendingPathComponent("outside.f32")
        try FileManager.default.copyItem(at: root.appendingPathComponent("parameters.f32"), to: outside)
        try FileManager.default.removeItem(at: root.appendingPathComponent("parameters.f32"))
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("parameters.f32"),
                                                  withDestinationURL: outside)
        try expectThrows(try SceneBundle(url: root))
    }

    func testHeldOutAndAbsentViewOrderEntriesAreRejected() throws {
        let root = try writeBundle(at: temporaryDirectory()) { manifest in
            var frames = manifest["frames"] as! [[String: Any]]
            frames[0]["split"] = "test"
            manifest["frames"] = frames
        }
        try expectThrows(try SceneBundle(url: root))
        try writeBundle(at: root)
        try Data([1, 0, 0, 0]).write(to: root.appendingPathComponent("view_order.u32"))
        try expectThrows(try SceneBundle(url: root))
    }

    func testImageSizeMismatchIsRejectedBeforeTraining() throws {
        let root = try writeBundle(at: temporaryDirectory()) { manifest in
            var frames = manifest["frames"] as! [[String: Any]]
            frames[0]["width"] = 3
            manifest["frames"] = frames
        }
        let bundle = try SceneBundle(url: root)
        try expectThrows(try bundle.image(for: bundle.frames[0]))
    }

    func testNonRigidAndInvalidCalibrationAreRejected() throws {
        let root = try writeBundle(at: temporaryDirectory()) { manifest in
            var frames = manifest["frames"] as! [[String: Any]]
            frames[0]["camera_to_world"] = [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
            manifest["frames"] = frames
        }
        try expectThrows(try SceneBundle(url: root))
        try writeBundle(at: root) { manifest in
            var frames = manifest["frames"] as! [[String: Any]]
            frames[0]["intrinsics"] = [0, 2, 0.5, 0.5]
            manifest["frames"] = frames
        }
        try expectThrows(try SceneBundle(url: root))
    }

    func testRecipeHintsAreValidatedAndBoundToTheBundle() throws {
        let root = try writeBundle(at: temporaryDirectory())
        let legacy = try SceneBundle(url: root)
        try expectEqual(legacy.recipeHint, "standard")
        try writeBundle(at: root) { $0["recipe_hint"] = "detail" }
        let detailed = try SceneBundle(url: root)
        try expectEqual(detailed.recipeHint, "detail")
        try expectEqual(detailed.initialParameters, legacy.initialParameters)
        try expect(detailed.fingerprint != legacy.fingerprint, "Recipe changes must alter checkpoint identity")
        for invalid: Any in ["unsupported", 42] {
            try writeBundle(at: root) { $0["recipe_hint"] = invalid }
            try expectThrows(try SceneBundle(url: root))
        }
    }
}
