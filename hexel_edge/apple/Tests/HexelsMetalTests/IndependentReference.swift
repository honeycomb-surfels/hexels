import Foundation
import simd

/// Double-precision test oracle. It intentionally shares no production render code.
struct ReferenceCamera {
    var width: Int = 13
    var height: Int = 11
    var fx: Double = 10
    var fy: Double = 11
    var cx: Double = 6
    var cy: Double = 5
    var origin = SIMD3<Double>(0.03, -0.02, 0)
    var right = SIMD3<Double>(1, 0, 0)
    var down = SIMD3<Double>(0, 1, 0)
    var forward = SIMD3<Double>(0, 0, 1)
    var near: Double = 0.2

    var intrinsics: [Float] {
        [Float(fx), Float(fy), Float(cx), Float(cy)]
    }

    var cameraToWorld: [Float] {
        [Float(right.x), Float(down.x), Float(forward.x), Float(origin.x),
         Float(right.y), Float(down.y), Float(forward.y), Float(origin.y),
         Float(right.z), Float(down.z), Float(forward.z), Float(origin.z), 0, 0, 0, 1]
    }
}

enum IndependentReference {
    static let background = SIMD3<Double>(0.07, 0.11, 0.17)

    static func fixture(z: Double = 2.6) -> [Double] {
        [0.08, -0.05, z, log(0.55), log(0.47), 0.92, 0.13, -0.19, 0.07,
         0.4, 0.3, 0.2, 0.06, -0.03, 0.02, 0.02, 0.05, -0.04, 0.9, log(7)]
    }

    static func sigmoid(_ value: Double) -> Double {
        if value >= 0 { return 1 / (1 + exp(-value)) }
        let e = exp(value)
        return e / (1 + e)
    }

    /// The quaternion is raw WXYZ, normalized before evaluating rotation columns.
    private static func axes(_ p: ArraySlice<Double>) -> (SIMD3<Double>, SIMD3<Double>, SIMD3<Double>) {
        let a = Array(p)
        let norm = sqrt(a.reduce(0) { $0 + $1 * $1 })
        let w = a[0] / norm, x = a[1] / norm, y = a[2] / norm, z = a[3] / norm
        let u = SIMD3<Double>(1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
        let v = SIMD3<Double>(2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))
        let n = SIMD3<Double>(2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
        return (u, v, n)
    }

    /// Sorted-center-depth alpha compositing, matching the selected V1 contract.
    static func render(
        parameters: [Double], camera: ReferenceCamera,
        background: SIMD3<Double> = background
    ) -> [SIMD4<Double>] {
        let count = parameters.count / 20
        func centerDepth(_ primitive: Int) -> Double {
            let offset = primitive * 20
            return simd_dot(SIMD3(parameters[offset], parameters[offset + 1], parameters[offset + 2])
                            - camera.origin, camera.forward)
        }
        let order = (0..<count).sorted {
            let left = centerDepth($0), right = centerDepth($1)
            return left == right ? $0 < $1 : left < right
        }
        return (0..<(camera.width * camera.height)).map { pixel in
            let x = Double(pixel % camera.width), y = Double(pixel / camera.width)
            let ray = camera.right * ((x - camera.cx) / camera.fx)
                + camera.down * ((y - camera.cy) / camera.fy) + camera.forward
            var transmission = 1.0
            var rgb = SIMD3<Double>.zero
            for index in order {
                let p = Array(parameters[(index * 20)..<((index + 1) * 20)])
                let center = SIMD3<Double>(p[0], p[1], p[2])
                let (axisU, axisV, normal) = axes(p[5...8])
                let denominator = simd_dot(normal, ray)
                if denominator == 0 { continue }
                let depth = simd_dot(normal, center - camera.origin) / denominator
                if depth < camera.near { continue }
                let offset = camera.origin + depth * ray - center
                let u = simd_dot(offset, axisU) * exp(-p[3])
                let v = simd_dot(offset, axisV) * exp(-p[4])
                let distance = max(abs(v), abs(sqrt(3) * 0.5 * u + 0.5 * v), abs(sqrt(3) * 0.5 * u - 0.5 * v)) - 1
                let softness = max(exp(p[19]), 1e-4)
                let alpha = min(0.99, sigmoid(p[18]) * sigmoid(-softness * distance))
                if alpha < 1 / 255.0 { continue }
                let candidate = transmission * (1 - alpha)
                if candidate < 1e-4 { break }
                let color = SIMD3<Double>(p[9], p[10], p[11])
                    + u * SIMD3<Double>(p[12], p[13], p[14])
                    + v * SIMD3<Double>(p[15], p[16], p[17])
                rgb += transmission * alpha * color
                transmission = candidate
            }
            rgb += transmission * background
            return SIMD4<Double>(rgb.x, rgb.y, rgb.z, 1 - transmission)
        }
    }

    static func target(camera: ReferenceCamera) -> [SIMD4<Double>] {
        (0..<(camera.width * camera.height)).map { pixel in
            let x = Double(pixel % camera.width), y = Double(pixel / camera.width)
            return SIMD4<Double>(0.2 + 0.017 * x, 0.15 + 0.011 * y,
                                 0.3 + 0.004 * x - 0.007 * y, 1)
        }
    }

    static func loss(_ rendered: [SIMD4<Double>], target: [SIMD4<Double>]) -> Double {
        zip(rendered, target).reduce(0) { sum, pair in
            let delta = pair.0 - pair.1
            return sum + delta.x * delta.x + delta.y * delta.y + delta.z * delta.z
        } / Double(rendered.count * 3)
    }

    static func finiteDifference(
        parameters: [Double], parameter: Int, camera: ReferenceCamera,
        target: [SIMD4<Double>], epsilon: Double = 1e-5
    ) -> Double {
        var plus = parameters, minus = parameters
        plus[parameter] += epsilon
        minus[parameter] -= epsilon
        return (loss(render(parameters: plus, camera: camera), target: target)
                - loss(render(parameters: minus, camera: camera), target: target)) / (2 * epsilon)
    }
}
