import Foundation
import simd

/// Unrotated, tightly packed, encoded-sRGB RGBA8 camera pixels.
public struct CaptureColorImage: Sendable {
    public let width: Int
    public let height: Int
    public let rgba: [UInt8]

    public init(width: Int, height: Int, rgba: [UInt8]) throws {
        guard width > 0, height > 0, width <= 4096, height <= 4096,
              rgba.count == width * height * 4 else {
            throw HexelsError.invalid("Capture RGB must be tightly packed RGBA8, at most 4096 pixels per side")
        }
        self.width = width; self.height = height; self.rgba = rgba
    }

    fileprivate func color(at pixel: SIMD2<Float>) -> SIMD3<Float>? {
        guard pixel.x.isFinite, pixel.y.isFinite,
              pixel.x >= -0.5, pixel.y >= -0.5,
              pixel.x <= Float(width) - 0.5, pixel.y <= Float(height) - 0.5 else { return nil }
        let x = max(0, min(Float(width - 1), pixel.x)), y = max(0, min(Float(height - 1), pixel.y))
        let x0 = Int(floor(x)), y0 = Int(floor(y)), x1 = min(width - 1, x0 + 1), y1 = min(height - 1, y0 + 1)
        let tx = x - Float(x0), ty = y - Float(y0)
        func value(_ xx: Int, _ yy: Int) -> SIMD3<Float> {
            let j = (yy * width + xx) * 4
            return SIMD3(Float(rgba[j]), Float(rgba[j+1]), Float(rgba[j+2])) / 255
        }
        return (1 - ty) * ((1 - tx) * value(x0, y0) + tx * value(x1, y0))
            + ty * ((1 - tx) * value(x0, y1) + tx * value(x1, y1))
    }
}

/// Registered axial camera-Z depth in meters, with optional ARKit 0/1/2 confidence.
/// Zero, negative, or nonfinite depth pixels are missing observations, not geometry.
public struct CaptureDepthMap: Sendable {
    public let width: Int
    public let height: Int
    public let meters: [Float]
    public let confidence: [UInt8]?

    public init(width: Int, height: Int, meters: [Float], confidence: [UInt8]? = nil) throws {
        guard width > 0, height > 0, width <= 2048, height <= 2048,
              meters.count == width * height,
              confidence == nil || (confidence!.count == meters.count && confidence!.allSatisfy({ $0 <= 2 })) else {
            throw HexelsError.invalid("Capture depth/confidence dimensions or confidence codes differ")
        }
        self.width = width; self.height = height; self.meters = meters; self.confidence = confidence
    }
}

public struct CaptureView: Sendable {
    public let camera: HexelCamera
    public let color: CaptureColorImage
    public init(camera: HexelCamera, color: CaptureColorImage) throws {
        try camera.validate()
        guard camera.width == color.width, camera.height == color.height else {
            throw HexelsError.invalid("Capture RGB and calibrated camera dimensions differ")
        }
        self.camera = camera; self.color = color
    }
}

/// A single observed world-space surface sample; RGB remains encoded sRGB.
public struct CaptureSample: Sendable {
    public let position: SIMD3<Float>
    public let normal: SIMD3<Float>
    public let color: SIMD3<Float>
    public let radius: Float
    public init(position: SIMD3<Float>, normal: SIMD3<Float>, color: SIMD3<Float>, radius: Float) {
        self.position = position; self.normal = normal; self.color = color; self.radius = radius
    }
}

public enum CaptureGeometry {
    /// ARKit camera axes are right/up/back; CV axes are right/down/forward.
    /// Postmultiply the raw ARCamera transform by diag(1,-1,-1,1).
    public static func cvCameraToWorld(arCameraToWorld m: simd_float4x4) -> simd_float4x4 {
        simd_float4x4(columns: (m.columns.0, -m.columns.1, -m.columns.2, m.columns.3))
    }

    public static func rowMajor(_ matrix: simd_float4x4) -> [Float] {
        (0..<4).flatMap { row in (0..<4).map { column in matrix[column][row] } }
    }

    /// Integer pixel centers with half-pixel resize alignment; use actual rounded sizes.
    public static func resizedIntrinsics(_ k: SIMD4<Float>, from old: SIMD2<Int>, to new: SIMD2<Int>) throws -> SIMD4<Float> {
        guard old.x > 0, old.y > 0, new.x > 0, new.y > 0,
              old.x <= 8192, old.y <= 8192, new.x <= 8192, new.y <= 8192,
              k.x > 0, k.y > 0, [k.x,k.y,k.z,k.w].allSatisfy(\.isFinite) else {
            throw HexelsError.invalid("Invalid capture intrinsic resize")
        }
        let sx = Float(new.x) / Float(old.x), sy = Float(new.y) / Float(old.y)
        return SIMD4(k.x * sx, k.y * sy, (k.z + 0.5) * sx - 0.5, (k.w + 0.5) * sy - 0.5)
    }

    /// A novel camera in the same world frame: linear translation/K/near and
    /// shortest-arc quaternion rotation. A generated camera has no ground-truth image.
    public static func interpolatedCamera(_ a: HexelCamera, _ b: HexelCamera,
                                          fraction t: Float = 0.5) throws -> HexelCamera {
        try a.validate(); try b.validate()
        guard a.width == b.width, a.height == b.height, t.isFinite, t >= 0, t <= 1 else {
            throw HexelsError.invalid("Camera interpolation requires equal image dimensions and a fraction in [0,1]")
        }
        if t == 0 { return a }
        if t == 1 { return b }
        let qa = simd_quatf(simd_float3x3(columns:(a.right,a.down,a.forward))).normalized
        var qb = simd_quatf(simd_float3x3(columns:(b.right,b.down,b.forward))).normalized
        if simd_dot(qa.vector,qb.vector) < 0 { qb = simd_quatf(vector:-qb.vector) }
        let rotation = simd_float3x3(simd_slerp(qa,qb,t).normalized)
        let origin = (1-t)*a.origin+t*b.origin
        let matrix = simd_float4x4(columns:(SIMD4(rotation.columns.0,0),SIMD4(rotation.columns.1,0),
            SIMD4(rotation.columns.2,0),SIMD4(origin,1)))
        let intrinsics = zip(a.intrinsics,b.intrinsics).map { (1-t)*$0.0+t*$0.1 }
        let result = HexelCamera(width:a.width,height:a.height,intrinsics:intrinsics,
            cameraToWorld:rowMajor(matrix),near:(1-t)*a.near+t*b.near)
        try result.validate()
        return result
    }

    /// Axial depth unprojection. Image orientation must match the raw camera pose/K.
    /// Normals use adjacent valid depths, with a camera-facing prior at holes/edges.
    public static func depthSamples(depth: CaptureDepthMap, view: CaptureView, stride: Int = 4,
                                    minConfidence: UInt8 = 1, maxSamples: Int = 20_000) throws -> [CaptureSample] {
        guard stride > 0, stride <= 2048, minConfidence <= 2, maxSamples > 0, maxSamples <= 50_000 else {
            throw HexelsError.invalid("Invalid capture depth sampling budget")
        }
        let k = try resizedIntrinsics(SIMD4(view.camera.intrinsics[0], view.camera.intrinsics[1], view.camera.intrinsics[2], view.camera.intrinsics[3]),
            from: SIMD2(view.camera.width, view.camera.height), to: SIMD2(depth.width, depth.height))
        var sampling = stride
        while ((depth.width + sampling - 1) / sampling) * ((depth.height + sampling - 1) / sampling) > maxSamples { sampling += 1 }
        func point(_ x: Int, _ y: Int) -> SIMD3<Float>? {
            let j = y * depth.width + x, z = depth.meters[j]
            guard z.isFinite, z >= max(0.2, view.camera.near), z <= 8,
                  depth.confidence == nil || depth.confidence![j] >= minConfidence else { return nil }
            return SIMD3((Float(x) - k.z) * z / k.x, (Float(y) - k.w) * z / k.y, z)
        }
        let camera = view.camera
        func world(_ local: SIMD3<Float>) -> SIMD3<Float> {
            camera.origin + local.x * camera.right + local.y * camera.down + local.z * camera.forward
        }
        var result: [CaptureSample] = []
        result.reserveCapacity(min(maxSamples, depth.meters.count))
        for y in Swift.stride(from: 0, to: depth.height, by: sampling) {
            for x in Swift.stride(from: 0, to: depth.width, by: sampling) {
                guard let p = point(x, y) else { continue }
                let position = world(p), direction = camera.origin-position
                guard [position.x,position.y,position.z].allSatisfy(\.isFinite),
                      simd_length_squared(direction).isFinite, simd_length_squared(direction) > 1e-8 else { continue }
                let towardCamera = simd_normalize(direction)
                var normal = towardCamera
                let x0 = max(0, x-1), x1 = min(depth.width-1, x+1)
                let y0 = max(0, y-1), y1 = min(depth.height-1, y+1)
                if x0 != x1, y0 != y1, let l = point(x0,y), let r = point(x1,y),
                   let t = point(x,y0), let b = point(x,y1),
                   [l.z,r.z,t.z,b.z].allSatisfy({ abs($0-p.z) <= max(0.03, p.z*0.03) }) {
                    let cross = simd_cross(world(r)-world(l), world(b)-world(t))
                    if simd_length_squared(cross) > 1e-14 {
                        normal = simd_normalize(cross)
                        if simd_dot(normal,towardCamera) < 0 { normal = -normal }
                    }
                }
                let pixel = SIMD2((Float(x)+0.5)*Float(camera.width)/Float(depth.width)-0.5,
                                  (Float(y)+0.5)*Float(camera.height)/Float(depth.height)-0.5)
                guard let rgb = view.color.color(at: pixel) else { continue }
                let radius = max(0.001, min(0.15, 0.65*p.z*Float(sampling)/min(k.x,k.y)))
                result.append(CaptureSample(position: position, normal: normal, color: rgb, radius: radius))
            }
        }
        return result
    }

    /// Observed AR feature points only; normals are camera-facing priors, not measured surfaces.
    public static func sparseSamples(worldPoints: [SIMD3<Float>], view: CaptureView,
                                     maxSamples: Int = 20_000) throws -> [CaptureSample] {
        guard maxSamples > 0, maxSamples <= 50_000, worldPoints.count <= 1_000_000 else {
            throw HexelsError.resourceLimit("Capture sparse point budget exceeded")
        }
        let camera = view.camera, k = camera.intrinsics
        var result: [CaptureSample] = []
        result.reserveCapacity(min(maxSamples, worldPoints.count))
        let step = max(1, (worldPoints.count + maxSamples - 1)/maxSamples)
        for index in Swift.stride(from: 0, to: worldPoints.count, by: step) {
            let position = worldPoints[index]
            guard [position.x,position.y,position.z].allSatisfy(\.isFinite) else { continue }
            let p = camera.cameraPoint(position)
            guard p.z >= max(0.2,camera.near), p.z <= 8 else { continue }
            let pixel = SIMD2(k[0]*p.x/p.z+k[2], k[1]*p.y/p.z+k[3])
            guard let rgb = view.color.color(at: pixel) else { continue }
            result.append(CaptureSample(position:position, normal:simd_normalize(camera.origin-position), color:rgb,
                radius:max(0.003,min(0.1,6*p.z/min(k[0],k[1])))))
        }
        return result
    }
}

/// Bounded deterministic voxel aggregation. Call add only with training observations.
/// Depth measurements and sparse camera-facing priors must be recorded separately in provenance.
public struct CaptureSeedAccumulator: Sendable {
    private struct Key: Hashable, Sendable { let x,y,z: Int32 }
    private struct Cell: Sendable {
        var position, normal, color: SIMD3<Float>
        var radius: Float
        var count: Int
    }
    public let voxelSize: Float
    public let maxCells: Int
    public private(set) var acceptedSamples = 0
    public private(set) var droppedSamples = 0
    public var cellCount: Int { cells.count }
    private var cells: [Cell] = []
    private var lookup: [Key:Int] = [:]

    public init(voxelSize: Float = 0.01, maxCells: Int = 20_000) throws {
        guard voxelSize.isFinite, voxelSize >= 0.001, voxelSize <= 0.5, maxCells > 0, maxCells <= 50_000 else {
            throw HexelsError.invalid("Capture voxel size/cell budget is invalid")
        }
        self.voxelSize = voxelSize; self.maxCells = maxCells
    }

    public mutating func add(_ samples: [CaptureSample]) throws {
        guard samples.count <= 50_000 else { throw HexelsError.resourceLimit("Capture sample batch exceeds 50K") }
        for sample in samples {
            let p = sample.position, n = sample.normal, c = sample.color
            guard [p.x,p.y,p.z,n.x,n.y,n.z,c.x,c.y,c.z,sample.radius].allSatisfy(\.isFinite),
                  simd_reduce_max(abs(p)) <= 10_000, simd_length(n).isFinite, simd_length(n) > 1e-6,
                  simd_reduce_min(c) >= 0, simd_reduce_max(c) <= 1,
                  sample.radius > 0, sample.radius <= 1 else {
                throw HexelsError.invalid("Capture sample contains invalid world geometry, color, or radius")
            }
            let key = Key(x:Int32(floor(p.x/voxelSize)), y:Int32(floor(p.y/voxelSize)), z:Int32(floor(p.z/voxelSize)))
            let unit = simd_normalize(n)
            if let index = lookup[key] {
                guard cells[index].count < 4096 else { droppedSamples += 1; continue }
                cells[index].position += p; cells[index].color += c; cells[index].radius += sample.radius
                cells[index].normal += simd_dot(cells[index].normal,unit) < 0 ? -unit : unit
                cells[index].count += 1; acceptedSamples += 1
            } else if cells.count < maxCells {
                lookup[key] = cells.count
                cells.append(Cell(position:p,normal:unit,color:c,radius:sample.radius,count:1))
                acceptedSamples += 1
            } else { droppedSamples += 1 }
        }
    }

    /// Contiguous finite V1 Float32 rows: position/log-scales/WXYZ/RGB/gradU/gradV/logit-opacity/log-softness.
    public func parameters() throws -> [Float] {
        guard !cells.isEmpty else { throw HexelsError.invalid("Capture has no valid observed training seeds") }
        var output: [Float] = []; output.reserveCapacity(cells.count*20)
        for cell in cells {
            let count = Float(cell.count), p = cell.position/count, n = simd_normalize(cell.normal)
            let reference = abs(n.z) < 0.9 ? SIMD3<Float>(0,0,1) : SIMD3<Float>(0,1,0)
            let u = simd_normalize(simd_cross(reference,n)), v = simd_cross(n,u)
            let q = simd_normalize(simd_quatf(simd_float3x3(columns:(u,v,n))))
            let c = cell.color/count, scale = max(voxelSize*0.6,min(voxelSize*3,cell.radius/count))
            output += [p.x,p.y,p.z,log(scale),log(scale), q.real,q.imag.x,q.imag.y,q.imag.z,
                       c.x,c.y,c.z, 0,0,0, 0,0,0, 0,log(15)]
        }
        try SceneBundle.validateParameters(output)
        return output
    }
}
