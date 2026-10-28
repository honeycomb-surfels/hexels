import Foundation
import Metal
import simd

public struct TrainingConfiguration: Sendable {
    public static let defaultRates: [Float] = [
        0.0001,0.0001,0.0001, 0.001,0.001, 0.0005,0.0005,0.0005,0.0005,
        0.0025,0.0025,0.0025, 0.001,0.001,0.001,0.001,0.001,0.001, 0.01,0.001,
    ]
    /// Appearance-focused candidate for geometrically checked initial seeds.
    /// Explicit opt-in; default rates and old checkpoint behavior are unchanged.
    public static let detailRates: [Float] = [
        0.000025,0.000025,0.000025, 0.001,0.001, 0.0001,0.0001,0.0001,0.0001,
        0.01,0.01,0.01, 0.002,0.002,0.002,0.002,0.002,0.002, 0.02,0.0005,
    ]
    public var learningRates: [Float]
    public var maxIntersections: Int
    public var maxWorkingSetBytes: Int
    public init(learningRates: [Float] = Self.defaultRates, maxIntersections: Int = 2_000_000,
                maxWorkingSetBytes: Int = 512 * 1024 * 1024) {
        self.learningRates = learningRates; self.maxIntersections = maxIntersections
        self.maxWorkingSetBytes = maxWorkingSetBytes
    }
}

private struct SceneUniforms {
    var origin, right, down, forward, intrinsics, backgroundNear: SIMD4<Float>
    var dimensions: SIMD4<UInt32>
}

private struct AdamUniforms {
    var parameterCount, step: UInt32
    var beta1, beta2, epsilon, beta1Power, beta2Power, padding: Float
}

private struct CheckpointHeader: Codable {
    var schemaVersion: Int
    var parameterCount: Int
    var step: Int
    var learningRates: [Float]
    var context: String?
}

/// Serialized compute engine; callers must not render/update it concurrently.
/// All geometry/appearance gradients are analytic GPU kernels. Binning/sorting
/// are conservatively bounded on the CPU in this first portable implementation.
public final class MetalHexelTrainer {
    public let deviceName: String
    public private(set) var stepCount = 0
    public private(set) var lastIntersectionCount = 0
    public private(set) var peakAllocatedGPUBytes = 0
    public private(set) var lastForwardGPUSeconds: Double = 0
    public private(set) var lastBackwardGPUSeconds: Double = 0
    public var checkpointContext: String?
    public var learningRates: [Float] { configuration.learningRates }
    public var allocatedGPUBytes: Int { device.currentAllocatedSize }
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let renderPipeline, backwardPipeline, adamPipeline: MTLComputePipelineState
    private let parameterBuffer, gradientBuffer, momentBuffer, varianceBuffer, rateBuffer: MTLBuffer
    private let configuration: TrainingConfiguration
    private let parameterCount: Int
    public var parameters: [Float] { floats(parameterBuffer, count: parameterCount) }

    public init(parameters: [Float], configuration: TrainingConfiguration = TrainingConfiguration()) throws {
        try SceneBundle.validateParameters(parameters)
        guard configuration.learningRates.count == 20,
              configuration.learningRates.allSatisfy({ $0.isFinite && $0 >= 0 }),
              configuration.maxIntersections > 0, configuration.maxIntersections <= 100_000_000,
              configuration.maxWorkingSetBytes > 0, parameters.count <= 20_000_000 else {
            throw HexelsError.invalid("Invalid learning rates or resource ceilings")
        }
        guard parameters.count * 16 < configuration.maxWorkingSetBytes else {
            throw HexelsError.resourceLimit("Parameter, gradient, and Adam state exceed the memory ceiling")
        }
        guard let device = MTLCreateSystemDefaultDevice(), device.supportsFamily(.apple7),
              let queue = device.makeCommandQueue() else {
            throw HexelsError.unavailable("An Apple7-or-later Metal GPU is required; physical GPU access may require leaving the sandbox")
        }
        self.device = device; self.queue = queue; self.deviceName = device.name
        self.configuration = configuration; parameterCount = parameters.count
        guard let sourceURL = Bundle.module.url(forResource: "Hexels", withExtension: "metal") else {
            throw HexelsError.unavailable("Missing bundled Metal source")
        }
        let source = try String(contentsOf: sourceURL, encoding: .utf8)
        let options = MTLCompileOptions(); options.languageVersion = .version3_0; options.fastMathEnabled = false
        let library = try device.makeLibrary(source: source, options: options)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw HexelsError.gpu("Missing kernel \(name)") }
            return try device.makeComputePipelineState(function: function)
        }
        renderPipeline = try pipeline("renderHexels")
        backwardPipeline = try pipeline("backwardHexels")
        adamPipeline = try pipeline("adamUpdate")
        func buffer(_ values: [Float]) throws -> MTLBuffer {
            guard let result = device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared) else {
                throw HexelsError.resourceLimit("Metal parameter allocation failed")
            }
            return result
        }
        parameterBuffer = try buffer(parameters)
        let zero = [Float](repeating: 0, count: parameters.count)
        gradientBuffer = try buffer(zero); momentBuffer = try buffer(zero); varianceBuffer = try buffer(zero)
        rateBuffer = try buffer(configuration.learningRates)
    }

    private func floats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }

    private func makeBuffer<T>(_ values: [T]) throws -> MTLBuffer {
        let result = values.withUnsafeBytes { raw in
            raw.baseAddress.flatMap { device.makeBuffer(bytes:$0,length:raw.count,options:.storageModeShared) }
        }
        guard !values.isEmpty, let result else {
            throw HexelsError.resourceLimit("Metal staging allocation failed")
        }
        return result
    }

    @discardableResult private func run(_ encoderBody: (MTLComputeCommandEncoder) -> Void) throws -> Double {
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw HexelsError.gpu("Cannot create a compute command")
        }
        encoderBody(encoder); encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        if let error = command.error { throw HexelsError.gpu(error.localizedDescription) }
        return max(0, command.gpuEndTime-command.gpuStartTime)
    }

    /// Six projected vertices enclose the full alpha-cutoff support, including
    /// soft boundaries. Near-plane crossings use the full viewport conservatively.
    private func bins(camera: HexelCamera, maxEntries: Int) throws -> ([UInt32], [UInt32]) {
        let p = parameters, tileSize = 16
        let tilesX = (camera.width + 15) / 16, tilesY = (camera.height + 15) / 16
        var lists = [[UInt32]](repeating: [], count: tilesX * tilesY)
        let count = p.count / 20
        let ordered = (0..<count).sorted { a, b in
            let aa = camera.cameraPoint(SIMD3(p[a*20], p[a*20+1], p[a*20+2])).z
            let bb = camera.cameraPoint(SIMD3(p[b*20], p[b*20+1], p[b*20+2])).z
            return aa == bb ? a < b : aa < bb
        }
        var intersections = 0
        for id in ordered {
            let j = id * 20, opacity: Float
            if p[j+18] >= 0 { opacity = 1 / (1 + exp(-p[j+18])) }
            else { let e = exp(p[j+18]); opacity = e / (1+e) }
            // Include equality and a roundoff margin: GPU sigmoid can round a
            // sharp interior boundary to 1. Never cull alpha-cutoff pixels due
            // to CPU/GPU exponential rounding differences.
            let upperOpacity = min(1,opacity * (1 + 8 * Float.ulpOfOne))
            guard upperOpacity >= 1/255 else { continue }
            let softness = max(exp(p[j+19]), 0.0001)
            let argument = 255 * upperOpacity - 1
            let rawBound = argument > 0 ? 1 + log(argument) / softness : 0
            let bound = rawBound + 4e-5 * max(1,abs(rawBound))
            guard bound.isFinite, bound > 0 else { continue }
            let position = SIMD3(p[j], p[j+1], p[j+2])
            let q = simd_normalize(SIMD4(p[j+5],p[j+6],p[j+7],p[j+8]))
            let w=q.x, x=q.y, y=q.z, z=q.w
            let u = SIMD3(1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)) * exp(p[j+3])
            let v = SIMD3(2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)) * exp(p[j+4])
            let a = bound / sqrt(Float(3))
            let vertices: [SIMD2<Float>] = [SIMD2(2*a,0),SIMD2(a,bound),SIMD2(-a,bound),
                SIMD2(-2*a,0),SIMD2(-a,-bound),SIMD2(a,-bound)]
            var minX = Float.infinity, minY = Float.infinity, maxX = -Float.infinity, maxY = -Float.infinity
            var minZ = Float.infinity, maxZ = -Float.infinity
            for vertex in vertices {
                let cp = camera.cameraPoint(position + vertex.x*u + vertex.y*v)
                minZ = min(minZ, cp.z); maxZ = max(maxZ, cp.z)
                if cp.z >= camera.near {
                    let px = camera.intrinsics[0]*cp.x/cp.z + camera.intrinsics[2]
                    let py = camera.intrinsics[1]*cp.y/cp.z + camera.intrinsics[3]
                    minX = min(minX,px); maxX = max(maxX,px); minY = min(minY,py); maxY = max(maxY,py)
                }
            }
            guard maxZ >= camera.near else { continue }
            if minZ <= camera.near {
                minX=0; minY=0; maxX=Float(camera.width-1); maxY=Float(camera.height-1)
            }
            guard minX.isFinite, minY.isFinite, maxX.isFinite, maxY.isFinite else {
                throw HexelsError.invalid("Nonfinite projected support")
            }
            guard maxX >= -1, maxY >= -1, minX <= Float(camera.width), minY <= Float(camera.height) else { continue }
            let x0 = Int(max(0,min(Float(camera.width-1),floor(minX-1)))) / tileSize
            let y0 = Int(max(0,min(Float(camera.height-1),floor(minY-1)))) / tileSize
            let x1 = Int(max(0,min(Float(camera.width-1),ceil(maxX+1)))) / tileSize
            let y1 = Int(max(0,min(Float(camera.height-1),ceil(maxY+1)))) / tileSize
            let added = (x1-x0+1)*(y1-y0+1)
            guard intersections <= maxEntries - added else {
                throw HexelsError.resourceLimit("Primitive/tile intersections exceed admitted ceiling \(maxEntries); reduce the workload, no contributions were truncated")
            }
            intersections += added
            for ty in y0...y1 { for tx in x0...x1 { lists[ty*tilesX+tx].append(UInt32(id)) } }
        }
        var offsets: [UInt32] = [0], ids: [UInt32] = []
        ids.reserveCapacity(max(intersections,1)); offsets.reserveCapacity(lists.count+1)
        for list in lists { ids.append(contentsOf:list); offsets.append(UInt32(ids.count)) }
        lastIntersectionCount = intersections
        if ids.isEmpty { ids = [0] } // offsets stay zero; Metal does not permit a zero-byte buffer.
        return (ids, offsets)
    }

    private func inputs(camera: HexelCamera, background: SIMD3<Float>) throws -> (MTLBuffer, MTLBuffer, SceneUniforms) {
        try camera.validate()
        guard background.x.isFinite, background.y.isFinite, background.z.isFinite else { throw HexelsError.invalid("Nonfinite background") }
        try SceneBundle.validateParameters(parameters)
        // Account for host staging + output/target/gradient image arrays and sort
        // staging as well as persistent GPU state; this is a conservative budget,
        // not a replacement for whole-process resident-memory measurements.
        let baseBytes = parameterCount*24 + camera.width*camera.height*80
        guard baseBytes < configuration.maxWorkingSetBytes else { throw HexelsError.resourceLimit("Image and model working set exceeds configured ceiling") }
        // Reserve conservatively for growing Swift tile arrays, flattening,
        // Metal copies and per-tile headers BEFORE building the lists.
        let tileCount = ((camera.width+15)/16)*((camera.height+15)/16)
        let available = configuration.maxWorkingSetBytes - baseBytes - tileCount*64
        guard available > 32 else { throw HexelsError.resourceLimit("No memory headroom for tile lists") }
        let effectiveCap = min(configuration.maxIntersections, available/32)
        let (ids, offsets) = try bins(camera:camera,maxEntries:effectiveCap)
        guard baseBytes + ids.count*12 + offsets.count*8 < configuration.maxWorkingSetBytes else {
            throw HexelsError.resourceLimit("Tile workspace exceeds configured memory ceiling")
        }
        let r=camera.right, d=camera.down, f=camera.forward, o=camera.origin, k=camera.intrinsics
        let uniforms = SceneUniforms(origin:SIMD4(o.x,o.y,o.z,0),right:SIMD4(r.x,r.y,r.z,0),
            down:SIMD4(d.x,d.y,d.z,0),forward:SIMD4(f.x,f.y,f.z,0),intrinsics:SIMD4(k[0],k[1],k[2],k[3]),
            backgroundNear:SIMD4(background.x,background.y,background.z,camera.near),
            dimensions:SIMD4(UInt32(camera.width),UInt32(camera.height),UInt32((camera.width+15)/16),16))
        return (try makeBuffer(ids), try makeBuffer(offsets), uniforms)
    }

    private func encode(_ encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState,
                        ids: MTLBuffer, offsets: MTLBuffer, uniforms: inout SceneUniforms, output: MTLBuffer,
                        target: MTLBuffer? = nil) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(parameterBuffer,offset:0,index:0); encoder.setBuffer(ids,offset:0,index:1)
        encoder.setBuffer(offsets,offset:0,index:2)
        encoder.setBytes(&uniforms,length:MemoryLayout<SceneUniforms>.stride,index:3)
        encoder.setBuffer(output,offset:0,index:4)
        if let target {
            encoder.setBuffer(target,offset:0,index:5); encoder.setBuffer(gradientBuffer,offset:0,index:6)
        }
        encoder.dispatchThreads(MTLSize(width:Int(uniforms.dimensions.x),height:Int(uniforms.dimensions.y),depth:1),
            threadsPerThreadgroup:MTLSize(width:16,height:min(16,pipeline.maxTotalThreadsPerThreadgroup/16),depth:1))
    }

    private func output(camera: HexelCamera) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length:camera.width*camera.height*16,options:.storageModeShared) else {
            throw HexelsError.resourceLimit("Cannot allocate output image")
        }
        return buffer
    }

    public func render(camera: HexelCamera, background: SIMD3<Float>) throws -> [SIMD4<Float>] {
        try autoreleasepool {
        let (ids, offsets, initial) = try inputs(camera:camera,background:background)
        var uniforms = initial
        let image = try output(camera:camera)
        lastForwardGPUSeconds = try run { encode($0,pipeline:renderPipeline,ids:ids,offsets:offsets,uniforms:&uniforms,output:image) }
        peakAllocatedGPUBytes = max(peakAllocatedGPUBytes,device.currentAllocatedSize)
        return Array(UnsafeBufferPointer(start:image.contents().assumingMemoryBound(to:SIMD4<Float>.self),count:camera.width*camera.height))
        }
    }

    public func lossAndGradients(camera: HexelCamera, target: [SIMD4<Float>], background: SIMD3<Float>) throws -> (loss:Float,gradients:[Float]) {
        let loss = try computeGradients(camera:camera,target:target,background:background)
        return (loss, floats(gradientBuffer,count:parameterCount))
    }

    private func computeGradients(camera: HexelCamera, target: [SIMD4<Float>], background: SIMD3<Float>) throws -> Float {
        try autoreleasepool {
        let (ids, offsets, initial) = try inputs(camera:camera,background:background)
        guard target.count == camera.width*camera.height,
              target.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else {
            throw HexelsError.invalid("Target must contain one finite RGBA float per pixel")
        }
        var uniforms = initial
        let image = try output(camera:camera), targetBuffer = try makeBuffer(target)
        memset(gradientBuffer.contents(),0,gradientBuffer.length)
        lastForwardGPUSeconds = try run { encode($0,pipeline:renderPipeline,ids:ids,offsets:offsets,uniforms:&uniforms,output:image) }
        lastBackwardGPUSeconds = try run { encode($0,pipeline:backwardPipeline,ids:ids,offsets:offsets,uniforms:&uniforms,output:image,target:targetBuffer) }
        peakAllocatedGPUBytes = max(peakAllocatedGPUBytes,device.currentAllocatedSize)
        let values = image.contents().assumingMemoryBound(to:SIMD4<Float>.self)
        var squared: Double = 0
        for i in target.indices {
            let delta = values[i] - target[i]
            squared += Double(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z)
        }
        let loss = Float(squared/Double(target.count*3))
        guard loss.isFinite, floats(gradientBuffer,count:parameterCount).allSatisfy(\.isFinite) else {
            throw HexelsError.gpu("Nonfinite render loss or gradients; update was not applied")
        }
        return loss
        }
    }

    @discardableResult public func step(camera: HexelCamera, target: [SIMD4<Float>], background: SIMD3<Float>) throws -> Float {
        try autoreleasepool {
        let loss = try computeGradients(camera:camera,target:target,background:background)
        guard stepCount < Int(UInt32.max)-1 else { throw HexelsError.invalid("Optimizer step overflow") }
        let next = stepCount+1
        var uniforms = AdamUniforms(parameterCount:UInt32(parameterCount),step:UInt32(next),beta1:0.9,beta2:0.999,
            epsilon:1e-8,beta1Power:pow(0.9,Float(next)),beta2Power:pow(0.999,Float(next)),padding:0)
        try run { encoder in
            encoder.setComputePipelineState(adamPipeline)
            for (index,buffer) in [parameterBuffer,gradientBuffer,momentBuffer,varianceBuffer].enumerated() { encoder.setBuffer(buffer,offset:0,index:index) }
            encoder.setBytes(&uniforms,length:MemoryLayout<AdamUniforms>.stride,index:4)
            encoder.setBuffer(rateBuffer,offset:0,index:5)
            encoder.dispatchThreads(MTLSize(width:parameterCount,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:min(256,adamPipeline.maxTotalThreadsPerThreadgroup),height:1,depth:1))
        }
        stepCount = next
        try SceneBundle.validateParameters(parameters)
        return loss
        }
    }

    /// Atomic binary checkpoint: optimizer moments and update count resume exactly.
    public func saveCheckpoint(to url: URL) throws {
        guard (checkpointContext?.utf8.count ?? 0) <= 1024 else { throw HexelsError.invalid("Checkpoint context is too large") }
        let header = CheckpointHeader(schemaVersion:1,parameterCount:parameterCount,step:stepCount,learningRates:configuration.learningRates,context:checkpointContext)
        let metadata = try JSONEncoder().encode(header)
        var length = UInt32(metadata.count).littleEndian
        var data = Data(bytes:&length,count:4); data.append(metadata)
        for buffer in [parameterBuffer,momentBuffer,varianceBuffer] { data.append(Data(bytes:buffer.contents(),count:parameterCount*4)) }
        try data.write(to:url,options:.atomic)
    }

    public static func loadCheckpoint(from url: URL, configuration: TrainingConfiguration = TrainingConfiguration()) throws -> MetalHexelTrainer {
        try SceneBundle.checkFileSize(url,maximum:configuration.maxWorkingSetBytes)
        let data = try Data(contentsOf:url,options:.mappedIfSafe)
        guard data.count >= 4, data.count <= configuration.maxWorkingSetBytes else { throw HexelsError.invalid("Invalid checkpoint size") }
        let headerLength = data.withUnsafeBytes { Int(UInt32(littleEndian:$0.loadUnaligned(as:UInt32.self))) }
        guard headerLength > 0, headerLength <= 65536, data.count >= 4+headerLength else { throw HexelsError.invalid("Invalid checkpoint header") }
        let header = try JSONDecoder().decode(CheckpointHeader.self,from:data.subdata(in:4..<4+headerLength))
        guard header.schemaVersion == 1, header.parameterCount > 0, header.parameterCount <= 20_000_000,
              header.parameterCount % 20 == 0, header.step >= 0, header.step < Int(UInt32.max)-1,
              data.count == 4+headerLength+header.parameterCount*12 else { throw HexelsError.invalid("Invalid checkpoint state") }
        guard header.parameterCount * 48 < configuration.maxWorkingSetBytes else {
            throw HexelsError.resourceLimit("Checkpoint restore staging exceeds memory admission limit")
        }
        let offset = 4+headerLength
        func array(_ index:Int) -> [Float] {
            (0..<header.parameterCount).map { j in
                data.withUnsafeBytes { Float(bitPattern:UInt32(littleEndian:$0.loadUnaligned(fromByteOffset:offset+(index*header.parameterCount+j)*4,as:UInt32.self))) }
            }
        }
        var resumed = configuration; resumed.learningRates = header.learningRates
        let result = try MetalHexelTrainer(parameters:array(0),configuration:resumed)
        let m = array(1), v = array(2)
        guard m.allSatisfy(\.isFinite), v.allSatisfy({ $0.isFinite && $0 >= 0 }) else { throw HexelsError.invalid("Invalid Adam checkpoint moments") }
        m.withUnsafeBytes { result.momentBuffer.contents().copyMemory(from:$0.baseAddress!,byteCount:$0.count) }
        v.withUnsafeBytes { result.varianceBuffer.contents().copyMemory(from:$0.baseAddress!,byteCount:$0.count) }
        result.stepCount = header.step
        result.checkpointContext = header.context
        return result
    }
}
