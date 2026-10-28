import Foundation
import simd

/// A bounded triangle surface in supplied world coordinates. Colors are encoded sRGB.
public struct TriangleSurface: Sendable {
    public let vertices: [SIMD3<Float>]
    public let normals: [SIMD3<Float>]?
    public let colors: [SIMD3<Float>]?
    public let triangles: [SIMD3<UInt32>]
    public let label: String

    public init(vertices: [SIMD3<Float>], normals: [SIMD3<Float>]? = nil,
                colors: [SIMD3<Float>]? = nil, triangles: [SIMD3<UInt32>],
                label: String = "Supplied triangle surface in world coordinates") throws {
        guard !vertices.isEmpty, vertices.count <= 350_000, !triangles.isEmpty, triangles.count <= 1_000_000,
              vertices.allSatisfy({ [$0.x,$0.y,$0.z].allSatisfy(\.isFinite) }),
              normals == nil || (normals!.count == vertices.count && normals!.allSatisfy({
                [$0.x,$0.y,$0.z].allSatisfy(\.isFinite) && simd_length($0).isFinite && simd_length($0) > 1e-6 })),
              colors == nil || (colors!.count == vertices.count && colors!.allSatisfy({ [$0.x,$0.y,$0.z].allSatisfy(\.isFinite) })),
              triangles.allSatisfy({ Int($0.x) < vertices.count && Int($0.y) < vertices.count && Int($0.z) < vertices.count }) else {
            throw HexelsError.invalid("Triangle surface exceeds bounds or contains invalid vertices/normals/colors/indices")
        }
        self.vertices = vertices; self.normals = normals; self.colors = colors
        self.triangles = triangles; self.label = label
    }

    /// Decode copied AR mesh float3 vertices and little-endian uint16/uint32 faces.
    /// No alignment assumptions; stride, offsets and every referenced index are checked.
    public static func fromStridedBuffers(vertices data: Data, vertexCount: Int, vertexStride: Int,
                                          vertexOffset: Int = 0, indices: Data, triangleCount: Int,
                                          indexBytes: Int, indexOffset: Int = 0,
                                          transform: simd_float4x4) throws -> TriangleSurface {
        guard vertexCount > 0, vertexCount <= 350_000, triangleCount > 0, triangleCount <= 1_000_000,
              vertexStride >= 12, vertexStride <= 4096, vertexOffset >= 0, vertexOffset <= data.count,
              indexOffset >= 0, indexOffset <= indices.count, indexBytes == 2 || indexBytes == 4,
              data.count <= 64*1024*1024, indices.count <= 64*1024*1024,
              (vertexCount-1)*vertexStride+12 <= data.count-vertexOffset,
              triangleCount*3*indexBytes <= indices.count-indexOffset else {
            throw HexelsError.invalid("Truncated/unsupported mesh vertex stride, face width, offsets, or counts")
        }
        let camera = HexelCamera(width:1,height:1,intrinsics:[1,1,0,0],cameraToWorld:CaptureGeometry.rowMajor(transform))
        try camera.validate()
        var points: [SIMD3<Float>] = []; points.reserveCapacity(vertexCount)
        data.withUnsafeBytes { bytes in
            for index in 0..<vertexCount {
                let offset = vertexOffset+index*vertexStride
                func component(_ axis: Int) -> Float {
                    Float(bitPattern: UInt32(littleEndian:bytes.loadUnaligned(fromByteOffset:offset+axis*4,as:UInt32.self)))
                }
                let p = transform * SIMD4(component(0),component(1),component(2),1)
                points.append(SIMD3(p.x,p.y,p.z))
            }
        }
        var faces: [SIMD3<UInt32>] = []; faces.reserveCapacity(triangleCount)
        indices.withUnsafeBytes { bytes in
            func item(_ index: Int) -> UInt32 {
                let offset = indexOffset+index*indexBytes
                return indexBytes == 2 ? UInt32(UInt16(littleEndian:bytes.loadUnaligned(fromByteOffset:offset,as:UInt16.self)))
                    : UInt32(littleEndian:bytes.loadUnaligned(fromByteOffset:offset,as:UInt32.self))
            }
            for index in 0..<triangleCount { faces.append(SIMD3(item(index*3),item(index*3+1),item(index*3+2))) }
        }
        return try TriangleSurface(vertices:points,triangles:faces,label:"Captured AR mesh geometry; uncolored world coordinates in meters")
    }
}

public typealias MeshSurface = TriangleSurface

public enum SurfaceExport {
    public static func transformed(_ surface: TriangleSurface, by matrix: simd_float4x4) throws -> TriangleSurface {
        let camera = HexelCamera(width:1,height:1,intrinsics:[1,1,0,0],cameraToWorld:CaptureGeometry.rowMajor(matrix))
        try camera.validate()
        let rotation = simd_float3x3(columns:(SIMD3(matrix.columns.0.x,matrix.columns.0.y,matrix.columns.0.z),
            SIMD3(matrix.columns.1.x,matrix.columns.1.y,matrix.columns.1.z),SIMD3(matrix.columns.2.x,matrix.columns.2.y,matrix.columns.2.z)))
        let vertices = surface.vertices.map { p -> SIMD3<Float> in
            let result = matrix * SIMD4(p.x,p.y,p.z,1); return SIMD3(result.x,result.y,result.z)
        }
        return try TriangleSurface(vertices:vertices,normals:surface.normals?.map { simd_normalize(rotation * $0) },
            colors:surface.colors,triangles:surface.triangles,label:surface.label)
    }

    /// Exact D=0 hexagon from max(|v|, |sqrt(3)/2*u ± v/2|)=1.
    /// Independent planar patches, not fused/watertight geometry. Optical alpha and
    /// soft tails are not represented by this surface; RGB gradients are sampled at vertices.
    public static func hexelPatches(parameters: [Float]) throws -> TriangleSurface {
        guard !parameters.isEmpty, parameters.count % 20 == 0, parameters.count/20 <= 50_000 else {
            throw HexelsError.resourceLimit("Hexel patch export requires 1–50K V1 cells")
        }
        try SceneBundle.validateParameters(parameters)
        let a = 1/sqrt(Float(3))
        let ring: [SIMD2<Float>] = [SIMD2(2*a,0),SIMD2(a,1),SIMD2(-a,1),SIMD2(-2*a,0),SIMD2(-a,-1),SIMD2(a,-1)]
        var vertices: [SIMD3<Float>] = [], normals: [SIMD3<Float>] = [], colors: [SIMD3<Float>] = []
        var triangles: [SIMD3<UInt32>] = []
        let count = parameters.count/20
        vertices.reserveCapacity(count*7); normals.reserveCapacity(count*7); colors.reserveCapacity(count*7); triangles.reserveCapacity(count*6)
        for j in Swift.stride(from:0,to:parameters.count,by:20) {
            let p = SIMD3(parameters[j],parameters[j+1],parameters[j+2])
            let q = simd_quatf(ix:parameters[j+6],iy:parameters[j+7],iz:parameters[j+8],r:parameters[j+5]).normalized
            let u = q.act(SIMD3<Float>(1,0,0))*exp(parameters[j+3]), v = q.act(SIMD3<Float>(0,1,0))*exp(parameters[j+4])
            let normal = q.act(SIMD3<Float>(0,0,1)), rgb = SIMD3(parameters[j+9],parameters[j+10],parameters[j+11])
            let gradU = SIMD3(parameters[j+12],parameters[j+13],parameters[j+14]), gradV = SIMD3(parameters[j+15],parameters[j+16],parameters[j+17])
            let base = UInt32(vertices.count)
            vertices.append(p); normals.append(normal); colors.append(rgb)
            for xy in ring {
                vertices.append(p+xy.x*u+xy.y*v); normals.append(normal); colors.append(rgb+xy.x*gradU+xy.y*gradV)
            }
            for edge in 0..<6 { triangles.append(SIMD3(base,base+1+UInt32(edge),base+1+UInt32((edge+1)%6))) }
        }
        return try TriangleSurface(vertices:vertices,normals:normals,colors:colors,triangles:triangles,
            label:"Hexels V1 hard-support independent surface patches; not fused or watertight; no optical alpha or soft tails; RGB clipped to encoded-sRGB bytes")
    }

    /// Stream ASCII PLY in bounded ~64KiB text chunks; write a new file atomically.
    public static func writePLY(_ surface: TriangleSurface, to url: URL) throws {
        try stream(to:url) { emit in
            try emit("ply\nformat ascii 1.0\ncomment \(safeLabel(surface.label))\nelement vertex \(surface.vertices.count)\nproperty float x\nproperty float y\nproperty float z\n")
            if surface.normals != nil { try emit("property float nx\nproperty float ny\nproperty float nz\n") }
            if surface.colors != nil { try emit("property uchar red\nproperty uchar green\nproperty uchar blue\n") }
            try emit("element face \(surface.triangles.count)\nproperty list uchar uint vertex_indices\nend_header\n")
            for index in surface.vertices.indices {
                let p = surface.vertices[index]
                var line = "\(p.x) \(p.y) \(p.z)"
                if let n = surface.normals?[index] { line += " \(n.x) \(n.y) \(n.z)" }
                if let c = surface.colors?[index] { line += " \(byte(c.x)) \(byte(c.y)) \(byte(c.z))" }
                try emit(line+"\n")
            }
            for face in surface.triangles { try emit("3 \(face.x) \(face.y) \(face.z)\n") }
        }
    }

    /// Stream standard uncolored OBJ vertices/normals and 1-based face indices.
    public static func writeOBJ(_ surface: TriangleSurface, to url: URL) throws {
        try stream(to:url) { emit in
            try emit("# \(safeLabel(surface.label))\n# Vertex colors/optical transparency are not represented in OBJ.\n")
            for p in surface.vertices { try emit("v \(p.x) \(p.y) \(p.z)\n") }
            if let normals = surface.normals { for n in normals { try emit("vn \(n.x) \(n.y) \(n.z)\n") } }
            for face in surface.triangles {
                let a = face.x+1, b = face.y+1, c = face.z+1
                try emit(surface.normals == nil ? "f \(a) \(b) \(c)\n" : "f \(a)//\(a) \(b)//\(b) \(c)//\(c)\n")
            }
        }
    }

    private static func byte(_ x: Float) -> Int { Int((min(1,max(0,x))*255).rounded()) }
    private static func safeLabel(_ text: String) -> String { String(text.prefix(512)).replacingOccurrences(of:"\n",with:" ").replacingOccurrences(of:"\r",with:" ") }

    private static func stream(to url: URL, body: (_ emit: (String) throws -> Void) throws -> Void) throws {
        guard url.isFileURL, !FileManager.default.fileExists(atPath:url.path) else {
            throw HexelsError.invalid("Use a new local file for surface export")
        }
        let temporary = url.deletingLastPathComponent().appendingPathComponent(".surface-\(UUID().uuidString).partial")
        guard FileManager.default.createFile(atPath:temporary.path,contents:nil) else { throw HexelsError.invalid("Cannot create surface export file") }
        defer { try? FileManager.default.removeItem(at:temporary) }
        let file = try FileHandle(forWritingTo:temporary)
        do {
            var buffer = ""; buffer.reserveCapacity(65_536)
            try body { line in
                buffer += line
                if buffer.utf8.count >= 65_536 { try file.write(contentsOf:Data(buffer.utf8)); buffer.removeAll(keepingCapacity:true) }
            }
            if !buffer.isEmpty { try file.write(contentsOf:Data(buffer.utf8)) }
            try file.close()
            try FileManager.default.moveItem(at:temporary,to:url)
        } catch { try? file.close(); throw error }
    }
}
