import CoreVideo
import Foundation
import HexelsMetal
import simd

struct CaptureSettings: Sendable, Equatable {
    var maxFrames = 48
    var longEdge = 640
    var maxSeeds = 20_000
    var voxelSize: Float = 0.02

    func validated() throws -> CaptureSettings {
        guard [24,48,64].contains(maxFrames), [640,1280].contains(longEdge),
              (64...20_000).contains(maxSeeds), voxelSize.isFinite, (0.005...0.1).contains(voxelSize) else {
            throw HexelsError.invalid("Capture requires 24/48/64 frames, 640/1280 pixels and 64–20K seed cells")
        }
        return self
    }
}

/// Owns at most one frame's pixel buffers. No ARFrame or ARPointCloud is retained.
struct CapturePayload: @unchecked Sendable {
    let color: CVPixelBuffer
    let depth: CVPixelBuffer?
    let confidence: CVPixelBuffer?
    let intrinsics: simd_float3x3
    let arCameraToWorld: simd_float4x4
    let timestamp: TimeInterval
    let featurePoints: [SIMD3<Float>]
    let landscapePoseError: Float
}

struct CapturedMeshBuffers: Sendable {
    let vertices: Data
    let vertexCount: Int
    let vertexStride: Int
    let vertexOffset: Int
    let indices: Data
    let triangleCount: Int
    let indexBytes: Int
    let transform: simd_float4x4
}

/// Accessed only by the bridge's single processing queue.
final class CaptureBundleRecorder {
    let settings: CaptureSettings
    let root: URL
    private let staging: URL
    private var accumulator: CaptureSeedAccumulator
    private var frames: [[String: Any]] = []
    private var positions: [SIMD3<Float>] = []
    private var timestamps: [Double] = []
    private var depthFrames = 0
    private var trainingDepthFrames = 0
    private var sparseFrames = 0
    private var maxPoseError: Float = 0
    private var writtenBytes = 0
    private var firstTimestamp: Double?
    var frameCount: Int { frames.count }
    var pointCount: Int { accumulator.cellCount }
    var translationSpan: Float {
        guard let first = positions.first else { return 0 }
        return positions.map { simd_distance(first,$0) }.max() ?? 0
    }
    var canFinish: Bool { frameCount >= 12 && pointCount >= 64 && translationSpan >= 0.12 }

    init(settings: CaptureSettings, storageRoot: URL? = nil) throws {
        self.settings = try settings.validated()
        let base = storageRoot ?? URL.applicationSupportDirectory.appendingPathComponent("HexelsEdge/Captures",isDirectory:true)
        guard base.isFileURL else { throw HexelsError.invalid("Private capture storage must be a local file URL") }
        root = base.appendingPathComponent(UUID().uuidString,isDirectory:true)
        staging = root.appendingPathComponent(".staging", isDirectory: true)
        accumulator = try CaptureSeedAccumulator(voxelSize: settings.voxelSize, maxCells: settings.maxSeeds)
        try FileManager.default.createDirectory(at: staging.appendingPathComponent("images"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: staging.appendingPathComponent("depth"), withIntermediateDirectories: true)
        var privateRoot = root, values = URLResourceValues(); values.isExcludedFromBackup = true
        try privateRoot.setResourceValues(values)
    }

    func ingest(_ payload: CapturePayload) throws {
        guard frames.count < settings.maxFrames else { return }
        guard payload.landscapePoseError < 0.001 else { throw HexelsError.invalid("Native-image camera orientation disagrees with ARKit's landscape view matrix") }
        let color = try CapturePixels.color(payload.color,longEdge:settings.longEdge)
        let rawSize = SIMD2(CVPixelBufferGetWidth(payload.color),CVPixelBufferGetHeight(payload.color))
        let k = try CaptureGeometry.resizedIntrinsics(SIMD4(payload.intrinsics[0,0],payload.intrinsics[1,1],payload.intrinsics[2,0],payload.intrinsics[2,1]),
            from:rawSize,to:SIMD2(color.width,color.height))
        let pose = CaptureGeometry.cvCameraToWorld(arCameraToWorld:payload.arCameraToWorld)
        let camera = HexelCamera(width:color.width,height:color.height,intrinsics:[k.x,k.y,k.z,k.w],cameraToWorld:CaptureGeometry.rowMajor(pose))
        let view = try HexelsMetal.CaptureView(camera:camera,color:color)
        let index = frames.count, name = String(format:"%05d",index)
        let split = index % 8 == 0 ? "test" : "train"
        let imagePath = "images/\(name).png"
        try CapturePixels.writePNG(color,to:staging.appendingPathComponent(imagePath))
        writtenBytes += try fileBytes(imagePath)
        var frame: [String: Any] = ["name":"\(name).png","image":imagePath,"width":color.width,"height":color.height,
            "intrinsics":[k.x,k.y,k.z,k.w],"camera_to_world":CaptureGeometry.rowMajor(pose),"split":split]
        if let buffer = payload.depth {
            let depth = try CapturePixels.depth(buffer,confidence:payload.confidence)
            let path = "depth/\(name).f32"
            try writeFloats(depth.meters,to:staging.appendingPathComponent(path))
            writtenBytes += depth.meters.count*4
            frame["depth_file"] = path; frame["depth_width"] = depth.width; frame["depth_height"] = depth.height
            if let confidence = depth.confidence {
                let path = "depth/\(name).confidence.u8"
                try Data(confidence).write(to:staging.appendingPathComponent(path),options:.atomic)
                writtenBytes += confidence.count; frame["confidence_file"] = path
            }
            depthFrames += 1
            if split == "train" {
                try accumulator.add(CaptureGeometry.depthSamples(depth:depth,view:view,stride:4,minConfidence:1,maxSamples:12_000))
                trainingDepthFrames += 1
            }
        } else if split == "train" {
            try accumulator.add(CaptureGeometry.sparseSamples(worldPoints:payload.featurePoints,view:view,maxSamples:12_000))
            sparseFrames += 1
        }
        guard writtenBytes <= 480*1_048_576 else { throw HexelsError.resourceLimit("Private capture exceeds 480 MiB; finish a smaller capture") }
        positions.append(SIMD3(pose.columns.3.x,pose.columns.3.y,pose.columns.3.z))
        if firstTimestamp == nil { firstTimestamp = payload.timestamp }
        timestamps.append(payload.timestamp-firstTimestamp!)
        maxPoseError = max(maxPoseError,payload.landscapePoseError)
        frames.append(frame)
    }

    func finish(meshes: [CapturedMeshBuffers], droppedMeshes: Int) throws -> (bundle: URL, mesh: URL?) {
        guard canFinish else { throw HexelsError.invalid("Need at least 12 accepted views, a 12 cm translated baseline and 64 valid seed cells. Move slowly around a textured static scene.") }
        let parameters = try accumulator.parameters()
        try writeFloats(parameters,to:staging.appendingPathComponent("parameters.f32"))
        let training = frames.indices.filter { frames[$0]["split"] as? String == "train" }
        var order = Data()
        for index in training { var value = UInt32(index).littleEndian; withUnsafeBytes(of:&value) { order.append(contentsOf:$0) } }
        try order.write(to:staging.appendingPathComponent("view_order.u32"),options:.atomic)
        var meshFile: String?
        var meshVertexCount = 0, meshTriangleCount = 0
        if !meshes.isEmpty {
            var vertices: [SIMD3<Float>] = [], triangles: [SIMD3<UInt32>] = []
            for mesh in meshes {
                let surface = try TriangleSurface.fromStridedBuffers(vertices:mesh.vertices,vertexCount:mesh.vertexCount,
                    vertexStride:mesh.vertexStride,vertexOffset:mesh.vertexOffset,indices:mesh.indices,
                    triangleCount:mesh.triangleCount,indexBytes:mesh.indexBytes,transform:mesh.transform)
                guard vertices.count+surface.vertices.count <= 250_000, triangles.count+surface.triangles.count <= 500_000 else {
                    throw HexelsError.resourceLimit("Captured mesh exceeds 250K vertices / 500K triangles; use a smaller area")
                }
                let base = UInt32(vertices.count)
                vertices.append(contentsOf:surface.vertices)
                triangles.append(contentsOf:surface.triangles.map { $0 &+ SIMD3(repeating:base) })
            }
            if !triangles.isEmpty {
                meshFile = "captured_surface.obj"
                let surface = try TriangleSurface(vertices:vertices,triangles:triangles,
                    label:"ARKit captured LiDAR surface, world meters; all-session geometry, not the trained Hexels model")
                try SurfaceExport.writeOBJ(surface,to:staging.appendingPathComponent(meshFile!))
                meshVertexCount = vertices.count; meshTriangleCount = triangles.count
            }
        }
        let provenance: [String: Any] = ["source":"local ARSession guided capture","world_units":"meters",
            "image_orientation":1,"pixel_coordinates":"integer centers; original sensor row order; no display transform",
            "camera_axes":"right/down/forward","tracking_pose_source":"ARKit visual-inertial estimate; all-session observations",
            "seed_rgb_and_depth_use_heldout_frames":false,"initialization":"training-frame confident axial depth; sparse tracked points when depth absent",
            "tracking_and_mesh_use_all_session_observations":true,"rgb_and_depth_seed_observations_use_training_split_only":true,
            "sparse_normal_limit":"camera-facing prior; not measured dense normals","pose_orientation_max_absolute_error":maxPoseError]
        var manifest: [String: Any] = ["schema_version":1,"variant":"v1","parameter_layout":"hexels_v1_log_wxyz_20",
            "primitive_count":parameters.count/20,"parameters_file":"parameters.f32","view_order_file":"view_order.u32",
            "recipe_hint":"detail","background":[0,0,0],"frames":frames,"provenance":provenance]
        if let meshFile { manifest["captured_surface_file"] = meshFile }
        let record: [String: Any] = ["schema_version":1,"frame_count":frameCount,"training_frames":training.count,
            "held_out_frames":frameCount-training.count,"depth_frames":depthFrames,"training_depth_frames":trainingDepthFrames,
            "sparse_training_frames":sparseFrames,"seed_cells":parameters.count/20,"seed_samples":accumulator.acceptedSamples,
            "dropped_seed_samples":accumulator.droppedSamples,"voxel_size_meters":settings.voxelSize,
            "translated_span_meters":translationSpan,"relative_frame_timestamps_seconds":timestamps,
            "mesh_vertices":meshVertexCount,"mesh_triangles":meshTriangleCount,"dropped_mesh_updates":droppedMeshes,
            "source_images_uploaded":false,"provenance":provenance]
        try writeJSON(record,to:staging.appendingPathComponent("capture.json"))
        try writeJSON(record,to:staging.appendingPathComponent("capture_summary.json"))
        try writeJSON(manifest,to:staging.appendingPathComponent("manifest.json"))
        let validated = try SceneBundle(url:staging)
        guard validated.initialParameters.count == parameters.count,
              validated.frames.count == frameCount else { throw HexelsError.invalid("Final capture bundle validation differs") }
        // Validate PNG decode/calibration before the directory becomes visible to the app.
        for frame in validated.frames { _ = try validated.image(for:frame,maxDecodedBytes:64*1_048_576) }
        guard let files = FileManager.default.enumerator(at:staging,includingPropertiesForKeys:[.fileSizeKey]) else { throw HexelsError.invalid("Cannot verify private capture file sizes") }
        var totalBytes = 0, fileCount = 0
        for case let file as URL in files {
            totalBytes += try file.resourceValues(forKeys:[.fileSizeKey]).fileSize ?? 0
            fileCount += 1
        }
        guard totalBytes <= 512*1_048_576, fileCount <= 2_000 else { throw HexelsError.resourceLimit("Completed capture exceeds the app's import bounds") }
        let final = root.appendingPathComponent("scene",isDirectory:true)
        try FileManager.default.moveItem(at:staging,to:final)
        return (final,meshFile.map { final.appendingPathComponent($0) })
    }

    func cancel() { try? FileManager.default.removeItem(at:root) }
    private func fileBytes(_ path: String) throws -> Int { try staging.appendingPathComponent(path).resourceValues(forKeys:[.fileSizeKey]).fileSize ?? 0 }
    private func writeFloats(_ values: [Float], to url: URL) throws {
        var data = Data(); data.reserveCapacity(values.count*4)
        for item in values { var bits = item.bitPattern.littleEndian; withUnsafeBytes(of:&bits) { data.append(contentsOf:$0) } }
        try data.write(to:url,options:.atomic)
    }
    private func writeJSON(_ object: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject:object,options:[.prettyPrinted,.sortedKeys]).write(to:url,options:.atomic)
    }
}
