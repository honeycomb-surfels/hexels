import CoreVideo
import Foundation
import HexelsMetal
import simd

private enum CheckError: Error { case failed(String) }
@MainActor @main struct CapturePipelineChecks {
    private static var checks = 0
    private static func expect(_ value: @autoclosure () -> Bool, _ message: String) throws {
        guard value() else { throw CheckError.failed(message) }; checks += 1
    }
    private static func rejects(_ operation: () throws -> Void, _ message: String) throws {
        do { try operation() } catch { checks += 1; return }
        throw CheckError.failed(message)
    }
    private static func pixelBuffer(width: Int,height: Int,format: OSType) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,width,height,format,
            [kCVPixelBufferIOSurfacePropertiesKey:[:]] as CFDictionary,&buffer)
        guard status == kCVReturnSuccess, let buffer else { throw CheckError.failed("CVPixelBufferCreate failed") }
        return buffer
    }
    private static func nv12(width: Int = 4,height: Int = 4,full: Bool = true,
                             y: (Int,Int)->UInt8,uv: (Int,Int)->(UInt8,UInt8) = { _,_ in (128,128) }) throws -> CVPixelBuffer {
        let buffer = try pixelBuffer(width:width,height:height,format:full ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
        CVPixelBufferLockBaseAddress(buffer,[]); defer { CVPixelBufferUnlockBaseAddress(buffer,[]) }
        for plane in 0..<2 {
            let rows = CVPixelBufferGetHeightOfPlane(buffer,plane), stride = CVPixelBufferGetBytesPerRowOfPlane(buffer,plane)
            guard let base = CVPixelBufferGetBaseAddressOfPlane(buffer,plane) else { throw CheckError.failed("Missing NV12 plane") }
            memset(base,231,rows*stride) // Padding must never become image data.
            for row in 0..<rows { for column in 0..<CVPixelBufferGetWidthOfPlane(buffer,plane) {
                if plane == 0 { base.storeBytes(of:y(column,row),toByteOffset:row*stride+column,as:UInt8.self) }
                else {
                    let pair = uv(column,row)
                    base.storeBytes(of:pair.0,toByteOffset:row*stride+column*2,as:UInt8.self)
                    base.storeBytes(of:pair.1,toByteOffset:row*stride+column*2+1,as:UInt8.self)
                }
            } }
        }
        return buffer
    }
    private static func depthBuffers(width: Int = 32,height: Int = 24) throws -> (CVPixelBuffer,CVPixelBuffer) {
        let depth = try pixelBuffer(width:width,height:height,format:kCVPixelFormatType_DepthFloat32)
        let confidence = try pixelBuffer(width:width,height:height,format:kCVPixelFormatType_OneComponent8)
        CVPixelBufferLockBaseAddress(depth,[]); CVPixelBufferLockBaseAddress(confidence,[])
        defer { CVPixelBufferUnlockBaseAddress(depth,[]); CVPixelBufferUnlockBaseAddress(confidence,[]) }
        guard let d = CVPixelBufferGetBaseAddress(depth), let c = CVPixelBufferGetBaseAddress(confidence) else { throw CheckError.failed("Missing depth plane") }
        for y in 0..<height { for x in 0..<width {
            d.storeBytes(of:Float(2),toByteOffset:y*CVPixelBufferGetBytesPerRow(depth)+x*4,as:Float.self)
            c.storeBytes(of:UInt8(2),toByteOffset:y*CVPixelBufferGetBytesPerRow(confidence)+x,as:UInt8.self)
        } }
        return (depth,confidence)
    }
    static func main() throws {
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("hexels-capture-checks-\(UUID().uuidString)",isDirectory:true)
        try FileManager.default.createDirectory(at:scratch,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:scratch) }
        let asymmetric = try nv12(y:{ x,y in UInt8(10+y*60+x*10) })
        let native = try CapturePixels.color(asymmetric,longEdge:640)
        try expect(native.width == 4 && native.height == 4,"No upsampling")
        try expect(native.rgba[0] == 10 && native.rgba[(3*4+3)*4] == 220,"Raw top-left/bottom-right rows and padding")
        let reduced = try CapturePixels.color(asymmetric,longEdge:2)
        try expect(reduced.width == 2 && reduced.height == 2 && reduced.rgba[0] == 45,"Half-pixel bilinear resize")
        let fullRed = try nv12(y:{ _,_ in 76 },uv:{ _,_ in (85,255) })
        let red601 = try CapturePixels.color(fullRed,longEdge:4)
        try expect(red601.rgba[0] >= 250 && red601.rgba[1] <= 2 && red601.rgba[2] <= 2,"Full-range601 red conversion")
        CVBufferSetAttachment(fullRed,kCVImageBufferYCbCrMatrixKey,kCVImageBufferYCbCrMatrix_ITU_R_709_2,.shouldPropagate)
        let red709 = try CapturePixels.color(fullRed,longEdge:4)
        try expect(red709.rgba[1] >= 20 && red709.rgba[1] <= 28,"709 matrix attachment affects conversion")
        let video = try nv12(full:false,y:{ _,row in row < 2 ? 16 : 235 })
        let videoRGB = try CapturePixels.color(video,longEdge:4)
        try expect(videoRGB.rgba[0] == 0 && videoRGB.rgba[3*4*4] == 255,"Video-range black/white endpoints")
        let gradient = try nv12(y:{ _,_ in 100 },uv:{ x,_ in (128,x == 0 ? 128 : 240) })
        let centered = try CapturePixels.color(gradient,longEdge:4)
        CVBufferSetAttachment(gradient,kCVImageBufferChromaLocationTopFieldKey,kCVImageBufferChromaLocation_Left,.shouldPropagate)
        let left = try CapturePixels.color(gradient,longEdge:4)
        try expect(Int(left.rgba[4])-Int(centered.rgba[4]) >= 30,"Left chroma registration differs from centered")
        let encoded = try nv12(y:{ _,_ in 64 })
        CVBufferSetAttachment(encoded,kCVImageBufferTransferFunctionKey,kCVImageBufferTransferFunction_ITU_R_709_2,.shouldPropagate)
        let srgb = try CapturePixels.color(encoded,longEdge:4)
        try expect((76...84).contains(Int(srgb.rgba[0])),"709 transfer explicitly converted to sRGB")
        CVBufferSetAttachment(encoded,kCVImageBufferColorPrimariesKey,kCVImageBufferColorPrimaries_P3_D65,.shouldPropagate)
        try rejects({ _ = try CapturePixels.color(encoded,longEdge:4) },"Unsupported wide-gamut primaries must reject")
        CVBufferSetAttachment(fullRed,kCVImageBufferYCbCrMatrixKey,kCVImageBufferYCbCrMatrix_ITU_R_2020,.shouldPropagate)
        try rejects({ _ = try CapturePixels.color(fullRed,longEdge:4) },"Unsupported2020 matrix must reject")
        let (depth,confidence) = try depthBuffers()
        let copied = try CapturePixels.depth(depth,confidence:confidence)
        try expect(copied.width == 32 && copied.height == 24 && copied.meters.allSatisfy { $0 == 2 },"Depth copied using real padded stride")
        try expect(copied.confidence?.allSatisfy { $0 == 2 } == true,"Confidence dimensions/stride preserved")
        CVPixelBufferLockBaseAddress(confidence,[])
        CVPixelBufferGetBaseAddress(confidence)!.storeBytes(of:UInt8(3),as:UInt8.self)
        CVPixelBufferUnlockBaseAddress(confidence,[])
        try rejects({ _ = try CapturePixels.depth(depth,confidence:confidence) },"Invalid confidence code must reject")
        CVPixelBufferLockBaseAddress(confidence,[])
        CVPixelBufferGetBaseAddress(confidence)!.storeBytes(of:UInt8(2),as:UInt8.self)
        CVPixelBufferUnlockBaseAddress(confidence,[])
        var completed: URL!
        do {
            let recorder = try CaptureBundleRecorder(settings:CaptureSettings(maxFrames:24,longEdge:640,maxSeeds:2_000,voxelSize:0.02),storageRoot:scratch)
            let color = try nv12(width:8,height:6,y:{ x,y in UInt8(10+y*30+x*5) })
            for index in 0..<12 {
                var pose = matrix_identity_float4x4; pose.columns.3.x = Float(index)*0.04
                let k = simd_float3x3(columns:(SIMD3(8,0,0),SIMD3(0,8,0),SIMD3(3.5,2.5,1)))
                try recorder.ingest(CapturePayload(color:color,depth:depth,confidence:confidence,intrinsics:k,
                    arCameraToWorld:pose,timestamp:Double(index)*0.5,featurePoints:[],landscapePoseError:0))
            }
            try expect(recorder.canFinish && recorder.pointCount >= 64,"Translated depth capture admits finishing")
            let result = try recorder.finish(meshes:[],droppedMeshes:0)
            completed = result.bundle
            try expect(result.mesh == nil,"No invented captured mesh when anchors are absent")
            let scene = try SceneBundle(url:result.bundle)
            try expect(scene.frames.count == 12 && scene.frames.filter { $0.split == "train" }.count == 10,"Deterministic training/heldout split")
            try expect(scene.recipeHint == "detail" && scene.initialParameters.count/20 == recorder.pointCount,"Real bundle recipe/parameter validation")
            try expect(scene.viewOrder.allSatisfy { scene.frames[Int($0)].split == "train" },"Heldout frames excluded from optimizer schedule")
            let decoded = try scene.image(for:scene.frames[0])
            let expected = try CapturePixels.color(color,longEdge:640)
            try expect(decoded.enumerated().allSatisfy { index,pixel in (0..<3).allSatisfy { abs(pixel[$0]-Float(expected.rgba[index*4+$0])/255) < 1e-6 } },"PNG/source/SceneBundle round-trip preserves all asymmetric RGB rows")
            let record = try JSONSerialization.jsonObject(with:Data(contentsOf:result.bundle.appendingPathComponent("capture_summary.json"))) as! [String:Any]
            try expect(record["frame_count"] as? Int == 12 && record["held_out_frames"] as? Int == 2,"Numeric summary agrees with published bundle")
            try expect(!FileManager.default.fileExists(atPath:recorder.root.appendingPathComponent(".staging").path),"Atomic publication removes staging name")
        }
        try expect(FileManager.default.fileExists(atPath:completed.appendingPathComponent("manifest.json").path),"Finalized bundle survives successful recorder handoff")
        let sparse = try CaptureBundleRecorder(settings:CaptureSettings(maxFrames:24,longEdge:640,maxSeeds:2_000,voxelSize:0.02),storageRoot:scratch)
        let worldPoints = (0..<100).map { SIMD3<Float>(Float($0%10)*0.08-0.36,Float($0/10)*0.08-0.36,-2) }
        let cameraK = simd_float3x3(columns:(SIMD3<Float>(4,0,0),SIMD3<Float>(0,4,0),SIMD3<Float>(1.5,1.5,1)))
        for index in 0..<12 {
            var pose = matrix_identity_float4x4; pose.columns.3.x = Float(index)*0.04
            try sparse.ingest(CapturePayload(color:asymmetric,depth:nil,confidence:nil,intrinsics:cameraK,
                arCameraToWorld:pose,timestamp:Double(index)*0.5,featurePoints:worldPoints,landscapePoseError:0))
        }
        try expect(sparse.canFinish,"Tracked-feature fallback makes valid bounded seeds without fabricated depth")
        let vertexValues: [Float] = [0,0,0,1,0,0,0,1,0]
        let indexValues: [UInt16] = [0,1,2]
        let vertices = vertexValues.withUnsafeBytes { Data($0) }, indices = indexValues.withUnsafeBytes { Data($0) }
        var meshTransform = matrix_identity_float4x4; meshTransform.columns.3.x = 5
        let mesh = CapturedMeshBuffers(vertices:vertices,vertexCount:3,vertexStride:12,vertexOffset:0,
            indices:indices,triangleCount:1,indexBytes:2,transform:meshTransform)
        let surfaced = try sparse.finish(meshes:[mesh],droppedMeshes:0)
        let obj = try String(contentsOf:surfaced.mesh!,encoding:.utf8)
        let firstVertex = obj.split(separator:"\n").first { $0.hasPrefix("v ") }!.split(separator:" ").dropFirst().compactMap { Float($0) }
        try expect(firstVertex == [5,0,0],"Captured OBJ publisher applies anchor world transform")
        let surfacedManifest = try JSONSerialization.jsonObject(with:Data(contentsOf:surfaced.bundle.appendingPathComponent("manifest.json"))) as! [String:Any]
        try expect(surfacedManifest["captured_surface_file"] as? String == "captured_surface.obj","Captured surface path explicitly bound in manifest")
        let sparseRecord = try JSONSerialization.jsonObject(with:Data(contentsOf:surfaced.bundle.appendingPathComponent("capture_summary.json"))) as! [String:Any]
        try expect(sparseRecord["depth_frames"] as? Int == 0 && sparseRecord["sparse_training_frames"] as? Int == 10,"Feature fallback truthfully records absent depth")
        try expect(sparseRecord["mesh_vertices"] as? Int == 3 && sparseRecord["mesh_triangles"] as? Int == 1,"Captured mesh counts match exported OBJ")
        let cancelled = try CaptureBundleRecorder(settings:CaptureSettings(),storageRoot:scratch)
        let cancelledRoot = cancelled.root
        try cancelled.ingest(CapturePayload(color:asymmetric,depth:nil,confidence:nil,intrinsics:cameraK,
            arCameraToWorld:matrix_identity_float4x4,timestamp:0,featurePoints:worldPoints,landscapePoseError:0))
        try expect(FileManager.default.fileExists(atPath:cancelledRoot.appendingPathComponent(".staging/images/00000.png").path),"Cancellation fixture contains an actual private camera-image file")
        cancelled.cancel()
        try expect(!FileManager.default.fileExists(atPath:cancelledRoot.path),"Cancel removes incomplete private capture root")
        print("PASS: \(checks) native pixel/depth/PNG/publication/cancel checks; synthetic private scratch only; no camera or network used")
    }
}
