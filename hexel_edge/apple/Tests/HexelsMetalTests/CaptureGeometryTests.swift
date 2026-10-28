import Foundation
import HexelsMetal
import simd

final class CaptureGeometryTests {
    private func view(width: Int = 5, height: Int = 5, cameraToWorld: simd_float4x4 = matrix_identity_float4x4) throws -> CaptureView {
        var bytes: [UInt8] = []
        for y in 0..<height { for x in 0..<width { bytes += [UInt8(x*40),UInt8(y*40),100,255] } }
        let image = try CaptureColorImage(width:width,height:height,rgba:bytes)
        return try CaptureView(camera:HexelCamera(width:width,height:height,intrinsics:[4,4,Float(width-1)/2,Float(height-1)/2],
            cameraToWorld:CaptureGeometry.rowMajor(cameraToWorld)),color:image)
    }

    func testARCameraAxesAndHalfPixelResize() throws {
        var ar = matrix_identity_float4x4
        ar.columns.3 = SIMD4(1,2,3,1)
        let cv = CaptureGeometry.cvCameraToWorld(arCameraToWorld:ar)
        try expectEqual(cv * SIMD4<Float>(0,0,2,1),SIMD4<Float>(1,2,1,1))
        try expectEqual(cv * SIMD4<Float>(0,1,0,1),SIMD4<Float>(1,1,3,1))
        try expectEqual(CaptureGeometry.rowMajor(cv),[1,0,0,1,0,-1,0,2,0,0,-1,3,0,0,0,1])
        let rotatedAR = simd_float4x4(columns:(SIMD4(0,0,-1,0),SIMD4(0,1,0,0),SIMD4(1,0,0,0),SIMD4(2,3,4,1)))
        let rotatedCV = CaptureGeometry.cvCameraToWorld(arCameraToWorld:rotatedAR)
        try expectEqual(rotatedCV * SIMD4<Float>(0,0,2,1),SIMD4<Float>(0,3,4,1))
        try expectEqual(rotatedCV * SIMD4<Float>(1,0,0,1),SIMD4<Float>(2,3,3,1))
        let k = SIMD4<Float>(101,103,49.2,37.8)
        let resized = try CaptureGeometry.resizedIntrinsics(k,from:SIMD2(101,77),to:SIMD2(51,39))
        let p = SIMD3<Float>(0.13,-0.2,2)
        let old = SIMD2(k.x*p.x/p.z+k.z,k.y*p.y/p.z+k.w)
        let new = SIMD2(resized.x*p.x/p.z+resized.z,resized.y*p.y/p.z+resized.w)
        try expectEqual(new.x,(old.x+0.5)*51/101-0.5,accuracy:1e-5)
        try expectEqual(new.y,(old.y+0.5)*39/77-0.5,accuracy:1e-5)
        try expectThrows(try CaptureGeometry.resizedIntrinsics(k,from:SIMD2(0,77),to:SIMD2(51,39)))
    }

    func testRegisteredDepthPlaneNormalsAndColor() throws {
        var ar = matrix_identity_float4x4; ar.columns.3 = SIMD4(1,2,3,1)
        let sampleView = try view(cameraToWorld:CaptureGeometry.cvCameraToWorld(arCameraToWorld:ar))
        let depth = try CaptureDepthMap(width:3,height:3,meters:[Float](repeating:2,count:9),confidence:[UInt8](repeating:2,count:9))
        let samples = try CaptureGeometry.depthSamples(depth:depth,view:sampleView,stride:1)
        try expectEqual(samples.count,9)
        try expectEqual(samples[4].position.x,1,accuracy:1e-6)
        try expectEqual(samples[4].position.y,2,accuracy:1e-6)
        try expectEqual(samples[4].position.z,1,accuracy:1e-6)
        for sample in samples {
            try expectEqual(sample.position.z,1,accuracy:1e-6)
            try expectEqual(sample.normal.x,0,accuracy:1e-6)
            try expectEqual(sample.normal.y,0,accuracy:1e-6)
            try expectEqual(sample.normal.z,1,accuracy:1e-6)
        }
        try expectEqual(samples[4].color.x,80/255,accuracy:1e-6)
        try expectEqual(samples[4].color.y,80/255,accuracy:1e-6)
        // First depth center maps to RGB (1/3,1/3), rather than zero: registration matters.
        try expectEqual(samples[0].color.x,(40/3)/255,accuracy:1e-6)
        try expectEqual(samples[0].position.x,1-2/2.4,accuracy:1e-6)
        try expectGreaterThan(samples[0].position.y,2)
    }

    func testNovelCameraInterpolationEndpointsShortestArcAndAdmission() throws {
        func camera(angle: Float, position: SIMD3<Float>, k: [Float], near: Float = 0.2, width: Int = 5) -> HexelCamera {
            var matrix = simd_float4x4(simd_quatf(angle:angle,axis:SIMD3<Float>(0,0,1)))
            matrix.columns.3 = SIMD4(position,1)
            return HexelCamera(width:width,height:5,intrinsics:k,cameraToWorld:CaptureGeometry.rowMajor(matrix),near:near)
        }
        let a = camera(angle:0,position:.zero,k:[4,6,2,2])
        let b = camera(angle:.pi/2,position:SIMD3(2,4,6),k:[8,10,3,4],near:0.4)
        let first = try CaptureGeometry.interpolatedCamera(a,b,fraction:0)
        let last = try CaptureGeometry.interpolatedCamera(a,b,fraction:1)
        try expectEqual(first.cameraToWorld,a.cameraToWorld); try expectEqual(last.cameraToWorld,b.cameraToWorld)
        let middle = try CaptureGeometry.interpolatedCamera(a,b)
        try expectEqual(middle.intrinsics,[6,8,2.5,3]); try expectEqual(middle.near,0.3,accuracy:1e-6)
        let m = middle.cameraToWorld, c = sqrt(Float(0.5))
        try expectEqual(m[0],c,accuracy:1e-6); try expectEqual(m[4],c,accuracy:1e-6)
        try expectEqual(m[1],-c,accuracy:1e-6); try expectEqual(m[5],c,accuracy:1e-6)
        try expectEqual([m[3],m[7],m[11]],[1,2,3])
        let before = camera(angle:170 * .pi/180,position:.zero,k:[4,4,2,2])
        let after = camera(angle:-170 * .pi/180,position:.zero,k:[4,4,2,2])
        let shortest = try CaptureGeometry.interpolatedCamera(before,after)
        try expectEqual(shortest.cameraToWorld[0],-1,accuracy:1e-6)
        try expectEqual(shortest.cameraToWorld[5],-1,accuracy:1e-6)
        try expectThrows(try CaptureGeometry.interpolatedCamera(a,b,fraction:.nan))
        try expectThrows(try CaptureGeometry.interpolatedCamera(a,b,fraction:-0.1))
        try expectThrows(try CaptureGeometry.interpolatedCamera(a,b,fraction:1.1))
        try expectThrows(try CaptureGeometry.interpolatedCamera(a,camera(angle:0,position:.zero,k:[4,4,2,2],width:4)))
        try expectThrows(try CaptureGeometry.interpolatedCamera(a,camera(angle:0,position:.zero,k:[0,4,2,2])))
    }

    func testDepthHolesConfidenceAndSamplingBounds() throws {
        var z = [Float](repeating:2,count:25); z[0] = .nan; z[1] = 0; z[2] = -1; z[3] = 9; z[4] = .infinity
        var confidence = [UInt8](repeating:2,count:25); confidence[5] = 0
        let depth = try CaptureDepthMap(width:5,height:5,meters:z,confidence:confidence)
        let sampleView = try view()
        try expectEqual(try CaptureGeometry.depthSamples(depth:depth,view:sampleView,stride:1).count,19)
        try expectLessThan(try CaptureGeometry.depthSamples(depth:depth,view:sampleView,stride:1,maxSamples:4).count,5)
        try expectThrows(try CaptureGeometry.depthSamples(depth:depth,view:sampleView,stride:0))
        try expectThrows(try CaptureDepthMap(width:5,height:5,meters:[2],confidence:nil))
        try expectThrows(try CaptureDepthMap(width:1,height:1,meters:[2],confidence:[3]))
        try expectThrows(try CaptureColorImage(width:1,height:1,rgba:[1,2,3]))
    }

    func testSparsePointsUseObservedColorsAndFacingPrior() throws {
        let sampleView = try view()
        let points: [SIMD3<Float>] = [SIMD3(0,0,2),SIMD3(9,0,2),SIMD3(0,0,-2),SIMD3(.nan,0,2),SIMD3(0,0,0.1)]
        let samples = try CaptureGeometry.sparseSamples(worldPoints:points,view:sampleView)
        try expectEqual(samples.count,1)
        try expectEqual(samples[0].normal,SIMD3<Float>(0,0,-1))
        try expectEqual(samples[0].color.x,80/255,accuracy:1e-6)
        try expectEqual(samples[0].color.y,80/255,accuracy:1e-6)
        try expectThrows(try CaptureGeometry.sparseSamples(worldPoints:points,view:sampleView,maxSamples:0))
    }

    func testVoxelSeedsBoundedFiniteAndNormalAligned() throws {
        var seeds = try CaptureSeedAccumulator(voxelSize:0.1,maxCells:2)
        let first = CaptureSample(position:SIMD3(0.01,0.02,1),normal:SIMD3(0,0,1),color:SIMD3(1,0,0),radius:0.08)
        let second = CaptureSample(position:SIMD3(0.03,0.04,1.02),normal:SIMD3(0,0,-1),color:SIMD3(0,0,1),radius:0.08)
        let third = CaptureSample(position:SIMD3(0.21,0,1),normal:SIMD3(0,1,0),color:SIMD3(0,1,0),radius:0.08)
        let fourth = CaptureSample(position:SIMD3(0.41,0,1),normal:SIMD3(0,1,0),color:SIMD3(0,1,0),radius:0.08)
        try seeds.add([first,second,third,fourth])
        try expectEqual(seeds.cellCount,2); try expectEqual(seeds.acceptedSamples,3); try expectEqual(seeds.droppedSamples,1)
        let p = try seeds.parameters()
        try expectEqual(p.count,40); try expect(p.allSatisfy(\.isFinite))
        try expectEqual(p[0],0.02,accuracy:1e-6); try expectEqual(p[1],0.03,accuracy:1e-6); try expectEqual(p[2],1.01,accuracy:1e-6)
        try expectEqual(p[9],0.5,accuracy:1e-6); try expectEqual(p[11],0.5,accuracy:1e-6)
        for (j,normal) in [(0,SIMD3<Float>(0,0,1)),(20,SIMD3<Float>(0,1,0))] {
            let q = simd_quatf(ix:p[j+6],iy:p[j+7],iz:p[j+8],r:p[j+5])
            try expectLessThan(simd_length(q.act(SIMD3(0,0,1))-normal),1e-6)
            try expectEqual(exp(p[j+3]),0.08,accuracy:1e-6)
            try expectEqual(p[j+18],0); try expectEqual(exp(p[j+19]),15,accuracy:1e-5)
        }
        try expectThrows(try CaptureSeedAccumulator(voxelSize:0.1,maxCells:50_001))
        try expectThrows(try seeds.add([CaptureSample(position:SIMD3(.nan,0,0),normal:SIMD3(0,0,1),color:.zero,radius:0.1)]))
        try expectThrows(try seeds.add([CaptureSample(position:SIMD3(0,0,1),normal:SIMD3(1e38,1e38,1e38),color:.zero,radius:0.1)]))
    }

    func testHexelSurfaceMatchesHardSupportAndGradientColor() throws {
        let q = simd_quatf(angle:0.7,axis:simd_normalize(SIMD3<Float>(1,2,3)))
        let p: [Float] = [1,2,3,log(0.2),log(0.4),q.real,q.imag.x,q.imag.y,q.imag.z,
                          0.4,0.5,0.6, 0.1,0,0, 0,0.1,0, 0,log(15)]
        let mesh = try SurfaceExport.hexelPatches(parameters:p)
        try expectEqual(mesh.vertices.count,7); try expectEqual(mesh.triangles.count,6)
        let center = SIMD3<Float>(1,2,3), u = q.act(SIMD3<Float>(1,0,0)), v = q.act(SIMD3<Float>(0,1,0)), n = q.act(SIMD3<Float>(0,0,1))
        for index in 1..<7 {
            let delta = mesh.vertices[index]-center, uu = simd_dot(delta,u)/0.2, vv = simd_dot(delta,v)/0.4
            let support = max(abs(vv),max(abs(sqrt(Float(3))/2*uu+0.5*vv),abs(sqrt(Float(3))/2*uu-0.5*vv)))
            try expectEqual(support,1,accuracy:2e-6)
            try expectEqual(simd_dot(delta,n),0,accuracy:1e-6)
            try expectEqual(mesh.colors![index].x,0.4+0.1*uu,accuracy:1e-6)
            try expectEqual(mesh.colors![index].y,0.5+0.1*vv,accuracy:1e-6)
        }
        for face in mesh.triangles {
            let cross = simd_cross(mesh.vertices[Int(face.y)]-mesh.vertices[Int(face.x)],mesh.vertices[Int(face.z)]-mesh.vertices[Int(face.x)])
            try expectGreaterThan(simd_dot(cross,n),0)
        }
        try expect(mesh.label.contains("not fused or watertight"))
        try expectThrows(try SurfaceExport.hexelPatches(parameters:Array(repeating:p,count:50_001).flatMap { $0 }))
    }

    func testMeshStrideOffsetsTransformsAndIndexAdmission() throws {
        var vertices = Data([0xff])
        for p in [SIMD3<Float>(0,0,0),SIMD3(1,0,0),SIMD3(0,1,0)] {
            for value in [p.x,p.y,p.z] { var bits = value.bitPattern.littleEndian; withUnsafeBytes(of:&bits) { vertices.append(contentsOf:$0) } }
            vertices.append(contentsOf:[0xaa,0xbb,0xcc,0xdd])
        }
        var indices = Data([0xff])
        for value: UInt16 in [0,1,2] { var number = value.littleEndian; withUnsafeBytes(of:&number) { indices.append(contentsOf:$0) } }
        var transform = matrix_identity_float4x4; transform.columns.3 = SIMD4(2,3,4,1)
        let mesh = try TriangleSurface.fromStridedBuffers(vertices:vertices,vertexCount:3,vertexStride:16,vertexOffset:1,
            indices:indices,triangleCount:1,indexBytes:2,indexOffset:1,transform:transform)
        try expectEqual(mesh.vertices,[SIMD3(2,3,4),SIMD3(3,3,4),SIMD3(2,4,4)])
        try expectEqual(mesh.triangles,[SIMD3<UInt32>(0,1,2)])
        try expect(mesh.colors == nil)
        try expectThrows(try TriangleSurface.fromStridedBuffers(vertices:vertices,vertexCount:4,vertexStride:16,vertexOffset:1,
            indices:indices,triangleCount:1,indexBytes:2,indexOffset:1,transform:transform))
        try expectThrows(try TriangleSurface.fromStridedBuffers(vertices:vertices,vertexCount:3,vertexStride:8,
            indices:indices,triangleCount:1,indexBytes:1,transform:transform))
        try expectThrows(try TriangleSurface(vertices:mesh.vertices,triangles:[SIMD3<UInt32>(0,1,3)]))
        var bad = transform; bad.columns.0.x = 2
        try expectThrows(try SurfaceExport.transformed(mesh,by:bad))
    }

    func testPLYOBJRoundTripCountsIndicesNormalsAndColors() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:directory) }
        let mesh = try TriangleSurface(vertices:[SIMD3(0,0,0),SIMD3(1,0,0),SIMD3(0,1,0)],
            normals:[SIMD3<Float>](repeating:SIMD3(0,0,1),count:3),colors:[SIMD3(-1,0.5,2),SIMD3(1,0,0),SIMD3(0,1,0)],
            triangles:[SIMD3<UInt32>(0,1,2)])
        let ply = directory.appendingPathComponent("surface.ply"), obj = directory.appendingPathComponent("surface.obj")
        try SurfaceExport.writePLY(mesh,to:ply); try SurfaceExport.writeOBJ(mesh,to:obj)
        let text = try String(contentsOf:ply,encoding:.utf8)
        try expect(text.contains("element vertex 3\n")); try expect(text.contains("element face 1\n"))
        let body = try requireValue(text.components(separatedBy:"end_header\n").last)
        let rows = body.split(separator:"\n").map { $0.split(separator:" ").map(String.init) }
        try expectEqual(rows.count,4)
        try expectEqual(rows[0].compactMap(Float.init),[0,0,0,0,0,1,0,128,255])
        try expectEqual(rows[3], ["3","0","1","2"])
        let objRows = try String(contentsOf:obj,encoding:.utf8).split(separator:"\n")
        try expectEqual(objRows.filter { $0.hasPrefix("v ") }.count,3)
        try expectEqual(objRows.filter { $0.hasPrefix("vn ") }.count,3)
        try expectEqual(objRows.filter { $0.hasPrefix("f ") }.map(String.init),["f 1//1 2//2 3//3"])
        try expectThrows(try SurfaceExport.writePLY(mesh,to:ply))
        try expectThrows(try TriangleSurface(vertices:[SIMD3(.infinity,0,0)],triangles:[SIMD3<UInt32>(0,0,0)]))
    }
}
