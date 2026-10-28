import CoreGraphics
import CoreVideo
import Foundation
import HexelsMetal
import ImageIO
import UniformTypeIdentifiers

/// Raw sensor rows stay top-to-bottom. No display orientation or mirroring is applied.
enum CapturePixels {
    static func color(_ buffer: CVPixelBuffer, longEdge: Int) throws -> CaptureColorImage {
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        guard width > 0, height > 0, width <= 4096, height <= 4096 else { throw HexelsError.resourceLimit("Camera image exceeds capture bounds") }
        let size = DevicePolicy.resizedDimensions(width: width, height: height, longEdge: longEdge)
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let format = CVPixelBufferGetPixelFormatType(buffer)
        let full = format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        guard full || format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
              CVPixelBufferGetPlaneCount(buffer) == 2,
              let yBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
              let uvBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) else {
            throw HexelsError.invalid("Unsupported AR camera pixel format; expected two-plane YCbCr")
        }
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        let uvStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        let uvWidth = CVPixelBufferGetWidthOfPlane(buffer, 1), uvHeight = CVPixelBufferGetHeightOfPlane(buffer, 1)
        guard yStride >= width, uvStride >= uvWidth * 2, uvWidth > 0, uvHeight > 0 else { throw HexelsError.invalid("Invalid camera plane strides") }
        let yBytes = yBase.assumingMemoryBound(to: UInt8.self), uvBytes = uvBase.assumingMemoryBound(to: UInt8.self)
        let attachment = CVBufferCopyAttachment(buffer, kCVImageBufferYCbCrMatrixKey, nil) as? String
        let primaries = CVBufferCopyAttachment(buffer,kCVImageBufferColorPrimariesKey,nil) as? String
        let transfer = CVBufferCopyAttachment(buffer,kCVImageBufferTransferFunctionKey,nil) as? String
        let chroma = CVBufferCopyAttachment(buffer,kCVImageBufferChromaLocationTopFieldKey,nil) as? String
        guard primaries == nil || primaries == (kCVImageBufferColorPrimaries_ITU_R_709_2 as String),
              transfer == nil || transfer == (kCVImageBufferTransferFunction_sRGB as String) || transfer == (kCVImageBufferTransferFunction_ITU_R_709_2 as String) else {
            throw HexelsError.invalid("Camera wide-gamut/HDR color encoding is unsupported by this bounded sRGB capture")
        }
        let chromaOrigin: SIMD2<Float>
        if chroma == nil || chroma == (kCVImageBufferChromaLocation_Center as String) { chromaOrigin = SIMD2(0.5,0.5) }
        else if chroma == (kCVImageBufferChromaLocation_Left as String) { chromaOrigin = SIMD2(0,0.5) }
        else if chroma == (kCVImageBufferChromaLocation_TopLeft as String) { chromaOrigin = .zero }
        else if chroma == (kCVImageBufferChromaLocation_Top as String) { chromaOrigin = SIMD2(0.5,0) }
        else { throw HexelsError.invalid("Unsupported interlaced camera chroma registration") }
        let coefficients: SIMD4<Float>
        if attachment == (kCVImageBufferYCbCrMatrix_ITU_R_709_2 as String) {
            coefficients = SIMD4(1.5748, -0.187324, -0.468124, 1.8556)
        } else if attachment == nil || attachment == (kCVImageBufferYCbCrMatrix_ITU_R_601_4 as String) {
            coefficients = SIMD4(1.402, -0.344136, -0.714136, 1.772)
        } else { throw HexelsError.invalid("Unsupported camera YCbCr color matrix") }
        let rec709Transfer = transfer == (kCVImageBufferTransferFunction_ITU_R_709_2 as String)
        func sample(_ bytes: UnsafePointer<UInt8>, _ stride: Int, _ w: Int, _ h: Int, _ x: Float, _ y: Float, _ channels: Int, _ channel: Int) -> Float {
            let left = Int(floor(x)), top = Int(floor(y))
            let x0 = max(0,min(w-1,left)), x1 = max(0,min(w-1,left+1))
            let y0 = max(0,min(h-1,top)), y1 = max(0,min(h-1,top+1))
            let a = x-Float(left), b = y-Float(top)
            return (1-b)*((1-a)*Float(bytes[y0*stride+x0*channels+channel])+a*Float(bytes[y0*stride+x1*channels+channel]))
                + b*((1-a)*Float(bytes[y1*stride+x0*channels+channel])+a*Float(bytes[y1*stride+x1*channels+channel]))
        }
        var rgba = [UInt8](repeating: 255, count: size.width * size.height * 4)
        for y in 0..<size.height { for x in 0..<size.width {
            let sx = (Float(x)+0.5)*Float(width)/Float(size.width)-0.5
            let sy = (Float(y)+0.5)*Float(height)/Float(size.height)-0.5
            let luma = sample(yBytes,yStride,width,height,sx,sy,1,0)
            let ux = (sx-chromaOrigin.x)*Float(uvWidth)/Float(width)
            let uy = (sy-chromaOrigin.y)*Float(uvHeight)/Float(height)
            let cb = (sample(uvBytes,uvStride,uvWidth,uvHeight,ux,uy,2,0)-128)/(full ? 255 : 224)
            let cr = (sample(uvBytes,uvStride,uvWidth,uvHeight,ux,uy,2,1)-128)/(full ? 255 : 224)
            let value = (luma-(full ? 0 : 16))/(full ? 255 : 219)
            let rgb = SIMD3(value+coefficients.x*cr, value+coefficients.y*cb+coefficients.z*cr, value+coefficients.w*cb)
            let offset = (y*size.width+x)*4
            for channel in 0..<3 {
                var encoded = min(1,max(0,rgb[channel]))
                if rec709Transfer {
                    let linear = encoded < 0.081 ? encoded/4.5 : pow((encoded+0.099)/1.099,1/0.45)
                    encoded = linear <= 0.0031308 ? 12.92*linear : 1.055*pow(linear,1/2.4)-0.055
                }
                rgba[offset+channel] = UInt8((min(1,max(0,encoded))*255).rounded())
            }
        } }
        return try CaptureColorImage(width: size.width, height: size.height, rgba: rgba)
    }

    static func depth(_ buffer: CVPixelBuffer, confidence: CVPixelBuffer?) throws -> CaptureDepthMap {
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        guard width > 0, height > 0, width <= 1024, height <= 1024,
              CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_DepthFloat32 else { throw HexelsError.invalid("Unsupported scene-depth layout") }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer), CVPixelBufferGetBytesPerRow(buffer) >= width * 4 else { throw HexelsError.invalid("Invalid depth stride") }
        var meters = [Float](repeating: 0, count: width*height)
        for y in 0..<height { for x in 0..<width {
            meters[y*width+x] = base.loadUnaligned(fromByteOffset: y*CVPixelBufferGetBytesPerRow(buffer)+x*4, as: Float.self)
        } }
        var values: [UInt8]?
        if let confidence {
            guard CVPixelBufferGetWidth(confidence) == width, CVPixelBufferGetHeight(confidence) == height,
                  CVPixelBufferGetPixelFormatType(confidence) == kCVPixelFormatType_OneComponent8 else { throw HexelsError.invalid("Depth confidence shape differs") }
            CVPixelBufferLockBaseAddress(confidence, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(confidence, .readOnly) }
            guard let data = CVPixelBufferGetBaseAddress(confidence), CVPixelBufferGetBytesPerRow(confidence) >= width else { throw HexelsError.invalid("Invalid confidence stride") }
            values = (0..<width*height).map { data.load(fromByteOffset: ($0/width)*CVPixelBufferGetBytesPerRow(confidence)+$0%width, as: UInt8.self) }
        }
        return try CaptureDepthMap(width: width, height: height, meters: meters, confidence: values)
    }

    static func writePNG(_ image: CaptureColorImage, to url: URL) throws {
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let provider = CGDataProvider(data: Data(image.rgba) as CFData),
              let cgImage = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: image.width*4, space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { throw HexelsError.invalid("Cannot encode captured image") }
        CGImageDestinationAddImage(destination,cgImage,[kCGImagePropertyOrientation:1] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { throw HexelsError.invalid("Captured image could not be saved") }
    }
}
