import CoreVideo
import ImageIO
import UIKit
import Vision

/// What one camera frame (or imported photo) says about the shot.
struct Reading {
    var quad: Quad?
    var pose: SheetPose?
    /// Median brightness of the sheet, 0–255.
    var sheetTone: Double = 0
    /// Share of the sheet covered by something clearly darker: the part.
    var partShare: Double = 0
    /// Upright image aspect (width / height), for drawing the overlay.
    var aspect: CGFloat = 3.0 / 4.0
}

enum FrameAnalyzer {
    /// A live frame from the back camera held in portrait (sensor buffers are landscape).
    static func analyze(_ pb: CVPixelBuffer, focalPx: Double) -> Reading {
        let bw = CVPixelBufferGetWidthOfPlane(pb, 0), bh = CVPixelBufferGetHeightOfPlane(pb, 0)
        var r = Reading(aspect: CGFloat(bh) / CGFloat(bw))
        guard let quad = detectSheet(VNImageRequestHandler(cvPixelBuffer: pb, orientation: .right)) else { return r }
        r.quad = quad
        r.pose = SheetGeometry.pose(quad, imageSize: CGSize(width: bh, height: bw), focalPx: focalPx)
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pb, 0)?.assumingMemoryBound(to: UInt8.self) else { return r }
        let bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        // Upright (u, v) is the buffer rotated 90° clockwise: buffer x = v·width, y = (1 − u)·height.
        (r.sheetTone, r.partShare) = sample(quad) { p in
            let x = min(bw - 1, max(0, Int(p.y * CGFloat(bw))))
            let y = min(bh - 1, max(0, Int((1 - p.x) * CGFloat(bh))))
            return Double(base[y * bpr + x])
        }
        return r
    }

    /// An imported photo. Focal length comes from its EXIF, else a typical 26 mm main camera.
    static func analyze(photo data: Data) -> Reading {
        guard let image = UIImage(data: data) else { return Reading() }
        let f35 = focal35(data) ?? 26
        let scale = 1600 / max(image.size.width, image.size.height)
        let size = CGSize(width: (image.size.width * scale).rounded(), height: (image.size.height * scale).rounded())
        let w = Int(size.width), h = Int(size.height)
        // Upright greyscale copy: Vision runs on it and its bytes are the luma samples.
        var grey = [UInt8](repeating: 0, count: w * h)
        let cg: CGImage? = grey.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                                      space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)
            else { return nil }
            UIGraphicsPushContext(ctx)
            ctx.translateBy(x: 0, y: CGFloat(h))
            ctx.scaleBy(x: 1, y: -1)
            image.draw(in: CGRect(origin: .zero, size: size))
            UIGraphicsPopContext()
            return ctx.makeImage()
        }
        var r = Reading(aspect: size.width / size.height)
        guard let cg, let quad = detectSheet(VNImageRequestHandler(cgImage: cg, orientation: .up)) else { return r }
        r.quad = quad
        let focalPx = f35 * hypot(size.width, size.height) / 43.2666
        r.pose = SheetGeometry.pose(quad, imageSize: size, focalPx: focalPx)
        (r.sheetTone, r.partShare) = sample(quad) { p in
            Double(grey[min(h - 1, max(0, Int(p.y * CGFloat(h)))) * w + min(w - 1, max(0, Int(p.x * CGFloat(w))))])
        }
        return r
    }

    static func focal35(_ data: Data) -> Double? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any],
              let f = exif[kCGImagePropertyExifFocalLenIn35mmFilm] as? Double, f > 0 else { return nil }
        return f
    }

    /// The largest quadrilateral in view; perspective makes the sheet a trapezoid, hence the loose angles.
    private static func detectSheet(_ handler: VNImageRequestHandler) -> Quad? {
        let req = VNDetectRectanglesRequest()
        req.minimumAspectRatio = 0.3
        req.maximumAspectRatio = 1.0
        req.quadratureTolerance = 40
        req.minimumSize = 0.2
        req.minimumConfidence = 0.5
        req.maximumObservations = 4
        try? handler.perform([req])
        func flip(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x, y: 1 - p.y) }  // Vision's origin is bottom-left
        return (req.results ?? [])
            .map { Quad(tl: flip($0.topLeft), tr: flip($0.topRight), br: flip($0.bottomRight), bl: flip($0.bottomLeft)) }
            .max { $0.area < $1.area }
    }

    /// Sheet tone (median of a grid inside the sheet) and the share of samples clearly darker than it.
    private static func sample(_ q: Quad, luma: (CGPoint) -> Double) -> (Double, Double) {
        var vals: [Double] = []
        vals.reserveCapacity(40 * 28)
        for i in 0..<40 {
            for j in 0..<28 {
                vals.append(luma(q.point(s: 0.05 + 0.9 * CGFloat(i) / 39, t: 0.05 + 0.9 * CGFloat(j) / 27)))
            }
        }
        vals.sort()
        let tone = vals[vals.count / 2]
        let dark = vals.filter { $0 < tone - max(40, 0.25 * tone) }.count
        return (tone, Double(dark) / Double(vals.count))
    }
}
