import CoreGraphics
import simd

/// The A4 sheet's corners in normalized image coordinates (origin top-left, upright image).
struct Quad: Equatable {
    var tl: CGPoint, tr: CGPoint, br: CGPoint, bl: CGPoint

    var points: [CGPoint] { [tl, tr, br, bl] }

    /// Smallest distance from any corner to the frame edge (0 = touching).
    var edgeMargin: CGFloat { points.map { min($0.x, $0.y, 1 - $0.x, 1 - $0.y) }.min() ?? 0 }

    var area: CGFloat {
        let p = points
        return abs((0..<4).reduce(0) { $0 + p[$1].x * p[($1 + 1) % 4].y - p[($1 + 1) % 4].x * p[$1].y }) / 2
    }

    func moved(from o: Quad) -> CGFloat {
        zip(points, o.points).map { hypot($0.x - $1.x, $0.y - $1.y) }.max() ?? 0
    }

    /// Bilinear point inside the quad: s along tl→tr, t along tl→bl.
    func point(s: CGFloat, t: CGFloat) -> CGPoint {
        let top = CGPoint(x: tl.x + (tr.x - tl.x) * s, y: tl.y + (tr.y - tl.y) * s)
        let bottom = CGPoint(x: bl.x + (br.x - bl.x) * s, y: bl.y + (br.y - bl.y) * s)
        return CGPoint(x: top.x + (bottom.x - top.x) * t, y: top.y + (bottom.y - top.y) * t)
    }
}

struct SheetPose {
    /// Angle between the camera's view direction and straight down onto the sheet.
    var tiltDeg: Double
    /// Camera to the centre of the sheet.
    var distanceMM: Double
}

/// Camera pose from the four sheet corners, the same idea as the server's solve_camera:
/// a plane homography, decomposed with the known focal length. The sheet's long side can
/// land either way round in the image, so both assignments are tried and the one whose
/// rotation is most nearly orthonormal wins.
enum SheetGeometry {
    static let a4 = (long: 297.0, short: 210.0)

    static func pose(_ q: Quad, imageSize: CGSize, focalPx: Double) -> SheetPose? {
        let w = Double(imageSize.width), h = Double(imageSize.height)
        let img = q.points.map { SIMD2((Double($0.x) - 0.5) * w / focalPx, (Double($0.y) - 0.5) * h / focalPx) }
        var best: (score: Double, pose: SheetPose)?
        for (sw, sh) in [(a4.long, a4.short), (a4.short, a4.long)] {
            let world = [SIMD2(0, 0), SIMD2(sw, 0), SIMD2(sw, sh), SIMD2(0, sh)]
            guard let H = homography(from: world, to: img) else { continue }
            let h1 = H.columns.0, h2 = H.columns.1, h3 = H.columns.2
            let n1 = length(h1), n2 = length(h2)
            guard n1 > 1e-12, n2 > 1e-12 else { continue }
            var lam = 2 / (n1 + n2)
            if h3.z < 0 { lam = -lam }  // sheet in front of the camera
            let r1 = lam * h1, r2 = lam * h2, t = lam * h3
            let r3 = normalize(cross(r1, r2))
            let score = abs(dot(h1 / n1, h2 / n2)) + abs(n1 / n2 - 1)
            let centre = t + r1 * (sw / 2) + r2 * (sh / 2)
            let pose = SheetPose(tiltDeg: acos(min(1, abs(r3.z))) * 180 / .pi, distanceMM: length(centre))
            if best == nil || score < best!.score { best = (score, pose) }
        }
        return best?.pose
    }

    /// 4-point DLT with h33 = 1. Columns of the result are H's columns.
    static func homography(from src: [SIMD2<Double>], to dst: [SIMD2<Double>]) -> simd_double3x3? {
        var a = [[Double]](repeating: [Double](repeating: 0, count: 9), count: 8)
        for i in 0..<4 {
            let (x, y, u, v) = (src[i].x, src[i].y, dst[i].x, dst[i].y)
            a[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y, u]
            a[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y, v]
        }
        guard let h = solve(&a) else { return nil }
        return simd_double3x3(columns: (SIMD3(h[0], h[3], h[6]), SIMD3(h[1], h[4], h[7]), SIMD3(h[2], h[5], 1)))
    }

    /// Gaussian elimination with partial pivoting on an n×(n+1) augmented matrix.
    private static func solve(_ a: inout [[Double]]) -> [Double]? {
        let n = a.count
        for c in 0..<n {
            guard let p = (c..<n).max(by: { abs(a[$0][c]) < abs(a[$1][c]) }), abs(a[p][c]) > 1e-12 else { return nil }
            a.swapAt(c, p)
            for r in 0..<n where r != c {
                let f = a[r][c] / a[c][c]
                if f != 0 { for k in c...n { a[r][k] -= f * a[c][k] } }
            }
        }
        return (0..<n).map { a[$0][n] / a[$0][$0] }
    }
}
