import Foundation

/// The two shots RePart wants: a gentle tilt, then a steep one from another side.
enum Shot: Int, CaseIterable {
    case first, second

    var tilt: ClosedRange<Double> { self == .first ? 12...32 : 38...58 }
    var idealTilt: Int { self == .first ? 20 : 45 }
    var title: String { self == .first ? "Photo 1 of 2 · gentle tilt" : "Photo 2 of 2 · steep, from another side" }
    var instructions: String {
        self == .first
            ? "Part on the white A4 sheet, darker surface underneath. Hold the phone about 30 cm away, tilted ~20°."
            : "Walk to another side of the sheet and tilt the phone more, ~45°. Keep all four corners in view."
    }
}

struct Check: Identifiable {
    let id: String
    let ok: Bool
}

struct Guidance {
    var checks: [Check] = []
    var hint = "Point the camera at the white A4 sheet"
    var allGood: Bool { !checks.isEmpty && checks.allSatisfy(\.ok) }

    static let names = ["Sheet", "Corners", "Part", "Tilt", "Distance", "Light", "Steady"]

    /// Each check in order; the hint is the first one that fails.
    static func evaluate(_ r: Reading, shot: Shot, steady: Bool) -> Guidance {
        guard let q = r.quad else {
            return Guidance(checks: names.map { Check(id: $0, ok: false) }, hint: "Point the camera at the white A4 sheet")
        }
        let tilt = r.pose?.tiltDeg ?? 0, dist = r.pose?.distanceMM ?? 0
        let want = "aim for ~\(shot.idealTilt)°"
        let rules: [(String, Bool, String)] = [
            ("Sheet", true, ""),
            ("Corners", q.edgeMargin > 0.03, "Move back: all four sheet corners must be in view"),
            ("Part", r.partShare > 0.004, "Put the part on the sheet"),
            ("Tilt", r.pose != nil && shot.tilt.contains(tilt),
             tilt < shot.tilt.lowerBound
                ? "Tilt the phone more: \(Int(tilt))°, \(want)" + (shot == .second ? ", from another side" : "")
                : "Too steep: \(Int(tilt))°, \(want)"),
            ("Distance", (200...500).contains(dist),
             dist < 200 ? "Move back a little (\(Int(dist / 10)) cm)" : "Move closer (\(Int(dist / 10)) cm, aim for ~30)"),
            ("Light", r.sheetTone > 105, "Too dark: add light, avoid your own shadow"),
            ("Steady", steady, "Hold still…"),
        ]
        let checks = rules.map { Check(id: $0.0, ok: $0.1) }
        let hint = rules.first { !$0.1 }?.2 ?? "Perfect, hold it there"
        return Guidance(checks: checks, hint: hint)
    }
}
