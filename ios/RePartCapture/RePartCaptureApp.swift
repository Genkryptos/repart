import SwiftUI

@main
struct RePartCaptureApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView().preferredColorScheme(.dark)
        }
    }
}

/// Photo 1 → photo 2 → review and upload. Retaking either shot comes back to the camera.
struct ContentView: View {
    @StateObject private var camera = CameraModel()
    @State private var photos: [Captured] = []
    @State private var shot = Shot.first
    @State private var reviewing = false

    var body: some View {
        Group {
            if reviewing {
                ReviewView(photos: photos) { s in
                    photos = Array(photos.prefix(s.rawValue))  // retaking photo 1 retakes both
                    go(to: s)
                }
            } else {
                CaptureView(camera: camera, shot: shot, previous: photos.first,
                            onCapture: accept,
                            onSkip: shot == .second ? { review() } : nil)
                    .id(shot)
            }
        }
        .onAppear { camera.start() }
        .onChange(of: reviewing) { _, r in r ? camera.stop() : camera.start() }
    }

    private func accept(_ c: Captured) {
        photos = Array(photos.prefix(shot.rawValue)) + [c]
        if shot == .first { go(to: .second) } else { review() }
    }

    private func go(to s: Shot) {
        shot = s
        camera.set(shot: s)
        reviewing = false
    }

    private func review() { reviewing = true }
}
