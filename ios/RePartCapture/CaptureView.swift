import AVFoundation
import PhotosUI
import SwiftUI

struct Captured {
    let data: Data
    let image: UIImage
    let reading: Reading
}

/// Live viewfinder with the sheet outlined: amber while something is off, green when the shot is good.
struct CaptureView: View {
    @ObservedObject var camera: CameraModel
    let shot: Shot
    let previous: Captured?
    let onCapture: (Captured) -> Void
    let onSkip: (() -> Void)?

    @State private var picked: PhotosPickerItem?
    @State private var imported: Captured?
    @State private var flash = false

    private var reading: Reading { imported?.reading ?? camera.reading }
    private var guidance: Guidance { imported.map { Guidance.evaluate($0.reading, shot: shot, steady: true) } ?? camera.guidance }
    private var live: Bool { imported == nil && camera.status == .running }

    var body: some View {
        VStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Image("Logo").resizable().frame(width: 26, height: 26).clipShape(RoundedRectangle(cornerRadius: 6))
                    Text("RePart").font(.title3.weight(.semibold))
                }
                .padding(.bottom, 4)
                Text(shot.title).font(.headline)
                Text(shot.instructions).font(.subheadline).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal)

            viewfinder
                .aspectRatio(reading.aspect, contentMode: .fit)
                .frame(maxWidth: .infinity)
                .overlay(Color.white.opacity(flash ? 0.8 : 0).allowsHitTesting(false))

            checksRow.padding(.horizontal)
            Text(guidance.hint)
                .font(.title3.weight(.semibold))
                .foregroundStyle(guidance.allGood ? Color.green : .primary)
                .multilineTextAlignment(.center)
                .frame(minHeight: 56)
                .padding(.horizontal)
            TiltGauge(tilt: reading.pose?.tiltDeg, target: shot.tilt).padding(.horizontal)
            Spacer(minLength: 0)
            controls.padding(.horizontal, 24).padding(.bottom, 8)
        }
        .padding(.top, 8)
        .onChange(of: picked) { _, item in
            guard let item else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self), let image = UIImage(data: data) {
                    imported = Captured(data: data, image: image, reading: FrameAnalyzer.analyze(photo: data))
                }
                picked = nil
            }
        }
    }

    @ViewBuilder private var viewfinder: some View {
        ZStack {
            if let imported {
                Image(uiImage: imported.image).resizable().scaledToFit()
            } else if camera.status == .running || camera.status == .starting {
                CameraPreview(session: camera.session)
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "camera.fill").font(.largeTitle)
                    Text(camera.status == .denied ? "Camera access is off. Turn it on in Settings › RePart, or import a photo."
                                                  : "No camera here. Import a photo to check it.")
                        .multilineTextAlignment(.center)
                }
                .foregroundStyle(.secondary)
                .padding()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(white: 0.12))
            }
            SheetOverlay(quad: reading.quad, good: guidance.allGood, hold: live ? camera.hold : 0)
        }
        .clipped()
    }

    private var checksRow: some View {
        HStack(spacing: 4) {
            ForEach(guidance.checks.isEmpty ? Guidance.names.map { Check(id: $0, ok: false) } : guidance.checks) { c in
                HStack(spacing: 2) {
                    Image(systemName: c.ok ? "checkmark.circle.fill" : "circle")
                    Text(c.id)
                }
                .font(.caption2)
                .foregroundStyle(c.ok ? Color.green : .secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .frame(maxWidth: .infinity)
            }
        }
    }

    private var controls: some View {
        HStack {
            PhotosPicker(selection: $picked, matching: .images) {
                Image(systemName: "photo.on.rectangle").font(.title2).frame(width: 64, height: 64)
            }
            .accessibilityLabel("Import a photo")
            Spacer()
            if let imported {
                Button {
                    onCapture(imported)
                    self.imported = nil
                } label: {
                    Label("Use photo", systemImage: "checkmark").font(.headline)
                        .padding(.horizontal, 20).padding(.vertical, 14)
                        .background(guidance.allGood ? Color.green : Color.orange, in: Capsule())
                        .foregroundStyle(.black)
                }
            } else {
                ShutterButton(progress: camera.hold, good: guidance.allGood) { camera.capture() }
                    .disabled(camera.status != .running)
            }
            Spacer()
            Group {
                if imported != nil {
                    Button { imported = nil } label: { Image(systemName: "xmark").font(.title2) }
                        .accessibilityLabel("Discard imported photo")
                } else if let onSkip {
                    Button("Skip", action: onSkip).font(.subheadline)
                } else if let previous {
                    Image(uiImage: previous.image).resizable().scaledToFill().frame(width: 48, height: 48).clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    Color.clear
                }
            }
            .frame(width: 64, height: 64)
        }
        .onAppear {
            camera.onPhoto = { data, r in
                guard let image = UIImage(data: data) else { return }
                UINotificationFeedbackGenerator().notificationOccurred(.success)
                withAnimation(.easeOut(duration: 0.25)) { flash = true }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { withAnimation { flash = false } }
                onCapture(Captured(data: data, image: image, reading: r))
            }
        }
    }
}

/// The detected sheet outline, drawn in the same aspect-fit frame as the image.
struct SheetOverlay: View {
    let quad: Quad?
    let good: Bool
    let hold: Double

    var body: some View {
        GeometryReader { geo in
            let s = geo.size
            ZStack {
                RoundedRectangle(cornerRadius: 4)
                    .strokeBorder(good ? Color.green : Color.clear, lineWidth: 6)
                if let q = quad {
                    let pts = q.points.map { CGPoint(x: $0.x * s.width, y: $0.y * s.height) }
                    Path { p in p.addLines(pts); p.closeSubpath() }
                        .fill((good ? Color.green : Color.orange).opacity(0.18 + 0.2 * hold))
                    Path { p in p.addLines(pts); p.closeSubpath() }
                        .stroke(good ? Color.green : Color.orange, style: StrokeStyle(lineWidth: 4, lineJoin: .round))
                    ForEach(0..<4, id: \.self) { i in
                        Circle().fill(good ? Color.green : Color.orange).frame(width: 14, height: 14).position(pts[i])
                    }
                }
            }
            .animation(.easeOut(duration: 0.12), value: quad)
        }
        .allowsHitTesting(false)
    }
}

/// Current tilt against the band this shot wants.
struct TiltGauge: View {
    let tilt: Double?
    let target: ClosedRange<Double>
    private let span = 0.0...70.0

    var body: some View {
        VStack(spacing: 4) {
            GeometryReader { geo in
                let w = geo.size.width
                let x = { (d: Double) in CGFloat((d - span.lowerBound) / (span.upperBound - span.lowerBound)) * w }
                ZStack(alignment: .leading) {
                    Capsule().fill(Color(white: 0.25)).frame(height: 8)
                    Capsule().fill(Color.green.opacity(0.6))
                        .frame(width: x(target.upperBound) - x(target.lowerBound), height: 8)
                        .offset(x: x(target.lowerBound))
                    if let tilt {
                        let on = target.contains(tilt)
                        Circle().fill(on ? Color.green : Color.orange).frame(width: 18, height: 18)
                            .offset(x: min(w, max(0, x(tilt))) - 9)
                    }
                }
                .frame(maxHeight: .infinity)
            }
            .frame(height: 20)
            HStack {
                Text("flat").font(.caption2).foregroundStyle(.secondary)
                Spacer()
                Text(tilt.map { "tilt \(Int($0))°" } ?? "tilt –").font(.caption.monospacedDigit())
                Spacer()
                Text("70°").font(.caption2).foregroundStyle(.secondary)
            }
        }
    }
}

struct ShutterButton: View {
    let progress: Double
    let good: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle().stroke(Color.white.opacity(0.35), lineWidth: 5)
                Circle().trim(from: 0, to: progress).stroke(Color.green, style: StrokeStyle(lineWidth: 5, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                Circle().fill(good ? Color.green : Color.white).padding(9)
            }
            .frame(width: 76, height: 76)
        }
        .accessibilityLabel("Take photo")
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var preview: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.preview.session = session
        v.preview.videoGravity = .resizeAspect  // show exactly what the photo will contain
        if let c = v.preview.connection, c.isVideoRotationAngleSupported(90) { c.videoRotationAngle = 90 }
        return v
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        // The connection only exists once the session has its input.
        if let c = uiView.preview.connection, c.isVideoRotationAngleSupported(90) { c.videoRotationAngle = 90 }
    }
}
