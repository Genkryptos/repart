import AVFoundation
import UIKit

/// Live camera: every ~80 ms a frame is analysed for the sheet, and once every check has
/// stayed green for a moment the photo is taken automatically.
final class CameraModel: NSObject, ObservableObject {
    enum Status { case starting, running, denied, unavailable }

    let session = AVCaptureSession()
    @Published var status = Status.starting
    @Published var reading = Reading()
    @Published var guidance = Guidance()
    /// 0…1 while all checks are green; the photo is taken at 1.
    @Published var hold: Double = 0
    var onPhoto: ((Data, Reading) -> Void)?

    static let holdSeconds = 0.8
    private let queue = DispatchQueue(label: "repart.camera")
    private let photoOutput = AVCapturePhotoOutput()
    private let videoOutput = AVCaptureVideoDataOutput()
    private var focalPx = 0.0
    private var formatWidth = 1.0
    private var configured = false
    // Touched only on `queue`.
    private var shot = Shot.first
    private var recent: [Quad] = []
    private var greenSince: Date?
    private var lastFrame = Date.distantPast
    private var capturing = false
    private var paused = false
    private var lastReading = Reading()

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: queue.async { self.configureAndRun() }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { ok in
                if ok { self.queue.async { self.configureAndRun() } } else { self.publish(status: .denied) }
            }
        default: publish(status: .denied)
        }
    }

    func stop() { queue.async { if self.session.isRunning { self.session.stopRunning() } } }

    func set(shot: Shot) {
        queue.async { self.shot = shot; self.greenSince = nil; self.paused = false }
    }

    /// Shutter button: take the photo now, whatever the checks say.
    func capture() { queue.async { self.takePhoto() } }

    private func publish(status: Status) { DispatchQueue.main.async { self.status = status } }

    private func configureAndRun() {
        if !configured {
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: device) else { return publish(status: .unavailable) }
            session.beginConfiguration()
            session.sessionPreset = .photo  // 4:3 frames, the same framing as the photo
            guard session.canAddInput(input), session.canAddOutput(photoOutput), session.canAddOutput(videoOutput) else {
                session.commitConfiguration()
                return publish(status: .unavailable)
            }
            session.addInput(input)
            session.addOutput(photoOutput)
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.setSampleBufferDelegate(self, queue: queue)
            session.addOutput(videoOutput)
            if let largest = device.activeFormat.supportedMaxPhotoDimensions.max(by: { $0.width * $0.height < $1.width * $1.height }) {
                photoOutput.maxPhotoDimensions = largest
            }
            if let c = photoOutput.connection(with: .video), c.isVideoRotationAngleSupported(90) { c.videoRotationAngle = 90 }
            session.commitConfiguration()
            // Focal length in pixels of the (landscape) video frame, from the horizontal field of view.
            let dims = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
            let fov = Double(device.activeFormat.videoFieldOfView) * .pi / 180
            formatWidth = Double(dims.width)
            focalPx = formatWidth / 2 / tan(fov / 2)
            configured = true
        }
        session.startRunning()
        publish(status: .running)
    }

    private func takePhoto() {
        guard !capturing, session.isRunning else { return }
        capturing = true
        let settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])
        settings.maxPhotoDimensions = photoOutput.maxPhotoDimensions
        photoOutput.capturePhoto(with: settings, delegate: self)
    }
}

extension CameraModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let now = Date()
        guard !capturing, !paused, now.timeIntervalSince(lastFrame) > 0.08,
              let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lastFrame = now
        // The frame may be smaller than the format the field of view was measured on.
        let frameFocal = focalPx * Double(CVPixelBufferGetWidth(pb)) / formatWidth
        let r = FrameAnalyzer.analyze(pb, focalPx: frameFocal)
        if let q = r.quad { recent = Array((recent + [q]).suffix(6)) } else { recent = [] }
        let steady = recent.count >= 4 && zip(recent, recent.dropFirst()).allSatisfy { $1.moved(from: $0) < 0.012 }
        let g = Guidance.evaluate(r, shot: shot, steady: steady)
        var hold = 0.0
        if g.allGood {
            greenSince = greenSince ?? now
            hold = min(1, now.timeIntervalSince(greenSince!) / Self.holdSeconds)
        } else {
            greenSince = nil
        }
        lastReading = r
        if hold >= 1 { takePhoto() }
        DispatchQueue.main.async {
            self.reading = r
            self.guidance = g
            self.hold = hold
        }
    }
}

extension CameraModel: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        let data = photo.fileDataRepresentation()
        queue.async {
            let reading = self.lastReading
            self.capturing = false
            self.greenSince = nil
            self.recent = []
            if data != nil { self.paused = true }  // until the next shot is set
            DispatchQueue.main.async {
                self.hold = 0
                if let data { self.onPhoto?(data, reading) }
            }
        }
    }
}
