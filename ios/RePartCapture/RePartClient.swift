import AVFoundation
import Foundation

/// Talks to the RePart server on the Mac: uploads the photos as a job, transcribes speech.
struct RePartClient {
    var server: String
    var accessCode: String

    struct ServerError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    var base: URL? { URL(string: server.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "/"))) }

    /// The web page that follows the fit live and shows the finished part.
    func jobPage(_ id: String) -> URL? {
        guard let base, var c = URLComponents(url: base, resolvingAgainstBaseURL: false) else { return nil }
        c.queryItems = [URLQueryItem(name: "job", value: id)] + (accessCode.isEmpty ? [] : [URLQueryItem(name: "code", value: accessCode)])
        return c.url
    }

    func createJob(photos: [Data], part: String, describe: String, answers: String, material: String) async throws -> String {
        var form = Multipart()
        for (i, data) in photos.enumerated() { form.file("photos", name: "photo\(i + 1).\(Self.ext(data))", data: data) }
        form.field("part", part)
        form.field("describe", describe)
        form.field("answers", answers)
        form.field("material", material)
        let body: [String: Any] = try await post("api/jobs", form)
        guard let id = body["id"] as? String else { throw ServerError(message: "The server didn't return a job.") }
        return id
    }

    /// Muse Voice Transcribe, through the server (it holds the API key).
    func transcribe(_ audio: URL) async throws -> String {
        var form = Multipart()
        form.file("audio", name: "speech.m4a", data: try Data(contentsOf: audio), type: "audio/mp4")
        let body: [String: Any] = try await post("api/transcribe", form)
        return (body["text"] as? String) ?? ""
    }

    private func post(_ path: String, _ form: Multipart) async throws -> [String: Any] {
        guard let url = base?.appendingPathComponent(path) else { throw ServerError(message: "Set the server address first.") }
        var req = URLRequest(url: url, timeoutInterval: 60)
        req.httpMethod = "POST"
        req.setValue("multipart/form-data; boundary=\(form.boundary)", forHTTPHeaderField: "Content-Type")
        if !accessCode.isEmpty { req.setValue(accessCode, forHTTPHeaderField: "X-Access-Code") }
        let data: Data, resp: URLResponse
        do {
            (data, resp) = try await URLSession.shared.upload(for: req, from: form.finish())
        } catch {
            throw ServerError(message: "Can't reach RePart at \(server). Is the Mac on the same Wi‑Fi and the server running?")
        }
        let body = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        if let http = resp as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw ServerError(message: (body["detail"] as? String) ?? "Server error \(http.statusCode).")
        }
        return body
    }

    /// The server goes by file extension; keep the camera's original bytes (and EXIF focal length).
    static func ext(_ d: Data) -> String {
        if d.starts(with: [0xFF, 0xD8]) { return "jpg" }
        if d.starts(with: [0x89, 0x50, 0x4E, 0x47]) { return "png" }
        return "heic"
    }
}

private struct Multipart {
    let boundary = "repart-\(UUID().uuidString)"
    private var body = Data()

    mutating func field(_ key: String, _ value: String) {
        body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(key)\"\r\n\r\n\(value)\r\n".data(using: .utf8)!)
    }

    mutating func file(_ key: String, name: String, data: Data, type: String = "image/jpeg") {
        body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(key)\"; filename=\"\(name)\"\r\nContent-Type: \(type)\r\n\r\n".data(using: .utf8)!)
        body.append(data)
        body.append("\r\n".data(using: .utf8)!)
    }

    func finish() -> Data { body + "--\(boundary)--\r\n".data(using: .utf8)! }
}

/// Tap to record, tap again to stop; returns the clip for transcription.
final class VoiceRecorder: ObservableObject {
    @Published var recordingField: String?
    private var recorder: AVAudioRecorder?
    private let url = FileManager.default.temporaryDirectory.appendingPathComponent("speech.m4a")

    func start(for field: String) async -> Bool {
        guard await AVAudioApplication.requestRecordPermission() else { return false }
        do {
            try AVAudioSession.sharedInstance().setCategory(.record, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            let r = try AVAudioRecorder(url: url, settings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 16_000, AVNumberOfChannelsKey: 1,
            ])
            r.record(forDuration: 90)
            recorder = r
            await MainActor.run { recordingField = field }
            return true
        } catch {
            return false
        }
    }

    func stop() -> URL? {
        recorder?.stop()
        recorder = nil
        recordingField = nil
        try? AVAudioSession.sharedInstance().setActive(false)
        return url
    }
}
