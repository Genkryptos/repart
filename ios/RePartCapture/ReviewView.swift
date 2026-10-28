import SafariServices
import SwiftUI

/// Both photos, what the part is, then upload; the fit is followed on RePart's own page.
struct ReviewView: View {
    let photos: [Captured]
    let retake: (Shot) -> Void

    @AppStorage("server") private var server = "http://192.168.1.10:8000"
    @AppStorage("accessCode") private var accessCode = ""
    @State private var part = ""
    @State private var describe = ""
    @State private var answers = ""
    @State private var material = "PETG"
    @State private var busy = false
    @State private var error: String?
    @State private var jobPage: JobPage?
    @StateObject private var voice = VoiceRecorder()
    @FocusState private var focused: String?

    private var client: RePartClient { RePartClient(server: server, accessCode: accessCode) }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    HStack(spacing: 12) {
                        ForEach(Array(photos.enumerated()), id: \.offset) { i, p in thumb(p, shot: Shot(rawValue: i)!) }
                        if photos.count == 1 {
                            Button { retake(.second) } label: {
                                VStack { Image(systemName: "plus"); Text("Photo 2").font(.caption) }
                                    .frame(maxWidth: .infinity, minHeight: 120)
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                } footer: {
                    Text("Two photos from different sides give the most accurate part.")
                }

                Section("What is it? (1–6 words)") {
                    field("part", "black plastic knob", text: $part)
                }
                Section("What does it do?") {
                    field("describe", "Fan regulator knob. The D-shaped hole fits the shaft.", text: $describe, long: true)
                }
                Section("Measurements the photo can't show") {
                    field("answers", "Shaft hole 6 mm with a flat, 10 mm deep", text: $answers, long: true)
                }
                Section {
                    Picker("Print material", selection: $material) {
                        ForEach(["PETG", "PLA", "ABS", "Nylon", "TPU"], id: \.self) { Text($0) }
                    }
                }
                Section("RePart server") {
                    TextField("http://192.168.1.10:8000", text: $server)
                        .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                    SecureField("Access code (if set)", text: $accessCode)
                }
                if let error {
                    Section { Text(error).foregroundStyle(.red) }
                }
            }
            .navigationTitle("Your part")
            .scrollDismissesKeyboard(.interactively)
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { focused = nil }
                }
            }
            .safeAreaInset(edge: .bottom) {
                if focused == nil { makeButton }
            }
            .fullScreenCover(item: $jobPage) { SafariView(url: $0.url).ignoresSafeArea() }
        }
    }

    private var makeButton: some View {
        Button(action: upload) {
            HStack {
                if busy { ProgressView().tint(.black) }
                Text(busy ? "Uploading…" : "Make the part").font(.headline)
            }
            .frame(maxWidth: .infinity).padding()
            .background(Color.green, in: RoundedRectangle(cornerRadius: 14))
            .foregroundStyle(.black)
        }
        .disabled(busy)
        .padding()
        .background(.bar)
    }

    private func thumb(_ p: Captured, shot: Shot) -> some View {
        VStack(spacing: 4) {
            Image(uiImage: p.image).resizable().scaledToFill()
                .frame(maxWidth: .infinity, minHeight: 120, maxHeight: 120).clipped()
                .clipShape(RoundedRectangle(cornerRadius: 10))
            Text(p.reading.pose.map { "\(Int($0.tiltDeg))° · \(Int($0.distanceMM / 10)) cm" } ?? "sheet not found")
                .font(.caption.monospacedDigit())
                .foregroundStyle(p.reading.pose != nil && shot.tilt.contains(p.reading.pose!.tiltDeg) ? Color.green : .orange)
            Button("Retake") { retake(shot) }.font(.caption)
        }
        .frame(maxWidth: .infinity)
    }

    private func field(_ id: String, _ prompt: String, text: Binding<String>, long: Bool = false) -> some View {
        HStack(alignment: .top) {
            if long {
                TextField(prompt, text: text, axis: .vertical).lineLimit(2...5).focused($focused, equals: id)
            } else {
                TextField(prompt, text: text).focused($focused, equals: id)
            }
            Button {
                Task { await toggleVoice(id, text) }
            } label: {
                Image(systemName: voice.recordingField == id ? "stop.circle.fill" : "mic.circle")
                    .font(.title2)
                    .foregroundStyle(voice.recordingField == id ? Color.red : .accentColor)
            }
            .buttonStyle(.borderless)
            .accessibilityLabel(voice.recordingField == id ? "Stop recording" : "Speak")
        }
    }

    private func toggleVoice(_ id: String, _ text: Binding<String>) async {
        if voice.recordingField == id {
            guard let url = voice.stop() else { return }
            do {
                let said = try await client.transcribe(url)
                let sep = id == "answers" ? "\n" : " "
                text.wrappedValue = [text.wrappedValue.trimmingCharacters(in: .whitespaces), said].filter { !$0.isEmpty }.joined(separator: sep)
            } catch {
                self.error = error.localizedDescription
            }
        } else {
            if voice.recordingField != nil { _ = voice.stop() }
            if !(await voice.start(for: id)) { error = "Microphone unavailable. Allow it in Settings › RePart." }
        }
    }

    private func upload() {
        error = nil
        let name = part.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { error = "Say what the part is, like 'black plastic knob'."; return }
        busy = true
        Task {
            defer { busy = false }
            do {
                let id = try await client.createJob(photos: photos.map(\.data), part: name, describe: describe,
                                                    answers: answers, material: material)
                if let url = client.jobPage(id) { jobPage = JobPage(url: url) }
            } catch {
                self.error = error.localizedDescription
            }
        }
    }
}

struct JobPage: Identifiable {
    let url: URL
    var id: String { url.absoluteString }
}

struct SafariView: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> SFSafariViewController { SFSafariViewController(url: url) }
    func updateUIViewController(_ vc: SFSafariViewController, context: Context) {}
}
