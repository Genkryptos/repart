# RePart Capture (iPhone)

A small camera app that tells you when the shot is good. It tracks the A4 sheet live
(Apple Vision), works out the phone's tilt and distance from the sheet's corners (the
same plane-homography idea the server uses), and turns the outline green when:

| Check    | Photo 1          | Photo 2                  |
|----------|------------------|--------------------------|
| Sheet    | found            | found                    |
| Corners  | all four in view | all four in view         |
| Part     | something dark on the sheet | same          |
| Tilt     | 12–32° (≈20°)    | 38–58° (≈45°), another side |
| Distance | 20–50 cm         | 20–50 cm                 |
| Light    | sheet not too dark | same                   |
| Steady   | outline still for ~0.5 s | same             |

Hold green for 0.8 s and it takes the photo itself (the shutter works any time). Then
name the part (type, or tap the mic: Muse Voice Transcribe via the server), pick the
material and tap **Make the part**: the photos go to the RePart server on your Mac and
the app opens RePart's page for that job, which shows the fit live and the 3D result.

## Run it

1. On the Mac, serve RePart on the Wi-Fi (not just localhost):

   ```bash
   uv run repart serve --host 0.0.0.0 --port 8000
   ```

2. Open `ios/RePartCapture.xcodeproj` in Xcode, select the **RePartCapture** target ›
   Signing & Capabilities, choose your team (a free Personal Team works; the app then
   lasts 7 days per install). Change the bundle id if Xcode says it's taken.
3. Plug in the iPhone, pick it as the run destination, press Run. First time only: on
   the phone, Settings › General › VPN & Device Management › trust your developer
   profile, and turn on Developer Mode if asked.
4. In the app's review screen, set **RePart server** to `http://<your Mac's Wi-Fi IP>:8000`
   (`ipconfig getifaddr en0`). Allow Local Network access when iOS asks.

The simulator has no camera; use the photo button to import photos and see the same checks.
