# TrueBeat browser extension (1.3.0)

A Chrome extension that checks the video playing in the current tab for AI
generation and face-swaps, using **your own TrueBeat backend**.

The extension captures and uploads. It never decides anything itself: the
verdict, the headline, the thresholds, the evidence numbers, the warnings
and the caveat all come from the backend (`trubeat serve`), so there is
exactly one implementation of the physics and one set of thresholds.

---

## What it does

1. You click **"Check this tab"**. The extension injects its panel into the
   page (there is no content script on other pages) and looks at every
   frame of the tab for `<video>` elements, iframes and open shadow roots
   included. A page Chrome keeps extensions out of (`chrome://` pages, the
   Chrome Web Store) is not recorded at all: its result would have nowhere
   to show, and the popup says so.
2. It picks the **largest playing video** (not paused, with a frame
   decoded); a paused banner no longer blocks a smaller video that plays.
   It records only what is left of that clip, less a second of start-up,
   so a short AI clip is not recorded across its loop seam or frozen on its
   last frame. A paused, ended or still-loading video is not recorded: the
   panel says why.
3. It records the tab in an offscreen document **at the tab's own size in
   device pixels** (its viewport times the screen's pixel ratio), scaled
   down to fit 1920×1080 with its shape kept, at up to 30 frames per
   second, as WebM at about 6 Mb/s for 1080p (proportionally less for
   smaller tabs, never under 1 Mb/s). The panel shows the size measured on
   the frames. (Before 1.3.0 Chrome picked the size: 960×540 on a 1080p
   screen.) If the browser refuses that size, the tab is recorded at
   Chrome's default size instead.
   If the page could not be read (for example a page that forbids
   scripting), the old behaviour applies: Chrome's default size, the whole
   tab.
4. While recording it checks, through a `<video>` element playing the tab
   stream, that the video's area (the whole tab when the video's box is
   unknown) is not uniformly black: a DRM-protected video is drawn black,
   even inside a page that is not. Three black samples in a row (about
   2 s) stop the recording with "Can't capture this video: protected
   stream (DRM) or black frames", and nothing is uploaded. (A video that
   is itself black for 2 s, such as a long fade, stops it too: check again
   a moment later.) A recording whose tab capture ended early, whose
   recorder failed, which stopped on its own, or which is shorter than 3 s
   is not uploaded either; the panel says which.
5. It uploads the recording to your backend, then reads the video's box
   again. If the video moved or changed size by more than 10% of its size
   during the recording (scrolled, resized, fullscreen toggled), or went
   away, the box no longer marks what was recorded, so the whole tab is
   analysed and the panel says so (as it does when the page can no longer
   be read).
6. It asks the backend to analyse **only the video's area**: the box,
   measured in the page's CSS pixels, is converted into the recording's
   pixels (scaled by recorded width / viewport width, and height alike,
   clamped to the frame) and sent with the call to `/analyze_video_region`.
   A backend from before 1.3.0 has only `/analyze_video`; it gets the whole
   tab and the panel says so, as it does when the box cannot be mapped onto
   the recording (a recording whose shape differs from the tab's). A video
   inside an iframe is placed when exactly one iframe on the page matches
   its frame's size (within 2 px); otherwise (and for videos nested two
   frames deep, in CSS-scaled frames, on a pinch-zoomed page, or in frames
   Chrome does not let the extension read: with `activeTab` alone that is
   typically a frame from another site, such as an embedded player) the
   whole tab is analysed.
7. A panel in the bottom-right of the page shows the backend's headline,
   detail, evidence rows (AI-video detector, face-swap detector, pulse),
   the heart rate (marked "not reliable: pulse not measurable" when the
   backend says so), the backend's warnings (e.g. `W_TRUNCATED`,
   `W_PARTIAL_COVERAGE`, `W_GRAPHICS_SKIPPED`, `W_FEW_FACES`,
   `W_UNSTABLE_TRACK`), what was recorded and which part was analysed, and
   the backend's caveat. The panel follows the page into fullscreen once
   the recording has ended (while recording it stays out of a fullscreen
   player, so it is never recorded over the video), and screen readers
   hear the verdict (a polite live region).

One check runs per tab, and the tab stays locked until its result or an
error arrives; one tab is recorded at a time.

**Tip:** the bigger the video is on screen, the more pixels its faces get:
use theater mode or a large window. Going fullscreen *during* a recording
moves the video, so that check analyses the whole tab (the panel says so,
and in fullscreen it appears once the recording ends). The AI-video check
looks at the video's whole area and needs no face.

**Verdicts** come from the backend (`app.verdict_for`); the panel shows its
headline as sent. There are three states:

| Headline (the backend's) | State | Colour | When |
|---|---|---|---|
| `No sign of AI generation found` | `real` | green | the whole-frame AI-video detector ran and nothing fired. Not proof the video is real |
| `AI-generated: declared by its Content Credentials` | `fake` | red | a C2PA manifest whose signature checks out declares AI generation |
| `Likely AI-generated` | `fake` | red | the AI-video detector is over its threshold |
| `Possible face-swap` | `fake` | red | the strict pulse face-swap check fired (or the face-swap detector, if its bundle is decisive; today it is evidence only, shown in its own row) |
| `Inconclusive: could not read the video` / `Inconclusive: no check could reach a decision` | `inconclusive` | amber | no check could decide, e.g. the clip is under 2 s, unreadable, or the AI-video model is not installed |

The caveat is always on screen, as the backend words it. Today's:

> Checks for fully AI-generated video with a whole-frame detector trained
> on 21 generator families. On held-out test sets it caught about 4 in 5
> clips from 2024-era tools (Sora, Kling, Runway, Luma, Pika), about half
> from the newest (Veo 3.1, Sora 2, Kling 2.6) and from some open-source
> ones, and wrongly flagged about 5% of real test videos
> (reports/aigen_detector_v2.md). Clips under 2 s are not checked. It also
> reads C2PA Content Credentials when present. Face-swaps: a weak pulse
> check can decide (it catches about a quarter of Deepfakes-style swaps and
> is blind to reenactment fakes such as Face2Face and NeuralTextures); a
> face-swap detector trained on FaceForensics++ is shown as evidence only,
> because it wrongly flagged 38% of real faces from other sources
> (docs/12). 'No sign of AI generation' is not proof a video is real.

## Install (load unpacked)

1. Start the backend on this machine:

   ```bash
   cd <trubeat-repo>
   uv pip install -e ".[serving]"
   trubeat serve          # http://127.0.0.1:7860
   ```

2. Get the extension folder:
   - **Download the packaged zip** (recommended):
     <https://github.com/luckyDuckBuck/trubeat/releases/latest/download/trubeat-extension.zip>: unzip it to get an `extension/` folder; or
   - work from a checkout: the folder is [`extension/`](./) in the
     [tool repository](https://github.com/luckyDuckBuck/trubeat).

3. Load it:
   - open `chrome://extensions`
   - turn on **Developer mode** (top right)
   - click **Load unpacked**
   - pick the `extension/` folder

4. Click the TrueBeat icon in the toolbar. The dot next to the title turns
   green when the backend is reachable; the line under the button says
   whether it can analyse just the video (a backend from before 1.3.0
   analyses the whole tab). Use **Options** to change the backend URL or
   the capture length.

> Chrome may ask for permission to capture the tab when you press the
> button: that is the `tabCapture` permission doing its job.

## Options

| Setting | Default | Notes |
|---------|---------|-------|
| Backend URL | `http://127.0.0.1:7860` | any TrueBeat backend that exposes `analyze_video_region` (1.3.0+) or `analyze_video` |
| Capture length | 20 s | 12–60 s, cut to what is left of the clip. The AI-video check and the face-swap detector need 2 s; the pulse face-swap check needs 10 s |

For a **hosted** backend (e.g. an `https://*.hf.space` URL) the extension asks
Chrome for permission to reach that origin when you save. Local addresses
need no extra permission.

## Permissions

| Permission | Why |
|---|---|
| `activeTab` | when you click the button, access to that one tab: to read where its video is and to draw the panel. Nothing runs in any page you have not clicked the button on |
| `scripting` | to inject the panel (`overlay.js`) and the video probe into that tab |
| `tabCapture` | to record the tab |
| `offscreen` | the recorder runs in an offscreen document (a service worker cannot record) |
| `storage` | your two settings (`chrome.storage.local`) and the per-tab lock (`chrome.storage.session`, in memory only) |
| host `http://127.0.0.1/*`, `http://localhost/*` | the default local backend; any other backend origin is optional and asked for when you save it |

Since 1.3.0 there is **no content script and no web-accessible resource**:
1.2 injected its panel script into every page you opened, which made Chrome
warn that the extension could "read and change all your data on all
websites" and let pages detect it. The panel is now injected only into the
tab you check.

## Privacy

- **No telemetry, no analytics, no other network traffic.** The only host
  the extension ever talks to is the backend URL you set.
- **What is stored where:**
  - the extension keeps the recording in memory only until the upload
    finishes, then drops it. Nothing is written to disk by the extension;
  - your backend keeps the uploaded recording while it analyses it and
    deletes it as soon as that analysis returns (`app.build_demo`,
    `analyze_json`);
  - a recording that is uploaded but never analysed (a call that failed),
    and every video uploaded through the web app, is swept from the
    backend's upload cache after an hour (Gradio's `delete_cache`, every
    5 minutes);
  - the extension stores your two settings in `chrome.storage.local` and,
    while a check runs, the tab's id, a check id and timestamps in
    `chrome.storage.session` (memory only, cleared when the browser
    closes). Nothing about the video itself is kept.
- A local backend keeps everything on your machine.

## How it works

| File | Role |
|------|------|
| `manifest.json` | MV3 manifest. Default `host_permissions` are localhost only; other origins are *optional* (requested at save time). No content scripts. |
| `background.js` | Service worker: owns the per-tab check lifecycle, probes every frame for the video to record (`probeFrame`, `chooseVideo`), plans the length, re-checks the video's box when the recording stops, injects the panel and relays state into it. It never holds a long network wait (a service worker idles out after ~30 s) and never computes a verdict. |
| `offscreen.html` / `offscreen.js` | The recorder and the whole backend conversation. `getUserMedia` + `MediaRecorder` are unavailable in a service worker, so the recording happens here, at the tab's size. It runs the **black-frame guard** (frames drawn through a `<video>` playing the tab stream), refuses failed or partial recordings, converts the video's box into recording pixels, uploads (a Blob cannot survive a runtime message), and waits for the analysis: a service worker would be idle-killed mid-stream, exactly while the backend is computing. |
| `overlay.js` | The bottom-right panel, injected on click and rendered in a closed shadow DOM so the page cannot restyle it. |
| `popup.html` / `popup.js` | "Check this tab" button + backend status dot. |
| `options.html` / `options.js` | Backend URL, capture length, "Test connection". |
| `icons/` | 16/32/48/128 PNGs (regenerate with `python scripts/make_icons.py`). |

**Why tab capture, not `<video>` scraping:** it works on any site (YouTube,
news, social) without depending on the page's markup or fetching the video
from anywhere, and it captures what is actually playing. The page is only
read for where its video is and how long it has left. The extension also
refuses to fake a result: a black, failed or partial recording is never
uploaded.

### The backend contract

The extension speaks the Gradio API (Gradio 6):

```
GET  {backend}/gradio_api/info                               -> named_endpoints
POST {backend}/gradio_api/upload  (multipart "files")        -> ["<server path>"]
POST {backend}/gradio_api/call/analyze_video_region          -> {"event_id": "..."}
       {"data": [{"path": "<server path>", "meta": {"_type": "gradio.FileData"}},
                 "{\"x0\":150,\"y0\":120,\"x1\":1110,\"y1\":660}"]}
POST {backend}/gradio_api/call/analyze_video                 -> {"event_id": "..."}
       {"data": [{"path": "<server path>", "meta": {"_type": "gradio.FileData"}}]}
GET  {backend}/gradio_api/call/<api>/<event_id>              -> server-sent events
```

- **Endpoint choice:** with a known video box and a backend that lists
  `/analyze_video_region`, the box goes there as JSON
  `{"x0":..,"y0":..,"x1":..,"y1":..}` in the recording's pixels (`""` means
  the whole frame). Otherwise `/analyze_video` (one input) is called and
  the whole tab is analysed.
- **The event stream:** Gradio sends `event: <name>` lines followed by
  `data: <json>`. `complete` carries `["<json string>"]`, the result;
  `error` carries `{"error": "<message>"}` (the backend launches with
  `show_error=True`; `null` when a backend hides it) or a bare string, and
  ends the wait with that message; `heartbeat` (about every 15 s) and
  `generating` are ignored, except that every byte proves the backend is
  alive.
- **Waiting:** 90 s without a single byte (heartbeats count) means the
  stream is dead; the whole wait is capped at 6 minutes, mid-stream
  included.
- **A dropped stream is re-POSTed, never re-read:** after a disconnect
  Gradio forgets the job, and reading its event again never brings the
  result. The extension POSTs the call again on the same uploaded file (up
  to 3 attempts in all). If the first, abandoned run has meanwhile finished
  and deleted the recording, the re-run cannot read it, and the panel says
  that instead of showing "could not read the video".

The result JSON has exactly these fields (`app.api_payload`,
`tests/js/api_keys.json`): `verdict`, `headline`, `detail`, `score`,
`hr_bpm`, `region_agreement_bpm`, `snr`, `pulse_measurable`,
`face_valid_frac`, `ai_label`, `ai_real_percentile`, `ai_score`,
`ai_threshold`, `decided_by`, `fsd_mode`, `fsd_label`, `fsd_score`,
`fsd_threshold`, `fsd_target_fpr`, `fsd_outside_fpr`,
`fsd_real_percentile`, `fsd_n_faces`, `fsd_n_sampled`, `fsd_top_time_s`,
`layers`, `provenance`, `duration_s`, `warnings`, `per_roi_hr`, `qc` and
`caveat`. It is produced by `src/trubeat/app.py` from the *same*
`analyze_video()` the web app uses: no thresholds are duplicated or
recalibrated on the client.

## Manual QA checklist

Run through this after loading the extension.

**A. Happy path**
- [ ] `trubeat serve` is running; the popup dot is green and says "Backend ready."
- [ ] `chrome://extensions` → TrueBeat → Details: no "read and change all
      your data on all websites" line (no content script since 1.3.0).
- [ ] Open a page with a real face (a talking-head video), press play.
- [ ] Press **Check this tab**. The panel shows "Capturing… Ns" counting up
      and "Recording the tab at W×H px" (on a 1080p screen with a
      maximised window, about the window's size in device pixels).
- [ ] The panel then shows one of the verdicts with the evidence rows, a
      "recorded" row ("… tab; analysed the video's W×H px area"), any
      warnings, and the caveat.
- [ ] The same clip uploaded to the web app gives the **same verdict and the
      same numbers** (the extension and the app are the same pipeline; the
      web app analyses the whole frame, so compare with a clip that fills
      it). To check this without a browser, run the extension's own client
      against the backend and diff it against the app's analysis:

      ```bash
      node tests/js/extension_print_result.mjs http://127.0.0.1:7860 clip.mp4
      python -c "from trubeat.app import analyze_video; print(analyze_video('clip.mp4').to_json())"
      ```

      Add a region as a third argument (`'{"x0":0,"y0":0,"x1":640,"y1":360}'`)
      to exercise `/analyze_video_region`.

**B. Where the video is**
- [ ] A page with a large paused video and a smaller playing one: the
      smaller one is analysed (the "recorded" row gives its size).
- [ ] An embedded player (an iframe) that plays: its area is analysed when
      it is the only frame of its size on the page.
- [ ] Scroll the page during the capture: the panel says the video moved
      and the whole tab was analysed.
- [ ] Start a check, then go fullscreen with the player's button: the panel
      is not drawn over the video while it records, then appears in
      fullscreen with "Uploading…", "Analysing…" and the result.

**C. Error states (all must be honest, none may invent a verdict)**
- [ ] **Backend stopped** → the panel says the backend is unreachable
      ("Is `trubeat serve` running?").
- [ ] **Capture denied** (click Cancel on Chrome's capture bar) → an honest
      "can't capture" message, no verdict, and the next click works.
- [ ] **Paused video** → "The video is paused. Press play, then check again."
- [ ] **Tab closed or reloaded mid-capture** → nothing is uploaded; the next
      click works.
- [ ] **DRM stream** (a Netflix/DRM-protected player) → "Can't capture this
      video: protected stream (DRM) or black frames", and **nothing is
      uploaded** (DevTools → Network on the offscreen document: no upload).
- [ ] **Backend error** (an exception in the analysis) → "The backend
      reported an error: …" within seconds, never "Analysing…" forever.
- [ ] **Backend killed mid-analysis** → the call is POSTed again; when the
      attempts run out, an honest connection message, never a raw network
      error.
- [ ] A second click on the same tab while it is being checked → the popup
      says a check is already running; after the result, a click works.

**D. Privacy**
- [ ] DevTools → Network: while checking a tab, the **only** requests are to
      the configured backend. No third-party hosts, no analytics.
- [ ] `chrome://extensions` → service worker → Console: no errors.
- [ ] After a check, the backend's upload folder (`/tmp/gradio/…` by
      default) no longer holds the capture.
- [ ] Reload the extension: no captured video appears anywhere on disk
      (`chrome.storage.local` holds only the two settings).

**E. Settings**
- [ ] Options → change the backend URL to a wrong one → "Test connection"
      reports it is unreachable; the popup dot turns red.
- [ ] Set a hosted backend URL → Chrome asks for permission for that origin;
      declining keeps the extension working but unable to reach it.

## Automated tests

```bash
node tests/js/extension_test.mjs                                   # offline checks
node tests/js/extension_test.mjs --backend http://127.0.0.1:7860   # + live checks (opt-in; small clip via --video)
```

This loads the shipped extension files and drives the real code paths on a
virtual clock: the probe and the choice of video, the capture size and the
region conversion, the black-frame guard reading frames through a `<video>`,
failed and partial captures, the per-tab lock, the panel's injection and
rendering (fullscreen included), the analysis wait on Gradio's real event
shapes (heartbeats, `event: error`, a silent stream, the deadline, the
re-POST after a drop), the whole flow end to end on a message bus that
delivers like Chrome, a real HTTP backend that drops its connection, and a
static audit for permissions, remote code, inline scripts, telemetry hosts
and any request that does not go to the configured backend. It also runs
under `pytest` (`tests/test_extension.py`).
