/**
 * TrueBeat extension — offscreen recorder and backend client.
 *
 * Runs in the offscreen document because getUserMedia/MediaRecorder are
 * unavailable in an MV3 service worker. It:
 *   - opens the tab-capture stream (chromeMediaSource: 'tab') at the tab's
 *     own size in device pixels, scaled down to fit 1920x1080, at most
 *     30 frames per second (1.3.0: Chrome's default was 960x540 on a
 *     1080p screen, which left a presenter's face ~30 px wide),
 *   - records it to WebM with a bitrate matched to that size,
 *   - converts the playing video's box from CSS pixels into the
 *     recording's pixels, so the backend analyses only the video (1.3.0),
 *   - samples frames through a <video> element to detect a uniformly black
 *     capture (protected/DRM streams and some players produce black frames),
 *   - reports live progress to the service worker,
 *   - refuses to upload a recording that failed part-way or is too short,
 *   - uploads the finished clip itself, because a Blob cannot cross a
 *     runtime message (it would arrive in the worker as {}),
 *   - runs the whole backend conversation afterwards too: endpoint
 *     discovery, the analysis call and the SSE wait (a service worker would
 *     be idle-killed mid-stream), reporting ANALYSIS_RESULT with the verdict
 *     or an honest CAPTURE_ERROR.
 *
 * Network: only the backend URL the service worker passes in (the one set
 * in Options); nothing else is contacted. The recording is never stored
 * here: its bytes live only until the upload POST finishes, and the backend
 * deletes its own copy once the analysis returns.
 */

// ------------------------------------------------------------- constants

const CAPTURE_FILENAME = "truebeat-capture.webm";
// The AI-video detector needs one 2 s segment; keep a margin (the same
// floor as background.js). A shorter recording is never uploaded.
const MIN_CAPTURE_SECONDS = 3;

// Capture size (docs/13 §3.4): the tab's viewport in device pixels, scaled
// down to fit 1920x1080 with its shape kept, at most 30 frames per second.
const MAX_CAPTURE_WIDTH = 1920;
const MAX_CAPTURE_HEIGHT = 1080;
const MAX_FRAME_RATE = 30;
// Encoder budget: 6 Mb/s at 1080p, in proportion to the pixel count below
// that, never under 1 Mb/s.
const BITRATE_AT_1080P = 6_000_000;
const MIN_BITRATE = 1_000_000;
// The backend ignores a region under 16 px a side (app.parse_region).
const MIN_REGION_PX = 16;
// A CSS-pixel box maps onto the recording only when the recording kept the
// viewport's shape; past 2% the whole tab is analysed instead of guessing.
const ASPECT_TOLERANCE = 0.02;

const PROGRESS_INTERVAL_MS = 500;
// Black-frame guard. A small copy of the frame is enough for the mean and
// spread of its brightness.
const SAMPLE_INTERVAL_MS = 700;
const SAMPLE_WIDTH = 160;
// A frame counts as "black" when it is both very dark and very flat.
const BLACK_LUMA_MAX = 12;
const BLACK_STD_MAX = 6;
// Consecutive black samples before we give up on the capture.
const BLACK_STRIKES = 3;
// How long a stopped recorder may take to hand over its file.
const STOP_GRACE_MS = 5000;

// The backend conversation. Gradio sends `event: heartbeat` about every
// 15 s while an analysis runs, so 90 s without a single byte means the
// stream is dead. After a disconnect Gradio forgets the job (re-reading its
// event never brings the result), so a dropped wait POSTs the call again on
// the same uploaded file, which the backend keeps until an analysis of it
// returns. The deadline bounds the whole wait, mid-stream included.
const INFO_TIMEOUT_MS = 30_000;
const CALL_TIMEOUT_MS = 30_000;
const UPLOAD_TIMEOUT_MS = 5 * 60 * 1000;
const SSE_IDLE_MS = 90_000;
const ANALYZE_DEADLINE_MS = 6 * 60 * 1000;
const ANALYZE_ATTEMPTS = 3; // the first call plus two re-POSTs after a drop
const RETRY_DELAY_MS = 1500;
const MAX_ERROR_CHARS = 500;

// ------------------------------------------------------------ pure helpers

function errText(err) {
  return String((err && err.message) || err || "unknown error");
}

function isBlackFrame(mean, std) {
  return mean < BLACK_LUMA_MAX && std < BLACK_STD_MAX;
}

/** Mean and standard deviation of Rec. 601 luma over RGBA pixels. */
function lumaStats(data) {
  let sum = 0;
  let sumSq = 0;
  let n = 0;
  for (let i = 0; i + 2 < data.length; i += 4) {
    const y = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    sum += y;
    sumSq += y * y;
    n += 1;
  }
  if (!n) return null;
  const mean = sum / n;
  return { mean, std: Math.sqrt(Math.max(0, sumSq / n - mean * mean)) };
}

/**
 * The recording size for a tab: its viewport in device pixels, scaled down
 * to fit 1920x1080 with the shape kept, in even numbers (what VP8/VP9 and
 * 4:2:0 chroma want). null when the viewport is unknown.
 */
function captureSize(view) {
  if (!view) return null;
  const dpr = Number(view.devicePixelRatio) > 0 ? Number(view.devicePixelRatio) : 1;
  const w = Number(view.innerWidth) * dpr;
  const h = Number(view.innerHeight) * dpr;
  if (!Number.isFinite(w) || !Number.isFinite(h) || w < 2 || h < 2) return null;
  const k = Math.min(1, MAX_CAPTURE_WIDTH / w, MAX_CAPTURE_HEIGHT / h);
  const even = (v) => Math.max(2, 2 * Math.floor(Math.round(v) / 2));
  return { width: even(w * k), height: even(h * k) };
}

/** Encoder bitrate for a recording size: 6 Mb/s at 1080p, proportionally less. */
function bitrateFor(width, height) {
  const px = Number(width) * Number(height);
  if (!(px > 0)) return 0;
  return Math.max(
    MIN_BITRATE,
    Math.round((BITRATE_AT_1080P * px) / (MAX_CAPTURE_WIDTH * MAX_CAPTURE_HEIGHT))
  );
}

/**
 * The video's box, measured in the tab's CSS pixels, in the recording's
 * pixels: scaled by capture width / innerWidth (and height), clamped to the
 * frame. null when it cannot be mapped (unknown sizes, a recording whose
 * shape differs from the viewport's, or a box under 16 px a side).
 */
function regionToCapture(rect, view, capture) {
  if (!rect || !view || !capture) return null;
  const vw = Number(view.innerWidth);
  const vh = Number(view.innerHeight);
  const cw = Number(capture.width);
  const ch = Number(capture.height);
  if (!(vw > 0 && vh > 0 && cw > 0 && ch > 0)) return null;
  const sx = cw / vw;
  const sy = ch / vh;
  if (Math.abs(sx - sy) > ASPECT_TOLERANCE * Math.max(sx, sy)) return null;
  const x = Number(rect.x);
  const y = Number(rect.y);
  const w = Number(rect.width);
  const h = Number(rect.height);
  if (![x, y, w, h].every(Number.isFinite) || w <= 0 || h <= 0) return null;
  const clampTo = (v, max) => Math.min(max, Math.max(0, v));
  const x0 = clampTo(Math.floor(x * sx), cw);
  const y0 = clampTo(Math.floor(y * sy), ch);
  const x1 = clampTo(Math.ceil((x + w) * sx), cw);
  const y1 = clampTo(Math.ceil((y + h) * sy), ch);
  if (x1 - x0 < MIN_REGION_PX || y1 - y0 < MIN_REGION_PX) return null;
  return { x0, y0, x1, y1 };
}

/** A region as the backend takes it, or null. */
function validRegion(r) {
  if (!r || typeof r !== "object") return null;
  const v = ["x0", "y0", "x1", "y1"].map((k) => Number(r[k]));
  if (!v.every(Number.isFinite)) return null;
  const [x0, y0, x1, y1] = v.map((n) => Math.round(n));
  if (x1 - x0 < MIN_REGION_PX || y1 - y0 < MIN_REGION_PX) return null;
  return { x0, y0, x1, y1 };
}

/**
 * getUserMedia constraints for the tab stream. With a known size the
 * recording is pinned to it (min = max) at up to 30 fps; without one the
 * browser picks its default size (the 1.2 behaviour).
 */
function captureConstraints(streamId, size) {
  const mandatory = { chromeMediaSource: "tab", chromeMediaSourceId: streamId };
  if (size) {
    mandatory.minWidth = size.width;
    mandatory.maxWidth = size.width;
    mandatory.minHeight = size.height;
    mandatory.maxHeight = size.height;
    mandatory.maxFrameRate = MAX_FRAME_RATE;
  }
  return { audio: false, video: { mandatory } };
}

/**
 * Which backend endpoint to call. A backend that can crop gets the region
 * through /analyze_video_region ([file, region JSON], "" = whole frame);
 * an older one gets /analyze_video ([file]) and analyses the whole tab.
 */
function pickEndpoint(endpoints, region) {
  const has = (name) => endpoints.includes(name);
  if (region && has("/analyze_video_region")) {
    return { apiName: "/analyze_video_region", withRegion: true };
  }
  if (has("/analyze_video")) return { apiName: "/analyze_video", withRegion: false };
  if (has("/analyze_video_region")) {
    return { apiName: "/analyze_video_region", withRegion: true };
  }
  const other = endpoints.find((e) => e.endsWith("analyze_video"));
  return other ? { apiName: other, withRegion: false } : null;
}

/**
 * Incremental server-sent-events parser: feed it decoded text, get back the
 * complete events ({event, data}). `event:` lines name the event (Gradio
 * sends complete / error / heartbeat / generating); a blank line ends it.
 */
function sseParser() {
  let buffer = "";
  let name = "";
  let data = [];
  const feed = (raw, out) => {
    const line = raw.endsWith("\r") ? raw.slice(0, -1) : raw;
    if (line === "") {
      if (data.length || name) out.push({ event: name || "message", data: data.join("\n") });
      name = "";
      data = [];
      return;
    }
    if (line.startsWith(":")) return; // a comment
    const colon = line.indexOf(":");
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") name = value;
    else if (field === "data") data.push(value);
  };
  return {
    push(text) {
      buffer += text;
      const lines = buffer.split("\n");
      buffer = lines.pop();
      const out = [];
      for (const l of lines) feed(l, out);
      return out;
    },
    /** The stream ended: flush an event that lacked its closing blank line. */
    end() {
      const out = [];
      if (buffer) feed(buffer, out);
      buffer = "";
      feed("", out);
      return out;
    },
  };
}

/** The readable text of a Gradio `event: error` payload. */
function backendErrorText(data) {
  let text = "";
  if (typeof data === "string") text = data;
  else if (data && typeof data === "object" && typeof data.error === "string") text = data.error;
  text = text.trim().slice(0, MAX_ERROR_CHARS);
  return text
    ? "The backend reported an error: " + text
    : "The backend reported an error but gave no details (see the terminal running `trubeat serve`).";
}

const UNREADABLE_REPLY = {
  verdict: "inconclusive",
  headline: "Unreadable backend reply",
  detail: "The backend answered, but not with TrueBeat's JSON.",
};

/** Gradio wraps the return value: ["<json string>"] -> the inner object. */
function resultFrom(data) {
  if (!Array.isArray(data) || !data.length) return { ...UNREADABLE_REPLY };
  const inner = data[0];
  if (typeof inner === "string") {
    try {
      const parsed = JSON.parse(inner);
      return parsed && typeof parsed === "object" ? parsed : { ...UNREADABLE_REPLY };
    } catch (_) {
      return { ...UNREADABLE_REPLY };
    }
  }
  return inner && typeof inner === "object" ? inner : { ...UNREADABLE_REPLY };
}

/** What one SSE event means for the wait: an outcome, or null to keep reading. */
function interpretEvent(ev) {
  if (ev.event === "heartbeat" || ev.event === "generating") return null;
  let data;
  let parsed = true;
  try {
    data = ev.data ? JSON.parse(ev.data) : null;
  } catch (_) {
    data = undefined;
    parsed = false;
  }
  const legacyError =
    ev.event === "message" && data && typeof data === "object" && !Array.isArray(data) && "error" in data;
  if (ev.event === "error" || legacyError) {
    return { outcome: "error", message: backendErrorText(parsed ? data : ev.data) };
  }
  if (ev.event === "complete" || (ev.event === "message" && Array.isArray(data))) {
    return { outcome: "result", verdict: resultFrom(data) };
  }
  return null;
}

// Exposed for the automated tests (tests/js/extension_test.mjs).
globalThis.__trueBeatIsBlackFrame = isBlackFrame;
globalThis.__trueBeatLumaStats = lumaStats;
globalThis.__trueBeatCaptureSize = captureSize;
globalThis.__trueBeatBitrate = bitrateFor;
globalThis.__trueBeatRegionToCapture = regionToCapture;
globalThis.__trueBeatCaptureConstraints = captureConstraints;
globalThis.__trueBeatPickEndpoint = pickEndpoint;
globalThis.__trueBeatSseParser = sseParser;

// --------------------------------------------------------------- messaging

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg) return;
  if (msg.type === "OFFSCREEN_START") {
    startCapture(msg)
      .then((info) => sendResponse({ ok: true, ...info }))
      .catch((err) => {
        const message = errText(err);
        post({
          type: "CAPTURE_ERROR",
          tabId: msg.tabId,
          checkId: msg.checkId ?? null,
          title: "Can't capture this tab",
          message,
        });
        sendResponse({ ok: false, error: message });
      });
    return true;
  }
  if (msg.type === "ANALYZE") {
    // Settle at once: the result (or an honest error) is posted as a
    // runtime message when the long analysis wait really ends.
    analyzeClip(msg).catch((err) => {
      post({
        type: "CAPTURE_ERROR",
        tabId: msg.tabId,
        checkId: msg.checkId ?? null,
        title: "Backend error",
        message: "The analysis failed inside the extension (" + errText(err) + ").",
      });
    });
    sendResponse({ ok: true });
    return true;
  }
});

function post(msg) {
  try {
    const sent = chrome.runtime.sendMessage(msg);
    if (sent && typeof sent.catch === "function") sent.catch(() => {});
  } catch (_) {
    /* the service worker is restarting: nothing to do */
  }
}

// ----------------------------------------------------------------- capture

/** The one recording in progress (one tab at a time). */
let activeCapture = null;

async function startCapture(msg) {
  // The offscreen page opens capture through the standard Web API (there is
  // no chrome-side wrapper for it in MV3). Fail with a readable message
  // when the API is missing instead of a raw TypeError.
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Tab capture is unavailable in this browser context.");
  }
  if (activeCapture) {
    throw new Error("Another tab is still being recorded. Wait for that check to finish, then try again.");
  }
  const cap = {
    tabId: msg.tabId,
    checkId: msg.checkId ?? null,
    // As planned by background.js (>= 3 s); a shorter plan is recorded and
    // then refused below rather than silently lengthened.
    seconds: Number(msg.seconds) > 0 ? Number(msg.seconds) : 20,
    backend: msg.backend || null,
    note: typeof msg.note === "string" ? msg.note : "",
    view: msg.view || null,
    rect: msg.rect || null,
    target: msg.target || null,
    size: null,
    seen: null,
    region: null,
    mimeType: "video/webm",
    chunks: [],
    stream: null,
    recorder: null,
    sampler: null,
    progressTimer: null,
    sampleTimer: null,
    stopTimer: null,
    graceTimer: null,
    startedAt: 0,
    blackStrikes: 0,
    black: false,
    failed: "",
    stopping: false,
    finished: false,
  };
  activeCapture = cap;
  try {
    const wanted = captureSize(cap.view);
    cap.stream = await openTabStream(msg.streamId, wanted);
    const track = cap.stream.getVideoTracks()[0];
    if (!track) throw new Error("The tab produced no video track.");
    const settings = (track.getSettings && track.getSettings()) || {};
    const width = Number(settings.width) || (wanted ? wanted.width : 0);
    const height = Number(settings.height) || (wanted ? wanted.height : 0);
    cap.size = width > 0 && height > 0 ? { width, height } : null;
    cap.region = cap.size ? regionToCapture(cap.rect, cap.view, cap.size) : null;
    track.addEventListener("ended", () => {
      failCapture(
        cap,
        "The tab stopped being captured before the recording finished (was it closed, reloaded or navigated away?)."
      );
    });

    const options = pickMimeType();
    if (options.mimeType) cap.mimeType = options.mimeType;
    if (wanted && cap.size) options.videoBitsPerSecond = bitrateFor(cap.size.width, cap.size.height);
    const recorder = new MediaRecorder(cap.stream, options);
    cap.recorder = recorder;
    recorder.ondataavailable = (e) => {
      if (e && e.data && e.data.size > 0) cap.chunks.push(e.data);
    };
    recorder.onerror = (e) => {
      failCapture(cap, "The recorder failed (" + errText(e && e.error) + ").");
    };
    recorder.onstop = () => finishCapture(cap);
    recorder.start(1000);
    cap.startedAt = Date.now();
  } catch (err) {
    teardown(cap);
    if (activeCapture === cap) activeCapture = null;
    throw err;
  }

  // Live progress (the plan note rides along, so it stays on screen) and
  // the black-frame guard.
  reportProgress(cap);
  cap.progressTimer = setInterval(() => reportProgress(cap), PROGRESS_INTERVAL_MS);
  startSampler(cap);
  cap.stopTimer = setTimeout(() => stopRecorder(cap), cap.seconds * 1000);
  // The start request must settle: background.js awaits this answer.
  return { size: cap.size, region: cap.region };
}

async function openTabStream(streamId, size) {
  try {
    return await navigator.mediaDevices.getUserMedia(captureConstraints(streamId, size));
  } catch (err) {
    if (size) {
      // A browser that rejects the size constraints still records at its
      // default size (what 1.2 did) rather than not at all.
      try {
        return await navigator.mediaDevices.getUserMedia(captureConstraints(streamId, null));
      } catch (_) {
        /* report the first refusal below */
      }
    }
    throw new Error("The browser refused the tab stream (" + errText(err) + ").");
  }
}

function reportProgress(cap) {
  if (cap.finished) return;
  const elapsed = cap.startedAt ? Math.min(cap.seconds, (Date.now() - cap.startedAt) / 1000) : 0;
  post({
    type: "PROGRESS",
    tabId: cap.tabId,
    checkId: cap.checkId,
    // the size measured on the frames once they arrive, else the one reported
    payload: { elapsed, total: cap.seconds, message: cap.note, size: cap.seen || cap.size },
  });
}

/**
 * The black-frame guard reads frames through a <video> element playing the
 * tab stream: a MediaStreamTrack is not something a canvas can draw.
 */
function startSampler(cap) {
  let video = null;
  try {
    video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.srcObject = cap.stream;
    if (document.body && typeof document.body.appendChild === "function") {
      document.body.appendChild(video);
    }
    const playing = typeof video.play === "function" ? video.play() : null;
    if (playing && typeof playing.catch === "function") playing.catch(() => {});
  } catch (_) {
    video = null;
  }
  if (!video) return;
  cap.sampler = { video, canvas: null, ctx: null };
  cap.sampleTimer = setInterval(() => sampleFrame(cap), SAMPLE_INTERVAL_MS);
}

/**
 * The part of the frame the guard looks at, in the <video>'s own pixels:
 * the video's area when it is known (that is what the backend analyses;
 * a protected video is drawn black inside a page that is not), else the
 * whole tab.
 */
function sampleArea(cap, video) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const r = cap.region;
  const size = cap.size;
  if (r && size && size.width > 0 && size.height > 0) {
    const kx = vw / size.width;
    const ky = vh / size.height;
    const sw = (r.x1 - r.x0) * kx;
    const sh = (r.y1 - r.y0) * ky;
    if (sw >= 1 && sh >= 1) return { sx: r.x0 * kx, sy: r.y0 * ky, sw, sh };
  }
  return { sx: 0, sy: 0, sw: vw, sh: vh };
}

/**
 * Sample the current frame and look for a uniformly black image.
 * DRM-protected streams and a few players hand the capturer black frames;
 * uploading those would produce a meaningless "no face" answer, so we stop
 * and say so honestly instead.
 */
function sampleFrame(cap) {
  const s = cap.sampler;
  if (!s || cap.finished || cap.stopping) return;
  const video = s.video;
  if (!(video.readyState >= 2) || !(video.videoWidth > 0) || !(video.videoHeight > 0)) {
    return; // no frame decoded yet
  }
  // The size of the frames really delivered (the track's settings are only
  // what was asked for); finishCapture maps the region onto it.
  cap.seen = { width: video.videoWidth, height: video.videoHeight };
  if (!s.ctx) {
    s.canvas = document.createElement("canvas");
    s.ctx =
      s.canvas && typeof s.canvas.getContext === "function"
        ? s.canvas.getContext("2d", { willReadFrequently: true })
        : null;
    if (!s.ctx) return;
  }
  const area = sampleArea(cap, video);
  const w = SAMPLE_WIDTH;
  const h = Math.max(1, Math.round((w * area.sh) / area.sw));
  if (s.canvas.width !== w || s.canvas.height !== h) {
    s.canvas.width = w;
    s.canvas.height = h;
  }
  let stats = null;
  try {
    s.ctx.drawImage(video, area.sx, area.sy, area.sw, area.sh, 0, 0, w, h);
    stats = lumaStats(s.ctx.getImageData(0, 0, w, h).data);
  } catch (_) {
    return; // an unreadable sample is no evidence either way
  }
  if (!stats) return;
  cap.blackStrikes = isBlackFrame(stats.mean, stats.std) ? cap.blackStrikes + 1 : 0;
  if (cap.blackStrikes >= BLACK_STRIKES) {
    cap.black = true;
    stopRecorder(cap);
  }
}

/** Stop recording; the recorder's stop event delivers the file. */
function stopRecorder(cap) {
  if (cap.finished || cap.stopping) return;
  cap.stopping = true;
  clearTimers(cap);
  const recorder = cap.recorder;
  if (recorder && recorder.state !== "inactive") {
    try {
      recorder.stop();
      cap.graceTimer = setTimeout(() => {
        if (!cap.finished) {
          cap.failed = cap.failed || "The recorder did not hand over the recording.";
          finishCapture(cap);
        }
      }, STOP_GRACE_MS);
      return;
    } catch (_) {
      /* finish below */
    }
  }
  finishCapture(cap);
}

/** The tab's track ended early or the recorder failed: nothing is uploaded. */
function failCapture(cap, reason) {
  if (cap.finished || cap.stopping || cap.failed) return;
  cap.failed = reason;
  stopRecorder(cap);
}

/** Decide, once, what happens to a stopped recording. */
function finishCapture(cap) {
  if (cap.finished) return;
  cap.finished = true;
  const elapsed = cap.startedAt ? (Date.now() - cap.startedAt) / 1000 : 0;
  teardown(cap);
  if (activeCapture === cap) activeCapture = null;
  if (!cap.stopping && !cap.failed) {
    // Nobody asked the recorder to stop: what it holds is a fragment.
    cap.failed = `The recording stopped on its own after ${elapsed.toFixed(1)} s of the planned ${cap.seconds} s.`;
  }
  const nothing = " Nothing was uploaded.";
  if (cap.failed) {
    reportCaptureError(cap, "Can't capture this tab", cap.failed + nothing);
    return;
  }
  if (cap.black) {
    reportCaptureError(
      cap,
      "Can't capture this video",
      "Can't capture this video: protected stream (DRM) or black frames." + nothing
    );
    return;
  }
  if (elapsed < MIN_CAPTURE_SECONDS) {
    reportCaptureError(
      cap,
      "Recording too short",
      `Only ${elapsed.toFixed(1)} s of the tab were recorded; at least ${MIN_CAPTURE_SECONDS} s are needed.` +
        nothing
    );
    return;
  }
  const blob = new Blob(cap.chunks, { type: cap.mimeType });
  cap.chunks = [];
  if (!blob.size) {
    reportCaptureError(cap, "Nothing was captured", "The recorder produced no data." + nothing);
    return;
  }
  const seen = cap.seen;
  if (seen && (!cap.size || seen.width !== cap.size.width || seen.height !== cap.size.height)) {
    // The browser delivered another size than it reported: map onto that.
    cap.size = { width: seen.width, height: seen.height };
    cap.region = regionToCapture(cap.rect, cap.view, cap.size);
  }
  // The service worker re-reads the video's box now, while the page still
  // shows what was recorded.
  post({ type: "CAPTURE_STOPPED", tabId: cap.tabId, checkId: cap.checkId });
  uploadCapture(cap, blob);
}

function reportCaptureError(cap, title, message) {
  post({ type: "CAPTURE_ERROR", tabId: cap.tabId, checkId: cap.checkId, title, message });
}

function clearTimers(cap) {
  if (cap.progressTimer) clearInterval(cap.progressTimer);
  if (cap.sampleTimer) clearInterval(cap.sampleTimer);
  if (cap.stopTimer) clearTimeout(cap.stopTimer);
  cap.progressTimer = null;
  cap.sampleTimer = null;
  cap.stopTimer = null;
}

function teardown(cap) {
  clearTimers(cap);
  if (cap.graceTimer) clearTimeout(cap.graceTimer);
  cap.graceTimer = null;
  if (cap.sampler) {
    const video = cap.sampler.video;
    try {
      if (typeof video.pause === "function") video.pause();
      video.srcObject = null;
      if (typeof video.remove === "function") video.remove();
    } catch (_) {
      /* best effort */
    }
    cap.sampler = null;
  }
  if (cap.stream) {
    try {
      cap.stream.getTracks().forEach((t) => t.stop());
    } catch (_) {
      /* best effort */
    }
    cap.stream = null;
  }
  cap.recorder = null;
}

/** Prefer a WebM container with VP8/VP9 so MediaRecorder works everywhere. */
function pickMimeType() {
  const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  for (const type of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(type)) {
      return { mimeType: type };
    }
  }
  return {};
}

// ------------------------------------------------------------------ upload

/**
 * Upload the finished recording and report the server-side path.
 * This runs here because the Blob lives here: a runtime message is
 * JSON-serialised, so a Blob would arrive in the service worker as {}.
 */
async function uploadCapture(cap, blob) {
  const base = cap.backend;
  if (!base) {
    reportCaptureError(cap, "Backend error", "No backend URL was given to the recorder.");
    return;
  }
  const form = new FormData();
  form.append("files", blob, CAPTURE_FILENAME);
  let res;
  try {
    res = await fetch(base + "/gradio_api/upload", {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(UPLOAD_TIMEOUT_MS),
    });
  } catch (err) {
    reportCaptureError(
      cap,
      "Backend error",
      "The backend was unreachable while storing the recording (" + errText(err) + ")."
    );
    return;
  }
  if (!res.ok) {
    reportCaptureError(
      cap,
      "Backend error",
      `The backend refused the recording (HTTP ${res.status}). Is \`trubeat serve\` running?`
    );
    return;
  }
  let uploaded = null;
  try {
    uploaded = await res.json();
  } catch (_) {
    reportCaptureError(cap, "Backend error", "The backend answered the upload with unreadable JSON.");
    return;
  }
  const serverPath = Array.isArray(uploaded) ? uploaded[0] : uploaded;
  if (typeof serverPath !== "string" || !serverPath) {
    reportCaptureError(cap, "Backend error", "The backend accepted the upload but returned no file path.");
    return;
  }
  post({
    type: "CAPTURE_UPLOADED",
    tabId: cap.tabId,
    checkId: cap.checkId,
    serverPath,
    filename: CAPTURE_FILENAME,
    capture: cap.size,
    region: cap.region,
    rect: cap.rect,
    view: cap.view,
    target: cap.target,
  });
}

// ---------------------------------------------------------------- analysis

/**
 * Analyse the uploaded copy of the clip and wait for the verdict.
 * This runs here because the wait itself lives here: a service worker is
 * idle-killed after ~30 s, and an in-flight analysis looks exactly like
 * idleness to it. Chrome's raw "network error" never reaches the overlay:
 * a dropped stream is retried by POSTing the call again, and only a
 * readable summary is ever posted.
 */
async function analyzeClip(msg) {
  const { tabId, serverPath } = msg;
  const checkId = msg.checkId ?? null;
  const base = msg.backend;
  const report = (m) => post({ tabId, checkId, ...m });
  const fail = (message) => report({ type: "CAPTURE_ERROR", title: "Backend error", message });

  if (!base || typeof serverPath !== "string" || !serverPath) {
    fail("The analysis request was incomplete.");
    return;
  }

  // 1. discover the endpoint (the user may point at any TrueBeat backend)
  let infoRes;
  try {
    infoRes = await fetch(base + "/gradio_api/info", {
      cache: "no-store",
      signal: AbortSignal.timeout(INFO_TIMEOUT_MS),
    });
  } catch (_) {
    fail(unreachableText(base));
    return;
  }
  if (!infoRes.ok) {
    fail(`The backend at ${base} answered HTTP ${infoRes.status}. Is \`trubeat serve\` running?`);
    return;
  }
  let info = null;
  try {
    info = await infoRes.json();
  } catch (_) {
    info = null;
  }
  const endpoints = Object.keys((info && info.named_endpoints) || {});
  const region = validRegion(msg.region);
  const plan = pickEndpoint(endpoints, region);
  if (!plan) {
    fail(
      "This backend does not expose the analyze_video endpoint " +
        `(found: ${endpoints.join(", ") || "none"}).`
    );
    return;
  }
  const file = { path: serverPath, meta: { _type: "gradio.FileData" } };
  const sentRegion = plan.withRegion ? region : null;
  const data = plan.withRegion ? [file, sentRegion ? JSON.stringify(sentRegion) : ""] : [file];
  let note = typeof msg.note === "string" ? msg.note : "";
  if (region && !plan.withRegion) {
    note = joinNotes(
      note,
      "This backend has no analyze_video_region endpoint (an older TrueBeat), so the whole tab was analysed."
    );
  }
  const capture =
    msg.capture && Number(msg.capture.width) > 0 && Number(msg.capture.height) > 0
      ? { width: Number(msg.capture.width), height: Number(msg.capture.height), region: sentRegion }
      : null;

  // 2. start the analysis; wait for it, POSTing again when the stream drops
  const deadline = Date.now() + ANALYZE_DEADLINE_MS;
  let attempts = 0;
  let timedOut = false;
  while (attempts < ANALYZE_ATTEMPTS) {
    attempts += 1;
    if (attempts > 1) {
      report({
        type: "ANALYSIS_STATUS",
        message:
          "The connection to your backend dropped; asking it to analyse the recording again " +
          `(attempt ${attempts} of ${ANALYZE_ATTEMPTS})…`,
      });
    }
    const call = await startCall(base, plan.apiName, data);
    if (call.error) {
      if (call.network && attempts > 1) {
        if (!(await pauseBeforeRetry(attempts, deadline))) break;
        continue;
      }
      fail(call.error);
      return;
    }
    const got = await readEventStream(base, plan.apiName, call.eventId, deadline);
    if (got.outcome === "result") {
      if (attempts > 1 && captureGone(got.verdict)) {
        fail(GONE_TEXT);
        return;
      }
      report({ type: "ANALYSIS_RESULT", result: got.verdict, capture, note });
      return;
    }
    if (got.outcome === "error") {
      // on a re-POST, an error is almost always the recording gone (the backend
      // keeps each one for about two minutes after analysing it, or it restarted)
      fail(attempts > 1 ? `${GONE_TEXT} (${got.message})` : got.message);
      return;
    }
    if (got.outcome === "deadline") {
      timedOut = true;
      break;
    }
    // "drop" or "idle": the stream died without an answer
    if (!(await pauseBeforeRetry(attempts, deadline))) break;
  }
  if (timedOut || Date.now() >= deadline) {
    fail(
      `Your backend gave no result within ${Math.round(ANALYZE_DEADLINE_MS / 60000)} minutes, so the ` +
        "extension stopped waiting. It may still be busy: see the terminal running `trubeat serve`."
    );
    return;
  }
  fail(
    `The connection to your backend dropped while analysing (${attempts} attempts). ` +
      "Is `trubeat serve` still running?"
  );
}

/** Wait before the next attempt; false when the budget is spent. */
async function pauseBeforeRetry(attempts, deadline) {
  if (attempts >= ANALYZE_ATTEMPTS || Date.now() + RETRY_DELAY_MS >= deadline) return false;
  await delay(RETRY_DELAY_MS);
  return true;
}

/** POST the analysis call; {eventId} or {error, network}. */
async function startCall(base, apiName, data) {
  let res;
  try {
    res = await fetch(base + "/gradio_api/call" + apiName, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data }),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (_) {
    return { error: unreachableText(base), network: true };
  }
  if (!res.ok) return { error: `The backend rejected the request (HTTP ${res.status}).` };
  let eventId = null;
  try {
    eventId = (await res.json()).event_id;
  } catch (_) {
    eventId = null;
  }
  if (!eventId) return { error: "The backend returned no event id." };
  return { eventId };
}

/**
 * One read of the SSE stream the backend opened for this analysis.
 *   { outcome: "result", verdict }  `event: complete` arrived;
 *   { outcome: "error", message }   `event: error` arrived (final, not retried);
 *   { outcome: "drop" }             the stream failed or ended without either;
 *   { outcome: "idle" }             no byte for SSE_IDLE_MS (heartbeats count);
 *   { outcome: "deadline" }         the overall deadline passed mid-stream.
 */
async function readEventStream(base, apiName, eventId, deadline) {
  const controller = new AbortController();
  let timer = null;
  let why = "";
  // The watchdog: re-armed by every chunk, it aborts the read after
  // SSE_IDLE_MS of silence, or when the overall deadline comes first.
  const arm = () => {
    clearTimeout(timer);
    const left = deadline - Date.now();
    const idle = left > SSE_IDLE_MS;
    timer = setTimeout(
      () => {
        why = idle ? "idle" : "deadline";
        controller.abort();
      },
      idle ? SSE_IDLE_MS : Math.max(0, left)
    );
  };
  const stopped = () => ({ outcome: why || "drop" });
  arm();
  try {
    let res;
    try {
      res = await fetch(base + "/gradio_api/call" + apiName + "/" + eventId, {
        cache: "no-store",
        signal: controller.signal,
      });
    } catch (_) {
      return stopped();
    }
    if (!res.ok || !res.body || typeof res.body.getReader !== "function") return stopped();
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    const parser = sseParser();
    const settle = (outcome) => {
      try {
        const done = typeof reader.cancel === "function" ? reader.cancel() : null;
        if (done && typeof done.catch === "function") done.catch(() => {});
      } catch (_) {
        /* the connection is closing anyway */
      }
      return outcome;
    };
    for (;;) {
      let chunk;
      try {
        chunk = await reader.read();
      } catch (_) {
        return stopped();
      }
      let events;
      if (chunk.done) {
        events = parser.end();
      } else {
        arm(); // any bytes, heartbeats included, prove the backend is alive
        events = parser.push(decoder.decode(chunk.value, { stream: true }));
      }
      for (const ev of events) {
        const outcome = interpretEvent(ev);
        if (outcome) return settle(outcome);
      }
      if (chunk.done) return stopped();
    }
  } finally {
    clearTimeout(timer);
  }
}

const GONE_TEXT =
  "The connection to your backend dropped during the analysis, and when the extension asked again " +
  "the backend no longer had the recording (it deletes each recording about two minutes after " +
  "analysing it, and a restart clears them). Check the tab again.";

/** A re-run that found the uploaded file gone (deleted after its grace period). */
function captureGone(result) {
  return !!(result && result.layers && typeof result.layers.file === "string");
}

function joinNotes(a, b) {
  return [a, b].filter(Boolean).join(" ");
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function unreachableText(base) {
  return `The backend at ${base} was unreachable. Is \`trubeat serve\` running?`;
}
