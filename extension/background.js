/**
 * TrueBeat extension — service worker.
 *
 * Responsibilities (and nothing else):
 *   1. own the capture lifecycle: popup -> tabCapture stream id -> offscreen
 *      recorder -> uploaded server path -> analysis result. One check per
 *      tab, locked until its result or an error arrives;
 *   2. find what to record: the largest *playing* <video> in any frame of
 *      the tab, its box in the tab's viewport, and the viewport's size, so
 *      the offscreen document records at the tab's real size and the
 *      backend analyses only the video (1.3.0). The box is read again when
 *      the recording stops; if the video moved, the whole tab is analysed;
 *   3. relay the analysis to the offscreen document: both the upload POST
 *      and the long network waits live there, because a Blob cannot survive
 *      a chrome.runtime message (it serialises to {}) and a service worker
 *      is idle-killed after ~30 s, which is exactly what an in-flight
 *      analysis stream looks like while the model computes;
 *   4. inject the in-page overlay (shadow DOM panel) on click, under
 *      activeTab, and push state into it. No content script is declared.
 *
 * No detection logic lives here. The extension never computes a verdict:
 * it analyses the server-side copy and renders whatever the backend
 * answers. Thresholds are never recalibrated client-side.
 *
 * Privacy: the only network destination is the backend URL the user set in
 * Options. Captures are never persisted; the offscreen document uploads
 * them straight to the backend, which deletes its copy once analysed.
 */

const DEFAULTS = {
  backend: "http://127.0.0.1:7860",
  captureSeconds: 20,
};

const OFFSCREEN_PATH = "offscreen.html";
const OVERLAY_FILE = "overlay.js";
const PING_TIMEOUT_MS = 4000;

// The AI-video detector needs at least one 2 s segment; keep a margin
// (offscreen.js refuses to upload anything shorter).
const MIN_CAPTURE_SECONDS = 3;
// From reading the clip's position to the recorder's first frame takes
// about a second (stream id, offscreen document, getUserMedia).
const START_MARGIN_S = 1;
// A video box that moved or changed size by more than this share of its
// size while recording no longer marks what was recorded (docs/13 §3.4).
const REGION_MOVE_TOLERANCE = 0.1;
// A child frame is placed in the tab by matching its viewport to exactly
// one <iframe> box of the top page, within this many CSS pixels.
const FRAME_MATCH_PX = 2;
// A check still unfinished after this long is stale: the capture (<= 60 s),
// the upload and the offscreen document's 6-minute analysis deadline all
// fit inside it. A recording (<= 60 s) that has not ended after two minutes
// is stale sooner, since it blocks every other tab.
const CHECK_TIMEOUT_MS = 10 * 60 * 1000;
const CAPTURE_TIMEOUT_MS = 2 * 60 * 1000;
const CHECKS_KEY = "checks";

/**
 * The checks in flight, by tab id: {checkId, phase, startedAt}, phase one
 * of "starting" | "capturing" | "uploading" | "analysing". Mirrored into
 * chrome.storage.session (memory only, never written to disk) so the lock
 * survives the worker being idled out mid-analysis. Nothing about the
 * video itself is kept.
 */
const checks = new Map();
/** tabId -> the probe taken when the recording stopped (memory only). */
const endProbes = new Map();
const session = chrome.storage && chrome.storage.session;
let checksLoaded = null;
let checkSeq = 0;

// ---------------------------------------------------------------- helpers

function normaliseBackend(url) {
  let u = (url || "").trim();
  if (!u) return DEFAULTS.backend;
  if (!/^https?:\/\//i.test(u)) u = "http://" + u;
  return u.replace(/\/+$/, "");
}

async function getSettings() {
  const stored = await chrome.storage.local.get(DEFAULTS);
  return {
    backend: normaliseBackend(stored.backend),
    captureSeconds: Math.min(
      60,
      Math.max(12, Number(stored.captureSeconds) || DEFAULTS.captureSeconds)
    ),
  };
}

function errText(err) {
  return String((err && err.message) || err || "unknown error");
}

// ------------------------------------------------------------ check state

function checksReady() {
  if (!checksLoaded) {
    checksLoaded = (async () => {
      if (!session) return;
      try {
        const got = await session.get(CHECKS_KEY);
        const saved = (got && got[CHECKS_KEY]) || {};
        for (const [tab, st] of Object.entries(saved)) {
          if (!checks.has(Number(tab))) checks.set(Number(tab), st);
        }
      } catch (_) {
        /* memory only */
      }
    })();
  }
  return checksLoaded;
}

async function saveChecks() {
  if (!session) return;
  try {
    await session.set({ [CHECKS_KEY]: Object.fromEntries(checks) });
  } catch (_) {
    /* memory only */
  }
}

function newCheckId() {
  checkSeq += 1;
  return Date.now().toString(36) + "-" + checkSeq.toString(36);
}

/** A message from a check this tab has since replaced. */
function isStale(msg) {
  const st = msg && checks.get(msg.tabId);
  return !!(st && msg.checkId != null && st.checkId !== msg.checkId);
}

async function setPhase(msg, phase) {
  const st = checks.get(msg.tabId);
  if (!st || (msg.checkId != null && st.checkId !== msg.checkId) || st.phase === phase) return;
  st.phase = phase;
  await saveChecks();
}

/** Release the tab's lock (only the given check's, when one is named). */
async function endCheck(tabId, checkId = null) {
  endProbes.delete(tabId);
  const st = checks.get(tabId);
  if (!st || (checkId != null && st.checkId !== checkId)) return;
  checks.delete(tabId);
  await saveChecks();
}

function isRecording(st) {
  return !!st && (st.phase === "starting" || st.phase === "capturing");
}

/** Forget stale checks (synchronously; the caller saves). True if any went. */
function dropTimedOut() {
  let changed = false;
  for (const [tab, st] of checks) {
    const limit = isRecording(st) ? CAPTURE_TIMEOUT_MS : CHECK_TIMEOUT_MS;
    if (!(Date.now() - Number(st && st.startedAt) < limit)) {
      checks.delete(tab);
      endProbes.delete(tab);
      changed = true;
    }
  }
  return changed;
}

// -------------------------------------------------------------- messaging

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    await checksReady();
    switch (msg && msg.type) {
      case "CHECK_TAB":
        sendResponse(await checkTab(msg.tabId));
        return;
      case "GET_STATUS":
        sendResponse(await status());
        return;
      case "PING_BACKEND":
        sendResponse(await pingBackend(msg.backend));
        return;
      case "PROGRESS":
        if (!isStale(msg)) {
          await setPhase(msg, "capturing");
          await pushOverlay(msg.tabId, { state: "capturing", ...(msg.payload || {}) });
        }
        break;
      case "CAPTURE_STOPPED":
        if (!isStale(msg)) {
          await setPhase(msg, "uploading");
          // Read the video's box again now, while the page still shows what
          // was recorded; the upload may take a while.
          endProbes.set(msg.tabId, probeTab(msg.tabId));
          await pushOverlay(msg.tabId, {
            state: "analyzing",
            message: "Uploading the recording to your backend…",
          });
        }
        break;
      case "CAPTURE_UPLOADED":
        if (!isStale(msg)) await handleCaptureUploaded(msg);
        break;
      case "ANALYSIS_STATUS":
        if (!isStale(msg)) {
          await pushOverlay(msg.tabId, {
            state: "analyzing",
            message: String(msg.message || "Analysing on your backend…"),
          });
        }
        break;
      case "ANALYSIS_RESULT":
        if (!isStale(msg)) {
          await endCheck(msg.tabId, msg.checkId);
          if (msg.result && typeof msg.result === "object") {
            await pushOverlay(msg.tabId, {
              state: "result",
              result: msg.result,
              capture: msg.capture || null,
              note: msg.note || "",
            });
          } else {
            await pushOverlay(msg.tabId, {
              state: "error",
              title: "Backend error",
              message: "The backend stream returned an unreadable result.",
            });
          }
        }
        break;
      case "CAPTURE_ERROR":
        if (!isStale(msg)) {
          await endCheck(msg.tabId, msg.checkId);
          await pushOverlay(msg.tabId, {
            state: "error",
            title: msg.title || "Capture failed",
            message: msg.message || "Unknown capture error",
          });
        }
        break;
      default:
        sendResponse({ ok: false, error: "unknown message" });
        return;
    }
    sendResponse({ ok: true });
  })();
  return true; // keep the channel open for the async response
});

// A closed tab's check is over: its result has nowhere to go.
if (chrome.tabs && chrome.tabs.onRemoved) {
  chrome.tabs.onRemoved.addListener((tabId) => {
    (async () => {
      await checksReady();
      await endCheck(tabId);
    })();
  });
}

async function status() {
  const s = await getSettings();
  const ping = await pingBackend(s.backend);
  return { backend: s.backend, captureSeconds: s.captureSeconds, ...ping };
}

/** Ping the backend's API-info endpoint (the popup dot, Options' test). */
async function pingBackend(raw) {
  const base = normaliseBackend(raw);
  try {
    const res = await fetch(base + "/gradio_api/info", {
      cache: "no-store",
      signal: AbortSignal.timeout(PING_TIMEOUT_MS),
    });
    if (!res.ok) return { reachable: false, error: `HTTP ${res.status}` };
    const info = await res.json();
    const names = Object.keys((info && info.named_endpoints) || {});
    return {
      reachable: true,
      hasAnalyze: names.includes("/analyze_video") || names.includes("/analyze_video_region"),
      hasRegion: names.includes("/analyze_video_region"),
      endpoints: names,
    };
  } catch (err) {
    return { reachable: false, error: errText(err) };
  }
}

// ---------------------------------------------------- what to record, how long

/**
 * Runs *in each frame of the page* (chrome.scripting, allFrames): every
 * <video> (open shadow roots included) with its box in this frame's
 * viewport (CSS px) and its playback state, the frame's viewport and
 * device pixel ratio, and, in the top frame, the content box of each
 * <iframe> so a child frame's video can be placed in the tab. Videos get
 * an id kept in this extension's isolated world (the page's DOM is not
 * touched), so the same element can be found again when the capture ends.
 */
function probeFrame() {
  const ids = (globalThis.__trueBeatVideoIds = globalThis.__trueBeatVideoIds || new WeakMap());
  let next = globalThis.__trueBeatNextVideoId || 1;
  const found = [];
  const seen = new Set();
  const collect = (root, depth) => {
    for (const v of root.querySelectorAll("video")) {
      if (!seen.has(v)) {
        seen.add(v);
        found.push(v);
      }
    }
    if (depth >= 4) return;
    for (const el of root.querySelectorAll("*")) {
      if (el.shadowRoot) collect(el.shadowRoot, depth + 1);
    }
  };
  collect(document, 0);
  const videos = found.map((v) => {
    if (!ids.has(v)) ids.set(v, next++);
    const r = v.getBoundingClientRect();
    return {
      id: ids.get(v),
      x: r.left,
      y: r.top,
      width: r.width,
      height: r.height,
      paused: !!v.paused,
      ended: !!v.ended,
      readyState: Number(v.readyState) || 0,
      duration: Number(v.duration),
      currentTime: Number(v.currentTime) || 0,
      loop: !!v.loop,
    };
  });
  globalThis.__trueBeatNextVideoId = next;
  const isTop = window.top === window;
  const iframes = [];
  if (isTop) {
    for (const el of document.querySelectorAll("iframe")) {
      const r = el.getBoundingClientRect();
      const cs = getComputedStyle(el);
      const pl = parseFloat(cs.paddingLeft) || 0;
      const pt = parseFloat(cs.paddingTop) || 0;
      const pr = parseFloat(cs.paddingRight) || 0;
      const pb = parseFloat(cs.paddingBottom) || 0;
      iframes.push({
        // the content box: where the child document's viewport is drawn
        x: r.left + el.clientLeft + pl,
        y: r.top + el.clientTop + pt,
        width: el.clientWidth - pl - pr,
        height: el.clientHeight - pt - pb,
        // a CSS transform scales the box on screen but not the child's layout
        scaled: Math.abs(r.width - el.offsetWidth) > 1 || Math.abs(r.height - el.offsetHeight) > 1,
      });
    }
  }
  const vv = window.visualViewport;
  return {
    isTop,
    parentIsTop: !isTop && window.parent === window.top,
    innerWidth: window.innerWidth,
    innerHeight: window.innerHeight,
    devicePixelRatio: window.devicePixelRatio || 1,
    zoomed: !!(vv && Math.abs((Number(vv.scale) || 1) - 1) > 0.01),
    videos,
    iframes,
  };
}

function rectOf(v) {
  const r = {
    x: Number(v && v.x),
    y: Number(v && v.y),
    width: Number(v && v.width),
    height: Number(v && v.height),
  };
  return [r.x, r.y, r.width, r.height].every(Number.isFinite) ? r : null;
}

function intersect(a, b) {
  if (!a || !b) return null;
  const x0 = Math.max(a.x, b.x);
  const y0 = Math.max(a.y, b.y);
  const x1 = Math.min(a.x + a.width, b.x + b.width);
  const y1 = Math.min(a.y + a.height, b.y + b.height);
  return x1 > x0 && y1 > y0 ? { x: x0, y: y0, width: x1 - x0, height: y1 - y0 } : null;
}

function viewportOf(result) {
  return {
    x: 0,
    y: 0,
    width: Number(result.innerWidth) || 0,
    height: Number(result.innerHeight) || 0,
  };
}

function framesOf(frames) {
  return (Array.isArray(frames) ? frames : []).filter(
    (f) => f && f.result && typeof f.result === "object"
  );
}

function topOf(list) {
  return list.find((f) => f.frameId === 0) || list.find((f) => f.result.isTop) || null;
}

function viewOf(top) {
  if (!top) return null;
  const r = top.result;
  const innerWidth = Number(r.innerWidth);
  const innerHeight = Number(r.innerHeight);
  if (!(innerWidth > 0 && innerHeight > 0)) return null;
  return {
    innerWidth,
    innerHeight,
    devicePixelRatio: Number(r.devicePixelRatio) > 0 ? Number(r.devicePixelRatio) : 1,
    zoomed: !!r.zoomed,
  };
}

/**
 * Where a video is in the tab's viewport (CSS px): {known, rect}. known is
 * false for a video in a frame that cannot be placed (nested deeper than
 * one level, or no single <iframe> box matches the frame's viewport within
 * 2 px); rect is its visible part, or null when it is off screen.
 */
function locate(f, v, top) {
  const own = intersect(rectOf(v), viewportOf(f.result));
  if (!own) return { known: true, rect: null };
  if (top && f === top) return { known: true, rect: own };
  if (!top || !f.result.parentIsTop) return { known: false, rect: null };
  const iw = Number(f.result.innerWidth);
  const ih = Number(f.result.innerHeight);
  const boxes = (Array.isArray(top.result.iframes) ? top.result.iframes : []).filter(
    (b) =>
      b &&
      !b.scaled &&
      Math.abs(Number(b.width) - iw) <= FRAME_MATCH_PX &&
      Math.abs(Number(b.height) - ih) <= FRAME_MATCH_PX
  );
  if (boxes.length !== 1) return { known: false, rect: null };
  const b = boxes[0];
  const placed = { x: Number(b.x) + own.x, y: Number(b.y) + own.y, width: own.width, height: own.height };
  return { known: true, rect: intersect(placed, viewportOf(top.result)) };
}

function plainView(view) {
  return view
    ? { innerWidth: view.innerWidth, innerHeight: view.innerHeight, devicePixelRatio: view.devicePixelRatio }
    : null;
}

function num(v) {
  return typeof v === "number" ? v : NaN;
}

/**
 * Pick the video to check from every frame's probe: the largest one on
 * screen that is *playing* (!paused, readyState >= 2); only when none plays,
 * the largest other one (so the plan can say it is paused). Returns
 *   info:   its playback state, for planCaptureSeconds (null: no video);
 *   rect:   its visible box in the tab's viewport, CSS px (null: unknown);
 *   view:   the tab's viewport and device pixel ratio (null: unknown);
 *   target: {frameId, id}, to find the same element again later.
 */
function chooseVideo(frames) {
  const list = framesOf(frames);
  const top = topOf(list);
  const view = viewOf(top);
  let playing = null;
  let other = null;
  for (const f of list) {
    for (const v of Array.isArray(f.result.videos) ? f.result.videos : []) {
      if (!v) continue;
      const where = locate(f, v, top);
      const visible = where.known ? where.rect : intersect(rectOf(v), viewportOf(f.result));
      const area = visible ? visible.width * visible.height : 0;
      if (!(area > 0)) continue;
      const cand = { f, v, area, rect: where.known ? where.rect : null };
      if (!v.paused && Number(v.readyState) >= 2) {
        if (!playing || area > playing.area) playing = cand;
      } else if (!other || area > other.area) {
        other = cand;
      }
    }
  }
  const pick = playing || other;
  if (!pick) return { info: null, rect: null, view: plainView(view), target: null };
  const v = pick.v;
  return {
    info: {
      duration: num(v.duration),
      currentTime: num(v.currentTime),
      loop: !!v.loop,
      paused: !!v.paused,
      ended: !!v.ended,
      readyState: num(v.readyState),
    },
    rect: pick.rect && view && !view.zoomed ? pick.rect : null,
    view: plainView(view),
    target: { frameId: pick.f.frameId, id: v.id },
  };
}

/** The same video found again: {rect, view} now (rect null when gone). */
function locateVideo(frames, target) {
  const list = framesOf(frames);
  const top = topOf(list);
  const view = viewOf(top);
  let rect = null;
  if (target) {
    const f = list.find((x) => x.frameId === target.frameId);
    const v =
      f && (Array.isArray(f.result.videos) ? f.result.videos : []).find((x) => x && x.id === target.id);
    if (f && v) {
      const where = locate(f, v, top);
      rect = where.known ? where.rect : null;
    }
  }
  return { rect: view && !view.zoomed ? rect : null, view: plainView(view) };
}

/**
 * Did the recorded box stay put? False when either box is unknown, the
 * viewport changed, or the box moved or changed size by more than 10% of
 * its size.
 */
function regionStable(start, end) {
  const a = start && start.rect;
  const b = end && end.rect;
  const va = start && start.view;
  const vb = end && end.view;
  if (!a || !b || !va || !vb) return false;
  if (
    Math.abs(va.innerWidth - vb.innerWidth) > 1 ||
    Math.abs(va.innerHeight - vb.innerHeight) > 1 ||
    Math.abs((va.devicePixelRatio || 1) - (vb.devicePixelRatio || 1)) > 0.01
  ) {
    return false;
  }
  const tx = REGION_MOVE_TOLERANCE * a.width;
  const ty = REGION_MOVE_TOLERANCE * a.height;
  return (
    Math.abs(b.x - a.x) <= tx &&
    Math.abs(b.y - a.y) <= ty &&
    Math.abs(b.width - a.width) <= tx &&
    Math.abs(b.height - a.height) <= ty
  );
}

/**
 * How many seconds to record: the configured length, cut to what is left
 * of the clip (a short AI clip would otherwise be recorded across its loop
 * seam, or frozen on its last frame), less a second of start-up. Returns
 * {seconds, note}, or {seconds: 0, note} when there is nothing to record.
 */
function planCaptureSeconds(configured, info) {
  const want = Math.max(
    MIN_CAPTURE_SECONDS,
    Math.round(Number(configured) || DEFAULTS.captureSeconds)
  );
  if (info && info.ended) {
    return { seconds: 0, note: "The video has ended. Rewind it, press play and check again." };
  }
  if (info && info.paused) {
    return { seconds: 0, note: "The video is paused. Press play, then check again." };
  }
  if (info && Number.isFinite(info.readyState) && info.readyState < 2) {
    return {
      seconds: 0,
      note: "The video is still loading. Wait until it plays, then check again.",
    };
  }
  if (!info || !Number.isFinite(info.duration) || info.duration <= 0) {
    return { seconds: want, note: "" }; // live stream or unknown: as configured
  }
  // Record only until the clip ends, looping or not: crossing a loop point
  // splices two moments of the clip together (a fake "cut").
  const left = info.duration - (Number(info.currentTime) || 0);
  if (left < MIN_CAPTURE_SECONDS) {
    return {
      seconds: 0,
      note: info.loop
        ? "The clip is about to loop. Check again right after it restarts."
        : "The video is almost over. Rewind it, press play and check again.",
    };
  }
  // The recorder starts about a second after this reading.
  const usable = left - START_MARGIN_S;
  if (usable < want) {
    const secs = Math.max(MIN_CAPTURE_SECONDS, Math.floor(usable));
    return { seconds: secs, note: `Short clip: recording ${secs} s (what is left of it).` };
  }
  return { seconds: want, note: "" };
}

// Exposed for the automated tests (tests/js/extension_test.mjs).
globalThis.__trueBeatPlanCapture = planCaptureSeconds;
globalThis.__trueBeatProbeFrame = probeFrame;
globalThis.__trueBeatChooseVideo = chooseVideo;
globalThis.__trueBeatLocateVideo = locateVideo;
globalThis.__trueBeatRegionStable = regionStable;

/**
 * Probe every frame the extension may script (Chrome skips frames it may
 * not read); the top frame alone if that is refused; [] when the page
 * forbids scripting (then the whole tab is recorded as configured).
 */
async function probeTab(tabId) {
  for (const allFrames of [true, false]) {
    try {
      const res = await chrome.scripting.executeScript({
        target: { tabId, allFrames },
        func: probeFrame,
      });
      return Array.isArray(res) ? res : [];
    } catch (_) {
      /* try the top frame alone, then give up */
    }
  }
  return [];
}

// --------------------------------------------------------------- capture

async function checkTab(tabId) {
  if (tabId == null) return { ok: false, error: "There is no tab to check." };
  // Check and take the lock before the first await, so two quick clicks
  // cannot both start a capture.
  const expired = dropTimedOut();
  if (checks.has(tabId)) {
    if (expired) await saveChecks();
    return { ok: false, error: "A check is already running in this tab. Wait for its result." };
  }
  for (const st of checks.values()) {
    if (isRecording(st)) {
      if (expired) await saveChecks();
      return {
        ok: false,
        error: "Another tab is being recorded right now. Wait for that recording to finish.",
      };
    }
  }
  const checkId = newCheckId();
  checks.set(tabId, { checkId, phase: "starting", startedAt: Date.now() });
  await saveChecks();
  const settings = await getSettings();

  const stop = async (title, message) => {
    await endCheck(tabId, checkId);
    await pushOverlay(tabId, { state: "error", title, message });
    return { ok: false, error: message };
  };

  // No content script is declared (1.3.0): the panel is injected now, on
  // the click (activeTab), before anything is pushed into it. A page that
  // cannot show it (chrome://, the Web Store) is not recorded at all: its
  // result would have nowhere to go.
  if (!(await injectOverlay(tabId))) {
    await endCheck(tabId, checkId);
    return {
      ok: false,
      error: "Chrome does not let extensions into this page, so TrueBeat cannot show a result here.",
    };
  }

  const found = chooseVideo(await probeTab(tabId));
  const plan = planCaptureSeconds(settings.captureSeconds, found.info);
  if (plan.seconds === 0) return stop("Check not started", plan.note);

  await pushOverlay(tabId, {
    state: "capturing",
    elapsed: 0,
    total: plan.seconds,
    message: plan.note || "Starting capture…",
  });

  let streamId;
  try {
    streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tabId });
  } catch (err) {
    return stop(
      "Can't capture this tab",
      "The browser refused tab capture (" + errText(err) + "). Reload the page and try again."
    );
  }

  let answer;
  try {
    await ensureOffscreen();
    answer = await chrome.runtime.sendMessage({
      type: "OFFSCREEN_START",
      streamId,
      tabId,
      checkId,
      seconds: plan.seconds,
      backend: settings.backend,
      note: plan.note,
      view: found.view,
      rect: found.rect,
      target: found.target,
    });
  } catch (err) {
    return stop("Can't capture this tab", "Could not start the recorder: " + errText(err));
  }
  if (answer && answer.ok === false) {
    return stop("Can't capture this tab", answer.error || "The recorder could not start.");
  }
  await setPhase({ tabId, checkId }, "capturing");
  return { ok: true };
}

async function ensureOffscreen() {
  const existing = await chrome.offscreen.hasDocument?.();
  if (existing) return;
  try {
    await chrome.offscreen.createDocument({
      url: OFFSCREEN_PATH,
      reasons: ["USER_MEDIA"],
      justification:
        "Record the current tab's video so it can be checked for AI generation and face-swaps.",
    });
  } catch (err) {
    // "Only a single offscreen document may be created" — already there.
    if (!String(err).includes("single offscreen document")) throw err;
  }
}

/**
 * The offscreen document uploaded the recording itself (a Blob cannot
 * cross a runtime message, so the message carries the server-side path,
 * never the clip). Validate that path, decide whether the video's box still
 * holds, then hand the analysis back to the offscreen document. The tab
 * stays locked until the result or an error arrives.
 */
async function handleCaptureUploaded(msg) {
  const tabId = msg.tabId;
  if (typeof msg.serverPath !== "string" || !msg.serverPath) {
    await endCheck(tabId, msg.checkId);
    await pushOverlay(tabId, {
      state: "error",
      title: "Backend error",
      message: "The backend accepted the recording but returned no usable path.",
    });
    return;
  }
  await setPhase(msg, "analysing");
  const settings = await getSettings();

  let region = msg.region && typeof msg.region === "object" ? msg.region : null;
  let note = "";
  if (!region && msg.rect) {
    note = "The video's position could not be mapped onto the recording, so the whole tab is analysed.";
  }
  if (region) {
    const pending = endProbes.get(tabId);
    endProbes.delete(tabId);
    const frames = pending ? await pending : await probeTab(tabId);
    if (!frames.length) {
      region = null;
      note = "The video's position could not be checked again after the recording, so the whole tab is analysed.";
    } else if (!regionStable({ rect: msg.rect, view: msg.view }, locateVideo(frames, msg.target))) {
      region = null;
      note = "The video moved, changed size or went away during the recording, so the whole tab is analysed.";
    }
  }

  await pushOverlay(tabId, {
    state: "analyzing",
    message: "Analysing on your backend…",
    note,
  });

  // The analysis wait must not live in the service worker: a stream that
  // sits idle while the model computes looks like idleness to the ~30 s
  // worker idle leash. Hand the server-side copy to the offscreen document,
  // which owns the connection; the verdict comes back as ANALYSIS_RESULT.
  try {
    await ensureOffscreen();
    await chrome.runtime.sendMessage({
      type: "ANALYZE",
      tabId,
      checkId: msg.checkId ?? null,
      serverPath: msg.serverPath,
      backend: settings.backend,
      region,
      capture: msg.capture || null,
      note,
    });
  } catch (err) {
    await endCheck(tabId, msg.checkId);
    await pushOverlay(tabId, {
      state: "error",
      title: "Backend error",
      message: "Could not hand the clip over for analysis: " + errText(err),
    });
  }
}

// ---------------------------------------------------------------- overlay

/** Inject the panel's script into the tab's top frame (idempotent). */
async function injectOverlay(tabId) {
  try {
    await chrome.scripting.executeScript({ target: { tabId }, files: [OVERLAY_FILE] });
    return true;
  } catch (_) {
    return false; // chrome://, the Web Store, the PDF viewer… cannot be scripted
  }
}

/** Post a state update to the panel, injecting it first if it is missing. */
async function pushOverlay(tabId, payload) {
  if (tabId == null) return;
  const message = { type: "TRUEBEAT_STATE", payload };
  try {
    await chrome.tabs.sendMessage(tabId, message);
    return;
  } catch (_) {
    // No panel in that tab (it navigated, or was opened before the click).
  }
  if (!(await injectOverlay(tabId))) return;
  try {
    await chrome.tabs.sendMessage(tabId, message);
  } catch (_) {
    // The tab cannot show the panel; the popup shows start errors.
  }
}
