/**
 * Automated tests for the TrueBeat browser extension.
 *
 * These load the *shipped* extension files (not copies) and exercise the
 * real code paths:
 *   1. offscreen.js — the backend client against a live `trubeat serve`
 *      (opt-in), plus the backend-unreachable path;
 *   2. overlay.js — verdict copy and colours, the result panel (the
 *      backend's own texts, pulse_measurable, warnings, escaping, the
 *      aria-live region) and fullscreen re-parenting;
 *   3. background.js — capture planning, the in-page probe and the choice
 *      of video, the region checks, the per-tab lock, overlay injection;
 *   4. offscreen.js — capture size and region mapping, the black-frame
 *      guard, capture failure modes, and the analysis wait on Gradio's real
 *      SSE shapes (event: complete / error / heartbeat), with its inactivity
 *      watchdog, its deadline and the re-POST after a dropped stream;
 *   5. end to end on a message bus that delivers like Chrome: click ->
 *      probe -> capture -> upload -> re-probe -> analysis -> result panel;
 *   6. a real HTTP backend: junk replies and a dropped connection;
 *   7. a static audit: no remote code, no telemetry, no requests to
 *      anything but the configured backend, the manifest's permissions.
 *
 * Timers inside the extension run on a virtual clock where it matters, so
 * a 90 s watchdog or a 6-minute deadline is tested in milliseconds.
 *
 * Run:  node tests/js/extension_test.mjs [--backend http://127.0.0.1:7860]
 */

import { existsSync, readFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createServer } from "node:http";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const EXT = join(ROOT, "extension");
// The JSON keys the backend sends, exported by tests/test_app.py from
// app.API_PAYLOAD_KEYS (docs/13 D-08): the contract the overlay reads.
const API_KEYS = JSON.parse(readFileSync(join(HERE, "api_keys.json"), "utf-8"));

let passed = 0;
let failed = 0;
const failures = [];

function check(name, condition, detail = "") {
  if (condition) {
    passed += 1;
    console.log(`  ok   ${name}`);
  } else {
    failed += 1;
    failures.push(`${name}${detail ? " — " + detail : ""}`);
    console.log(`  FAIL ${name}${detail ? " — " + detail : ""}`);
  }
}

let currentSection = "";

function section(title) {
  currentSection = title;
  console.log(`\n${title}`);
}

/** Run one section; an exception is a reported failure, not the end of the run. */
async function guarded(body) {
  try {
    await body();
  } catch (err) {
    const where = String((err && err.stack) || err).split("\n").slice(0, 3).join(" | ");
    check(`${currentSection.split(" — ")[0]} ran to the end without throwing`, false, where);
  }
}

// --------------------------------------------------------------- fixtures

const backendArgIndex = process.argv.indexOf("--backend");
const videoArgIndex = process.argv.indexOf("--video");
// Live backend checks are opt-in (--backend or TRUBEAT_BACKEND). They used
// to run whenever *any* server answered on :7860, which turned a plain
// `pytest` into a multi-GB upload and an OOM kill on an 8 GB laptop
// (docs/11 F12).
const BACKEND =
  backendArgIndex > -1
    ? process.argv[backendArgIndex + 1]
    : process.env.TRUBEAT_BACKEND || null;
// A small default clip (FF++ c23, ~1 MB). The harness reads the whole file
// into memory to upload it, so anything huge is refused below.
const MAX_LIVE_VIDEO_BYTES = 200 * 1024 * 1024;
const VIDEO =
  videoArgIndex > -1
    ? process.argv[videoArgIndex + 1]
    : join(ROOT, "data", "raw", "ffpp", "original_sequences", "youtube", "c23", "videos", "000.mp4");

/** Model the browser's message fidelity: chrome.runtime messages are
 * JSON-serialised, so anything that cannot be JSON (a Blob) arrives as {}. */
function rtMessage(m) {
  return m === undefined ? undefined : JSON.parse(JSON.stringify(m));
}

/** A chrome API stub good enough for the scripts to load. Every outbound
 * message goes through the same serialisation the real browser applies. */
function chromeStub(overrides = {}) {
  const store = { backend: BACKEND, captureSeconds: 20 };
  const sessionStore = {};
  return {
    storage: {
      local: {
        get: async (keys) => {
          const out = {};
          const list = Array.isArray(keys) ? keys : Object.keys(keys);
          for (const k of list) out[k] = store[k] ?? keys[k];
          return out;
        },
        set: async (obj) => Object.assign(store, obj),
      },
      session: sessionArea(sessionStore),
    },
    runtime: {
      onMessage: { addListener: () => {} },
      sendMessage: async (m) => rtMessage(m),
      id: "test",
    },
    offscreen: {
      hasDocument: async () => true,
      createDocument: async () => {},
    },
    tabCapture: { getMediaStreamId: async () => "stream-id" },
    tabs: {
      sendMessage: async (_tab, m) => rtMessage(m),
      query: async () => [],
      onRemoved: { addListener: () => {} },
    },
    scripting: { executeScript: async () => [], insertCSS: async () => {} },
    ...overrides,
  };
}

/** chrome.storage.session over a plain object (memory, like the real one). */
function sessionArea(backing) {
  return {
    get: async (key) => {
      const keys = typeof key === "string" ? [key] : Array.isArray(key) ? key : Object.keys(key || {});
      const out = {};
      for (const k of keys) if (k in backing) out[k] = rtMessage(backing[k]);
      return out;
    },
    set: async (obj) => {
      Object.assign(backing, rtMessage(obj));
    },
  };
}

/** Load a classic extension script into a fresh context with our stubs. */
function loadScript(file, globals) {
  const src = readFileSync(join(EXT, file), "utf-8");
  const fn = new Function(...Object.keys(globals), `${src}\n;`);
  return fn(...Object.values(globals));
}

/** Let every pending promise chain settle (two real macrotask turns). */
const turn = async (n = 2) => {
  for (let i = 0; i < n; i++) await new Promise((r) => setImmediate(r));
};

/**
 * A virtual clock for the extension's timers and Date.now(): advance(ms)
 * fires due timers in order, letting promise chains settle in between.
 */
function makeClock(start = Date.UTC(2026, 10, 7, 12)) {
  let now = start;
  let seq = 0;
  const timers = new Map();
  const add = (fn, ms, every) => {
    const id = ++seq;
    timers.set(id, { fn, at: now + Math.max(0, Number(ms) || 0), every });
    return id;
  };
  return {
    Date: { now: () => now },
    setTimeout: (fn, ms) => add(fn, ms, 0),
    clearTimeout: (id) => void timers.delete(id),
    setInterval: (fn, ms) => {
      const every = Math.max(1, Number(ms) || 0);
      return add(fn, every, every);
    },
    clearInterval: (id) => void timers.delete(id),
    now: () => now,
    async advance(ms) {
      const end = now + ms;
      for (;;) {
        await turn();
        let id = null;
        let t = null;
        for (const [k, v] of timers) if (v.at <= end && (!t || v.at < t.at)) [id, t] = [k, v];
        if (!t) break;
        now = t.at;
        if (t.every) t.at += t.every;
        else timers.delete(id);
        try {
          t.fn();
        } catch (e) {
          console.log("  (a timer threw)", e);
        }
      }
      now = end;
      await turn();
    },
  };
}

/** Advance virtual time until cond() holds (or the budget runs out). */
async function runUntil(clock, cond, { step = 500, maxMs = 15 * 60 * 1000 } = {}) {
  for (let t = 0; t <= maxMs; t += step) {
    await turn();
    if (cond()) return true;
    await clock.advance(step);
  }
  return !!cond();
}

const jsonRes = (value, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => value,
});

// Gradio 6.30's GET /gradio_api/call/<api>/<event_id> stream, verbatim shapes.
const HEARTBEAT = "event: heartbeat\ndata: null\n\n";
const completeEvent = (obj) => `event: complete\ndata: ${JSON.stringify([JSON.stringify(obj)])}\n\n`;
const errorEvent = (data) => `event: error\ndata: ${JSON.stringify(data)}\n\n`;

/**
 * A fetch body whose reader plays `steps` ({text, delay}, {hang}, {drop},
 * {end}; or a function of the read index). Reads honour the request's
 * AbortSignal the way fetch does: an abort rejects the read in flight.
 */
function sseBody(steps, { clock, signal } = {}) {
  const encoder = new TextEncoder();
  let i = 0;
  let pending = null;
  let aborted = !!(signal && signal.aborted);
  const abortError = () => Object.assign(new Error("This operation was aborted"), { name: "AbortError" });
  if (signal) {
    signal.addEventListener("abort", () => {
      aborted = true;
      if (pending) {
        const reject = pending;
        pending = null;
        reject(abortError());
      }
    });
  }
  return {
    getReader: () => ({
      read: () =>
        new Promise((resolve, reject) => {
          if (aborted) {
            reject(abortError());
            return;
          }
          const step = typeof steps === "function" ? steps(i) : steps[i];
          i += 1;
          if (!step || step.end) {
            resolve({ done: true, value: undefined });
            return;
          }
          if (step.drop) {
            reject(new TypeError("network error"));
            return;
          }
          pending = reject;
          if (step.hang) return; // never answers: only an abort ends it
          const deliver = () => {
            if (pending !== reject) return;
            pending = null;
            resolve({ done: false, value: encoder.encode(step.text) });
          };
          if (step.delay && clock) clock.setTimeout(deliver, step.delay);
          else deliver();
        }),
      cancel: async () => {
        aborted = true;
      },
    }),
  };
}

/**
 * A fake TrueBeat backend for fetch: /info, /upload, POST /call/<api>
 * (event ids ev-1, ev-2, … one per POST) and the per-event SSE stream
 * produced by stream({attempt, eventId, signal, clock}).
 */
function fakeBackend({
  clock,
  endpoints = { "/analyze_video": {}, "/analyze_video_region": {} },
  uploadPath = "/tmp/gradio/abc123/truebeat-capture.webm",
  stream,
} = {}) {
  const log = [];
  let calls = 0;
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    const entry = { url: u, method: opts.method || "GET", body: opts.body };
    log.push(entry);
    if (u.endsWith("/gradio_api/info")) return jsonRes({ named_endpoints: endpoints });
    if (u.endsWith("/gradio_api/upload")) return jsonRes([uploadPath]);
    if (/\/gradio_api\/call\/[a-z_]+$/.test(u) && entry.method === "POST") {
      calls += 1;
      entry.eventId = "ev-" + calls;
      return jsonRes({ event_id: entry.eventId });
    }
    const m = u.match(/\/gradio_api\/call\/[a-z_]+\/(ev-(\d+))$/);
    if (m) {
      return stream({ attempt: Number(m[2]), eventId: m[1], signal: opts.signal, clock });
    }
    return jsonRes({}, 404);
  };
  return {
    fetch: fetchImpl,
    log,
    posts: () => log.filter((e) => e.method === "POST" && /\/gradio_api\/call\//.test(e.url)),
    gets: () => log.filter((e) => e.method === "GET" && /\/gradio_api\/call\/[a-z_]+\/ev-/.test(e.url)),
    uploads: () => log.filter((e) => e.url.endsWith("/gradio_api/upload")),
  };
}

const streamOf = (steps) => ({ signal, clock }) => ({
  ok: true,
  status: 200,
  body: sseBody(steps, { clock, signal }),
});

const VERDICT = {
  verdict: "fake",
  headline: "Possible face-swap",
  detail: "The face-swap detector scores the face +1.40, at or above its threshold.",
  score: 0.91,
  decided_by: "face_swap",
  caveat: "Backend caveat: 'No sign of AI generation' is not proof a video is real.",
  warnings: [],
};

/** Frame pixels (RGBA) for the sampler's canvas: black, or a varied picture. */
function pixels(w, h, kind) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const v = kind === "black" ? 2 : ((i * 37) % 200) + 20;
    data[i * 4] = v;
    data[i * 4 + 1] = v;
    data[i * 4 + 2] = v;
    data[i * 4 + 3] = 255;
  }
  return data;
}

function makeTrack(settings) {
  const handlers = {};
  return {
    kind: "video",
    handlers,
    stopped: false,
    addEventListener: (type, fn) => {
      handlers[type] = fn;
    },
    getSettings: () => ({ ...settings }),
    stop() {
      this.stopped = true;
    },
  };
}

/**
 * The globals the offscreen document sees, with fakes for the media stack:
 * getUserMedia (records the constraints), a MediaRecorder that behaves like
 * Chrome's (stop -> dataavailable -> stop event, asynchronously), and a
 * <video>/<canvas> pair whose pixels the test chooses ("normal"/"black").
 */
function offscreenGlobals({ chrome, clock, fetchImpl, settings = { width: 1920, height: 1080 }, delivered = settings, gum } = {}) {
  const env = {
    recorders: [],
    drawn: [],
    drawArgs: [],
    videos: [],
    gumCalls: [],
    frame: "normal",
    track: makeTrack(settings),
  };
  env.stream = { getVideoTracks: () => [env.track], getTracks: () => [env.track] };
  const getUserMedia =
    gum ||
    (async (constraints) => {
      env.gumCalls.push(rtMessage(constraints));
      return env.stream;
    });
  function FakeRecorder(stream, options) {
    this.stream = stream;
    this.options = options;
    this.state = "inactive";
    env.recorders.push(this);
  }
  FakeRecorder.isTypeSupported = (t) => t.startsWith("video/webm");
  FakeRecorder.prototype.start = function () {
    this.state = "recording";
  };
  FakeRecorder.prototype.stop = function () {
    if (this.state === "inactive") return;
    this.state = "inactive";
    Promise.resolve().then(() => {
      if (this.ondataavailable) {
        this.ondataavailable({ data: new Blob([new Uint8Array(96).fill(7)], { type: "video/webm" }) });
      }
      if (this.onstop) this.onstop();
    });
  };
  const document = {
    body: {
      appendChild: (el) => {
        el.attached = true;
      },
    },
    createElement: (tag) => {
      if (tag === "video") {
        const v = {
          tagName: "VIDEO",
          readyState: 4,
          videoWidth: delivered.width,
          videoHeight: delivered.height,
          play() {
            this.playing = true;
            return Promise.resolve();
          },
          pause() {
            this.playing = false;
          },
          remove() {
            this.removed = true;
          },
        };
        env.videos.push(v);
        return v;
      }
      if (tag === "canvas") {
        return {
          width: 0,
          height: 0,
          getContext: () => ({
            // Like Chrome: a MediaStreamTrack is not a CanvasImageSource.
            drawImage: (src, ...args) => {
              if (!src || src.tagName !== "VIDEO") {
                throw new TypeError("Failed to execute 'drawImage': the provided value is not an image source.");
              }
              env.drawn.push(src);
              env.drawArgs.push(args);
            },
            getImageData: (_x, _y, w, h) => ({ data: pixels(w, h, env.frame) }),
          }),
        };
      }
      return {};
    },
  };
  env.globals = {
    chrome,
    navigator: { mediaDevices: { getUserMedia } },
    window: { MediaRecorder: FakeRecorder },
    MediaRecorder: FakeRecorder,
    document,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    setInterval: clock.setInterval,
    clearInterval: clock.clearInterval,
    Date: clock.Date,
    fetch: fetchImpl || (async () => {
      throw new TypeError("fetch failed");
    }),
    FormData,
    Blob,
    TextDecoder,
    TextEncoder,
    AbortController,
    AbortSignal,
    console,
  };
  return env;
}

/** offscreen.js alone: what it posts to the runtime, on a virtual clock. */
function offscreenEnv({ clock = makeClock(), fetchImpl, settings, delivered, gum } = {}) {
  const listeners = [];
  const sent = [];
  const chrome = chromeStub({
    runtime: {
      onMessage: { addListener: (fn) => listeners.push(fn) },
      sendMessage: async (m) => {
        sent.push(rtMessage(m));
        return {};
      },
      id: "test",
    },
  });
  const env = offscreenGlobals({ chrome, clock, fetchImpl, settings, delivered, gum });
  loadScript("offscreen.js", env.globals);
  return Object.assign(env, {
    clock,
    listeners,
    sent,
    /** Deliver a message; resolves with the listener's answer. */
    send: (msg) =>
      new Promise((resolve) => {
        const keep = listeners[0](rtMessage(msg), { id: "test" }, (r) => resolve(rtMessage(r)));
        if (keep !== true) resolve(undefined);
      }),
    of: (type) => sent.filter((m) => m.type === type),
  });
}

/** Run one ANALYZE in an offscreen env until it posts its outcome. */
async function analyzeOnce(env, extra = {}, { maxMs = 15 * 60 * 1000 } = {}) {
  const before = env.sent.length;
  const done = () =>
    env.sent.slice(before).find((m) => m.type === "ANALYSIS_RESULT" || m.type === "CAPTURE_ERROR");
  await env.send({
    type: "ANALYZE",
    tabId: 7,
    checkId: "c-1",
    serverPath: "/tmp/gradio/abc123/truebeat-capture.webm",
    backend: "http://127.0.0.1:9",
    ...extra,
  });
  await runUntil(env.clock, () => !!done(), { maxMs });
  return done() || null;
}

/**
 * background.js + offscreen.js on one message bus that delivers like Chrome:
 * a runtime message reaches every *other* extension context, and the first
 * sendResponse answers it. tabs.sendMessage only reaches a tab whose panel
 * was injected (no content script is declared since 1.3.0).
 */
function busEnv({
  clock = makeClock(),
  backend,
  frames = [],
  settings = { width: 1920, height: 1080 },
  captureSeconds = 20,
} = {}) {
  const listeners = [];
  const deliveries = [];
  const overlays = [];
  const tabCalls = [];
  const scriptingCalls = [];
  const injected = new Set();
  const settingsStore = { backend: "http://127.0.0.1:9", captureSeconds };
  const sessionStore = {};
  const probe = { frames, count: 0 };
  const runtimeFor = (owner) => ({
    id: "test",
    onMessage: { addListener: (fn) => listeners.push({ owner, fn }) },
    sendMessage: (m) =>
      new Promise((resolve) => {
        const d = rtMessage(m);
        deliveries.push({ from: owner, ...d });
        let answered = false;
        let waiting = 0;
        const respond = (r) => {
          if (answered) return;
          answered = true;
          resolve(rtMessage(r));
        };
        for (const l of [...listeners]) {
          if (l.owner === owner) continue;
          let keep;
          try {
            keep = l.fn(rtMessage(d), { id: "test" }, respond);
          } catch (_) {
            /* one broken handler never kills the bus */
          }
          if (keep === true) waiting += 1;
        }
        if (!waiting) respond(undefined);
      }),
  });
  const chromeFor = (owner) => ({
    storage: {
      local: {
        get: async (keys) => ({ ...keys, ...settingsStore }),
        set: async (o) => Object.assign(settingsStore, o),
      },
      session: sessionArea(sessionStore),
    },
    runtime: runtimeFor(owner),
    offscreen: { hasDocument: async () => true, createDocument: async () => {} },
    tabCapture: { getMediaStreamId: async () => "sid-123" },
    tabs: {
      sendMessage: async (tabId, m) => {
        tabCalls.push({ kind: "send", tabId, state: m.payload && m.payload.state });
        if (!injected.has(tabId)) {
          throw new Error("Could not establish connection. Receiving end does not exist.");
        }
        overlays.push(rtMessage(m.payload));
        return { ok: true };
      },
      query: async () => [],
      onRemoved: { addListener: () => {} },
    },
    scripting: {
      executeScript: async (details) => {
        scriptingCalls.push(details);
        if (details.files) {
          tabCalls.push({ kind: "inject", tabId: details.target.tabId, files: details.files });
          injected.add(details.target.tabId);
          return [{ frameId: 0, result: null }];
        }
        probe.count += 1;
        const f = typeof probe.frames === "function" ? probe.frames(probe.count) : probe.frames;
        return rtMessage(f || []);
      },
      insertCSS: async (d) => {
        scriptingCalls.push({ insertCSS: d });
        if (!d || !d.files || !d.files.length) throw new Error("No source code or file specified.");
      },
    },
  });
  const off = offscreenGlobals({ chrome: chromeFor("offscreen"), clock, fetchImpl: backend.fetch, settings });
  loadScript("offscreen.js", off.globals);
  loadScript("background.js", {
    chrome: chromeFor("background"),
    fetch: backend.fetch,
    Date: clock.Date,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    AbortSignal,
    console,
  });
  const popup = runtimeFor("popup");
  return {
    clock,
    off,
    backend,
    deliveries,
    overlays,
    tabCalls,
    scriptingCalls,
    probe,
    popup,
    settingsStore,
    check: (tabId = 7) => popup.sendMessage({ type: "CHECK_TAB", tabId }),
    last: () => overlays[overlays.length - 1] || null,
    finished: () => overlays.some((p) => p.state === "result" || p.state === "error"),
    sent: (type) => deliveries.filter((d) => d.type === type),
  };
}

/** One frame's probe result, as background.probeFrame returns it. */
function frameResult({
  frameId = 0,
  isTop = frameId === 0,
  parentIsTop = !isTop,
  innerWidth = 1280,
  innerHeight = 720,
  devicePixelRatio = 1.5,
  videos = [],
  iframes = [],
  zoomed = false,
} = {}) {
  return {
    frameId,
    result: { isTop, parentIsTop, innerWidth, innerHeight, devicePixelRatio, zoomed, videos, iframes },
  };
}

function videoInfo(over = {}) {
  return {
    id: 1,
    x: 100,
    y: 80,
    width: 640,
    height: 360,
    paused: false,
    ended: false,
    readyState: 4,
    duration: 120,
    currentTime: 10,
    loop: false,
    ...over,
  };
}

// -------------------------------------------- 1. the backend client (ANALYZE)

section("1. offscreen.js — the backend client (ANALYZE against a backend)");
await guarded(async () => {
  // The whole backend conversation lives in the offscreen document
  // (background.js only relays). These drive the shipped ANALYZE handler
  // with the real fetch and real timers, and read what it posts.
  const listeners = [];
  const sent = [];
  loadScript("offscreen.js", {
    chrome: chromeStub({
      runtime: {
        onMessage: { addListener: (fn) => listeners.push(fn) },
        sendMessage: async (m) => {
          sent.push(rtMessage(m));
          return {};
        },
        id: "test",
      },
    }),
    navigator: {},
    window: {},
    document: { createElement: () => ({}) },
    fetch,
    FormData,
    Blob,
    TextDecoder,
    console,
  });

  // Run one analysis and return the message it posted to the runtime.
  const analyse = async (serverPath, backendUrl, capMs = 540000) => {
    const before = sent.length;
    const startedAt = Date.now();
    listeners[0]({ type: "ANALYZE", tabId: 7, serverPath, backend: backendUrl }, null, () => {});
    for (;;) {
      const outcome = sent
        .slice(before)
        .find((m) => m.type === "ANALYSIS_RESULT" || m.type === "CAPTURE_ERROR");
      if (outcome) return outcome;
      if (Date.now() - startedAt > capMs) return null;
      await new Promise((r) => setTimeout(r, 100));
    }
  };

  // Backend unreachable -> an honest error, never a fabricated verdict.
  const posted = await analyse("/tmp/gradio/truebeat/dead/dead.webm", "http://127.0.0.1:9");
  check(
    "unreachable backend posts a readable backend error",
    posted &&
      posted.type === "CAPTURE_ERROR" &&
      /unreachable|HTTP|refused|fetch failed|ECONNREFUSED/i.test(posted.message || ""),
    JSON.stringify(posted)
  );

  if (!BACKEND) {
    console.log(
      "  skip live backend checks (opt-in: pass --backend URL or set TRUBEAT_BACKEND)"
    );
  } else if (!existsSync(VIDEO) || statSync(VIDEO).size > MAX_LIVE_VIDEO_BYTES) {
    check(
      "live test video is present and small enough to upload from memory",
      false,
      `${VIDEO} (limit ${MAX_LIVE_VIDEO_BYTES} bytes; pass --video <small clip>)`
    );
  } else {
    // Is a backend actually up? If not, the live checks are skipped (CI has
    // no backend) rather than failed — the offline checks still run.
    let up = false;
    try {
      const ping = await fetch(BACKEND + "/gradio_api/info", { signal: AbortSignal.timeout(4000) });
      up = ping.ok;
    } catch (_) {
      up = false;
    }
    if (!up) {
      console.log(
        `  skip live backend checks (no backend at ${BACKEND}) — ` +
          "start one with `trubeat serve` to exercise them"
      );
    } else {
      let apiResult = null;
      let liveError = null;
      try {
        // The harness uploads exactly like the extension's offscreen
        // document, then analyses the server-side copy through ANALYZE.
        const form = new FormData();
        form.append("files", new Blob([readFileSync(VIDEO)], { type: "video/mp4" }), "test.mp4");
        const upRes = await fetch(BACKEND + "/gradio_api/upload", { method: "POST", body: form });
        if (!upRes.ok) throw new Error("HTTP " + upRes.status + " while uploading");
        const path = (await upRes.json())[0];
        apiResult = await analyse(path, BACKEND);
      } catch (e) {
        liveError = e;
      }
      const result = apiResult && apiResult.type === "ANALYSIS_RESULT" ? apiResult.result : null;
      const postedError = apiResult && apiResult.type === "CAPTURE_ERROR" ? apiResult.message : null;
      if (liveError || postedError || !result) {
        check(
          "live backend returns a verdict",
          false,
          "backend error: " +
            String((liveError && liveError.message) || postedError || "no result").slice(0, 200)
        );
      } else {
        check("live backend returns a verdict", !!result, "no result");
        check(
          "verdict is one of the three states",
          ["real", "fake", "inconclusive"].includes(result.verdict),
          String(result.verdict)
        );
        check(
          "the live reply carries exactly the documented keys (tests/js/api_keys.json)",
          JSON.stringify(Object.keys(result)) === JSON.stringify(API_KEYS),
          JSON.stringify(Object.keys(result))
        );
        check("result carries the score", "score" in result);
        check("result carries hr_bpm", "hr_bpm" in result);
        check("result carries region_agreement_bpm", "region_agreement_bpm" in result);
        check("result carries snr", "snr" in result);
        check("result carries pulse_measurable", "pulse_measurable" in result);
        check("result carries the warnings list", Array.isArray(result.warnings));
        check("result carries qc", !!result.qc && typeof result.qc === "object");
        check("result carries the caveat", /face-swap/i.test(result.caveat || ""));
        check(
          "headline uses the app wording",
          /No sign of AI generation|Likely AI-generated|Possible face-swap|AI-generated: declared|Inconclusive/.test(
            result.headline || ""
          ),
          String(result.headline)
        );
      }
    }
  }
});

// ------------------------------------------------ 2. overlay verdict mapping

/** A small DOM for the overlay: elements that can be (re)parented. */
function overlayDom() {
  const docListeners = {};
  const made = [];
  const el = (tag) => {
    const node = {
      tagName: String(tag).toUpperCase(),
      style: {},
      children: [],
      parentNode: null,
      isConnected: false,
      className: "",
      id: "",
      textContent: "",
      _html: "",
      _subs: {},
      set innerHTML(v) {
        this._html = v;
      },
      get innerHTML() {
        return this._html;
      },
      appendChild(child) {
        if (child.parentNode) {
          child.parentNode.children = child.parentNode.children.filter((c) => c !== child);
        }
        child.parentNode = this;
        child.isConnected = true;
        this.children.push(child);
        return child;
      },
      remove() {
        if (this.parentNode) {
          this.parentNode.children = this.parentNode.children.filter((c) => c !== this);
        }
        this.parentNode = null;
        this.isConnected = false;
      },
      attachShadow() {
        return { append: () => {}, appendChild: () => {} };
      },
      addEventListener() {},
      querySelector(sel) {
        if (!this._subs[sel]) {
          this._subs[sel] = {
            selector: sel,
            style: {},
            textContent: "",
            innerHTML: "",
            addEventListener: () => {},
          };
        }
        return this._subs[sel];
      },
    };
    made.push(node);
    return node;
  };
  const body = el("body");
  body.isConnected = true;
  const document = {
    body,
    documentElement: body,
    fullscreenElement: null,
    createElement: el,
    addEventListener: (type, fn) => {
      (docListeners[type] = docListeners[type] || []).push(fn);
    },
  };
  return {
    document,
    el,
    body,
    fire: (type) => (docListeners[type] || []).forEach((fn) => fn()),
    panel: () => made.find((n) => n.className === "panel"),
    host: () => made.find((n) => n.id === "truebeat-overlay-host"),
  };
}

/** Load overlay.js into a fresh DOM; show(payload) returns the body HTML. */
function loadOverlay() {
  const dom = overlayDom();
  let listener = null;
  loadScript("overlay.js", {
    chrome: chromeStub({
      runtime: { onMessage: { addListener: (f) => (listener = f) }, id: "test" },
    }),
    window: {},
    document: dom.document,
    console,
  });
  const show = (payload) => {
    let answered = null;
    listener({ type: "TRUEBEAT_STATE", payload }, null, (r) => (answered = r));
    show.answered = answered;
    const panel = dom.panel();
    return panel ? panel.querySelector(".body").innerHTML : "";
  };
  return { dom, show, listener: () => listener };
}

section("2. overlay.js — verdict copy, the app's headlines and the caveat");
await guarded(async () => {
  const { listener } = loadOverlay();
  check("overlay listens for state updates", typeof listener() === "function");
  const styles = globalThis.__trueBeatVerdictStyles;
  const headline = globalThis.__trueBeatFallbackHeadline;
  const caveat = globalThis.__trueBeatStandingCaveat;
  check("overlay exposes the three verdict styles", !!styles);
  if (styles) {
    check("real -> green", styles.real.accent === "#22c55e", styles.real.accent);
    check("fake -> red", styles.fake.accent === "#ef4444", styles.fake.accent);
    check("inconclusive -> amber", styles.inconclusive.accent === "#f59e0b", styles.inconclusive.accent);
    check(
      "the three states are the app's wording",
      styles.real.label === "No sign of AI generation found" &&
        styles.fake.label === "Likely AI-generated" &&
        styles.inconclusive.label === "Inconclusive",
      JSON.stringify(styles)
    );
  }
  if (headline) {
    check(
      "a fake verdict without a headline is named by the layer that decided it",
      headline({ verdict: "fake", decided_by: "provenance" }) ===
        "AI-generated: declared by its Content Credentials" &&
        headline({ verdict: "fake", decided_by: "ai_video" }) === "Likely AI-generated" &&
        headline({ verdict: "fake", decided_by: "face_swap" }) === "Possible face-swap" &&
        headline({ verdict: "fake", decided_by: "pulse" }) === "Possible face-swap",
      [
        headline({ verdict: "fake", decided_by: "provenance" }),
        headline({ verdict: "fake", decided_by: "face_swap" }),
      ].join(" | ")
    );
    check(
      "real and inconclusive fall back to the app's headlines",
      headline({ verdict: "real" }) === "No sign of AI generation found" &&
        headline({ verdict: "inconclusive" }) === "Inconclusive" &&
        headline({}) === "Inconclusive"
    );
  }
  // The fallback labels must be words the backend itself uses (C-16): each
  // is a string literal in app.py (Inconclusive as the prefix of one).
  const appSrc = readFileSync(join(ROOT, "src", "trubeat", "app.py"), "utf-8");
  const literals = new Set([...appSrc.matchAll(/"([^"\n]+)"/g)].map((m) => m[1]));
  const labels = [
    ...Object.values(styles || {}).map((s) => s.label),
    ...Object.values(globalThis.__trueBeatFakeHeadlines || {}),
  ];
  const strangers = labels.filter(
    (l) => !literals.has(l) && !(l === "Inconclusive" && [...literals].some((s) => s.startsWith("Inconclusive")))
  );
  check(
    "every fallback headline is one the backend emits (app.verdict_for)",
    labels.length >= 6 && strangers.length === 0,
    strangers.join(" | ")
  );
  check(
    "no 1.2 label the app never emits ('Likely AI-generated or manipulated')",
    !readFileSync(join(EXT, "overlay.js"), "utf-8").includes("or manipulated")
  );
  check("overlay carries the standing caveat", /Face2Face/.test(caveat || ""));
  check("caveat never claims a pass proves authenticity", /not proof/.test(caveat || ""), caveat);
  check(
    "caveat names the out-of-scope families",
    /Face2Face/.test(caveat || "") && /NeuralTextures/.test(caveat || "")
  );
  check(
    "caveat says the face-swap detector is evidence only, with its measured false alarms",
    /evidence only/.test(caveat || "") && /16%/.test(caveat || ""), // v2's gate-4a rate (docs/13 §10)
    caveat
  );
});

section("2c. overlay.js — the result panel renders what the backend says");
await guarded(async () => {
  const { dom, show } = loadOverlay();
  const html = show({
    state: "result",
    result: {
      verdict: "fake",
      decided_by: "face_swap",
      fsd_mode: "decisive",
      fsd_score: 1.4,
      fsd_threshold: 0.5,
      fsd_n_faces: 15,
      fsd_n_sampled: 16,
      score: -0.2,
    },
  });
  check("the panel answers the message (no dangling sender)", show.answered && show.answered.ok === true);
  check(
    "the detector's score, threshold and faces are shown",
    /face-swap detector<\/td><td class="v">score 1\.40 \(flags at 0\.50\); face in 15 of 16 frames/.test(html),
    html
  );
  check("decided by reads 'face swap'", /decided by<\/td><td class="v">face swap/.test(html), html);
  check("the pulse score is labelled as the pulse check's", /pulse face-swap score/.test(html), html);
  check(
    "a reply without a headline gets the app's own ('Possible face-swap')",
    /Possible face-swap/.test(html),
    html
  );
  const ev = show({ state: "result", result: { verdict: "real", fsd_mode: "evidence", fsd_score: 1.4, fsd_threshold: 0.5 } });
  check("an evidence-only detector says so", /flags at 0\.50\), evidence only/.test(ev), ev);
  const ev2 = show({
    state: "result",
    result: { verdict: "real", fsd_mode: "evidence", fsd_score: 1.4, fsd_threshold: 0.5, fsd_outside_fpr: 0.383 },
  });
  check(
    "an evidence-only row quotes its measured false alarms",
    /evidence only \(flagged 38% of real faces/.test(ev2),
    ev2
  );
  const none = show({ state: "result", result: { verdict: "real", ai_score: -1.0, ai_threshold: 1.1 } });
  check("no detector row when it did not run", !/class="k">face-swap detector/.test(none), none);

  // C-09: an unqualified heart rate is never shown when the pulse is noise.
  const noise = show({ state: "result", result: { verdict: "real", hr_bpm: 71.2, pulse_measurable: false } });
  check(
    "a heart rate the backend calls unmeasurable is marked not reliable",
    /heart rate<\/td><td class="v">71\.2 bpm \(not reliable: pulse not measurable\)/.test(noise),
    noise
  );
  const okPulse = show({ state: "result", result: { verdict: "real", hr_bpm: 71.2, pulse_measurable: true } });
  check(
    "a measurable pulse is shown plain",
    /71\.2 bpm</.test(okPulse) && !/not reliable/.test(okPulse),
    okPulse
  );
  const warned = show({
    state: "result",
    result: {
      verdict: "inconclusive",
      headline: "Inconclusive: no check could reach a decision",
      warnings: [
        "W_TRUNCATED: the pulse layer analysed the first 120s",
        "W_PARTIAL_COVERAGE: scored on one 2 s segment",
        "W_GRAPHICS_SKIPPED: 1 segment looked like a title card",
        "W_FEW_FACES: a face was found in 2 of 16 sampled frames",
        "W_UNSTABLE_TRACK: face tracked in 41% of frames <script>alert(1)</script>",
      ],
    },
  });
  check(
    "the backend's warnings are listed under the table",
    /<ul class="warn"><li>W_TRUNCATED: the pulse layer analysed the first 120s<\/li>/.test(warned) &&
      /W_PARTIAL_COVERAGE/.test(warned) &&
      /W_GRAPHICS_SKIPPED/.test(warned) &&
      /W_FEW_FACES/.test(warned) &&
      /W_UNSTABLE_TRACK/.test(warned),
    warned
  );
  check(
    "warnings are escaped (no markup from the backend reaches the page)",
    !/<script>/.test(warned) && /&lt;script&gt;alert\(1\)&lt;\/script&gt;/.test(warned),
    warned
  );
  const oldBackend = show({ state: "result", result: { verdict: "real", qc: { warnings: ["W_FEW_FACES: none"] } } });
  check("an older backend's qc.warnings are shown too", /W_FEW_FACES: none/.test(oldBackend), oldBackend);
  const own = show({
    state: "result",
    result: {
      verdict: "real",
      headline: "No sign of AI generation found <b>",
      detail: "Detail & more",
      caveat: "The backend's own caveat.",
    },
  });
  check(
    "the backend's headline, detail and caveat are shown verbatim, escaped",
    /No sign of AI generation found &lt;b&gt;/.test(own) &&
      /Detail &amp; more/.test(own) &&
      /<div class="caveat">The backend&#39;s own caveat\.<\/div>/.test(own) &&
      !/Checks for fully AI-generated video/.test(own),
    own
  );
  const fallback = show({ state: "result", result: { verdict: "real" } });
  check(
    "a reply without a caveat gets the standing one",
    /Checks for fully AI-generated video/.test(fallback),
    fallback
  );
  const panel = dom.panel();
  check(
    "the panel has a polite live region for screen readers (C-21)",
    panel && /role="status"/.test(panel.innerHTML) && /aria-live="polite"/.test(panel.innerHTML),
    panel && panel.innerHTML
  );
  check(
    "the verdict is announced in the live region",
    panel && panel.querySelector(".sr").textContent === "TrueBeat: No sign of AI generation found",
    panel && panel.querySelector(".sr").textContent
  );
  const cropped = show({
    state: "result",
    result: { verdict: "real" },
    capture: { width: 1920, height: 1080, region: { x0: 150, y0: 120, x1: 1110, y1: 660 } },
  });
  check(
    "the panel says what was recorded and which part was analysed",
    /recorded<\/td><td class="v">1920×1080 px tab; analysed the video&#39;s 960×540 px area/.test(cropped),
    cropped
  );
  const whole = show({
    state: "result",
    result: { verdict: "real" },
    capture: { width: 1280, height: 720, region: null },
    note: "The video moved, changed size or went away during the recording, so the whole tab is analysed.",
  });
  check(
    "a whole-tab analysis says so, with the reason",
    /1280×720 px; analysed the whole tab/.test(whole) && /class="note">The video moved/.test(whole),
    whole
  );
  // C-13: the plan note stays on screen through the capture.
  const capturing = show({
    state: "capturing",
    elapsed: 3.2,
    total: 6,
    message: "Short clip: recording 6 s (what is left of it).",
    size: { width: 1920, height: 1080 },
  });
  check(
    "the capture panel keeps the plan note and shows the capture size",
    /Capturing… 3s \/ 6s/.test(capturing) &&
      /Short clip: recording 6 s/.test(capturing) &&
      /Recording the tab at 1920×1080 px\./.test(capturing),
    capturing
  );
  const err = show({ state: "error", title: "Backend error", message: "x <img src=x onerror=alert(1)>" });
  check(
    "error text is escaped and announced",
    /x &lt;img src=x onerror=alert\(1\)&gt;/.test(err) &&
      panel.querySelector(".sr").textContent.startsWith("TrueBeat: Backend error"),
    err
  );
});

section("2d. overlay.js — the panel follows the page into fullscreen (C-11)");
await guarded(async () => {
  const { dom, show } = loadOverlay();
  show({ state: "analyzing", message: "Analysing on your backend…" });
  const host = dom.host();
  check("the panel starts in the page body", host && host.parentNode === dom.body);
  const player = dom.el("div");
  dom.document.fullscreenElement = player;
  dom.fire("fullscreenchange");
  check("going fullscreen moves the panel into the fullscreen element", host.parentNode === player);
  dom.document.fullscreenElement = null;
  dom.fire("fullscreenchange");
  check("leaving fullscreen moves it back to the page", host.parentNode === dom.body);
  dom.document.fullscreenElement = dom.el("video");
  dom.fire("fullscreenchange");
  check("a fullscreen <video> (which draws no children) leaves the panel in the page", host.parentNode === dom.body);
  // While the tab is recorded, the panel is not painted over a fullscreen
  // player (it would be recorded into the analysed frames); it moves in as
  // soon as the recording ends.
  const stage0 = dom.el("div");
  dom.document.fullscreenElement = null;
  dom.fire("fullscreenchange");
  show({ state: "capturing", elapsed: 1, total: 20 });
  dom.document.fullscreenElement = stage0;
  dom.fire("fullscreenchange");
  check("while recording, the panel stays out of the fullscreen player", host.parentNode === dom.body);
  show({ state: "analyzing", message: "Uploading the recording to your backend…" });
  check("when the recording ends, the panel moves into the fullscreen player", host.parentNode === stage0);
  // A panel first drawn while the page is already fullscreen goes there.
  const second = loadOverlay();
  const stage = second.dom.el("div");
  second.dom.document.fullscreenElement = stage;
  second.show({ state: "analyzing", message: "…" });
  check("a panel created in fullscreen is placed in the fullscreen element", second.dom.host().parentNode === stage);
});

// ------------------------------------------ 3. capture planning (background)

section("3a. background.js — capture length matches the clip, with a start-up margin");
await guarded(async () => {
  loadScript("background.js", { chrome: chromeStub(), fetch, console });
  const plan = globalThis.__trueBeatPlanCapture;
  check("background exposes the capture planner", typeof plan === "function");
  if (plan) {
    const eq = (got, want, name) => check(name, got.seconds === want, JSON.stringify(got));
    eq(plan(20, null), 20, "no <video> found: record as configured");
    eq(plan(20, { duration: NaN, currentTime: 0, loop: false }), 20, "live stream (no duration): record as configured");
    eq(plan(20, { duration: 8, currentTime: 0, loop: false }), 7,
      "an 8 s clip records 7 s (a second of start-up), not 20 across its end");
    eq(plan(20, { duration: 8, currentTime: 2, loop: true }), 5,
      "a looping 8 s clip at 2 s records 5 s: it ends before the loop point (C-13)");
    eq(plan(20, { duration: 8, currentTime: 6, loop: true }), 0,
      "a looping clip about to restart: no capture, a message instead");
    eq(plan(20, { duration: 30, currentTime: 0, loop: false, paused: true }), 0,
      "a paused video is not recorded (it would be 20 s of one frame)");
    eq(plan(20, { duration: 60, currentTime: 50, loop: false }), 9, "records only what is left of the clip, less start-up");
    eq(plan(20, { duration: 60, currentTime: 59, loop: false }), 0,
      "nothing left to record: no capture, a message instead");
    eq(plan(20, { duration: 60, currentTime: 56.5, loop: false }), 3,
      "the start-up margin never plans less than the 3 s minimum");
    eq(plan(20, { duration: 60, currentTime: 39.5, loop: false }), 19,
      "20.5 s left records 19 s, not 20 (start-up would cross the end)");
    check("the shortened plan says why", /Short clip: recording 7 s/.test(plan(20, { duration: 8, currentTime: 0 }).note));
    eq(plan(20, { duration: 120, currentTime: 0, loop: false }), 20, "a long clip records the configured 20 s");
    const ended = plan(20, { duration: 30, currentTime: 30, paused: true, ended: true });
    check("an ended video says it has ended", ended.seconds === 0 && /has ended/.test(ended.note), JSON.stringify(ended));
    const loading = plan(20, { duration: 30, currentTime: 0, paused: false, readyState: 1 });
    check("a video still loading is not recorded", loading.seconds === 0 && /still loading/.test(loading.note), JSON.stringify(loading));
  }
});

section("3b. background.js — the in-page probe (allFrames) and the choice of video (C-10)");
await guarded(async () => {
  loadScript("background.js", { chrome: chromeStub(), fetch, console });
  const choose = globalThis.__trueBeatChooseVideo;
  const locateVideo = globalThis.__trueBeatLocateVideo;
  const stable = globalThis.__trueBeatRegionStable;
  const probeSrc = String(globalThis.__trueBeatProbeFrame);

  // executeScript serialises probeFrame and runs it in the page: run its
  // source with nothing but page globals, so a closure over background.js
  // would fail here too.
  const runProbe = (page) =>
    new Function("document", "window", "getComputedStyle", "globalThis", `return (${probeSrc})();`)(
      page.document,
      page.window,
      page.getComputedStyle,
      page.isolated
    );
  const fakeVideo = (o) => ({
    tagName: "VIDEO",
    paused: false,
    ended: false,
    readyState: 4,
    duration: 30,
    currentTime: 1,
    loop: false,
    ...o,
    getBoundingClientRect: () => ({ left: o.x, top: o.y, width: o.w, height: o.h }),
  });
  const fakeIframe = ({ x, y, w, h, border = 0, padding = 0, scale = 1 }) => ({
    tagName: "IFRAME",
    clientLeft: border,
    clientTop: border,
    clientWidth: w - 2 * border,
    clientHeight: h - 2 * border,
    offsetWidth: w,
    offsetHeight: h,
    padding,
    getBoundingClientRect: () => ({ left: x, top: y, width: w * scale, height: h * scale }),
  });
  const page = ({ videos = [], iframes = [], shadowHosts = [], top = true, parentIsTop = true, w = 1280, h = 720, dpr = 1.5 }) => {
    const root = (vids, extra) => ({
      querySelectorAll: (sel) =>
        sel === "video" ? vids : sel === "iframe" ? iframes : sel === "*" ? [...vids, ...iframes, ...extra] : [],
    });
    const window = { innerWidth: w, innerHeight: h, devicePixelRatio: dpr, visualViewport: { scale: 1 } };
    const topWin = top ? window : { name: "top" };
    window.top = topWin;
    window.parent = top ? window : parentIsTop ? topWin : { name: "middle" };
    return {
      document: root(videos, shadowHosts),
      window,
      getComputedStyle: (el) => {
        const p = `${el.padding || 0}px`;
        return { paddingLeft: p, paddingTop: p, paddingRight: p, paddingBottom: p };
      },
      isolated: {},
    };
  };

  // A paused hero banner no longer blocks the playing article video.
  const banner = fakeVideo({ x: 0, y: 0, w: 1280, h: 500, paused: true });
  const article = fakeVideo({ x: 200, y: 520, w: 400, h: 190, duration: 8, currentTime: 2, loop: true });
  const topPage = page({ videos: [banner, article] });
  const topResult = runProbe(topPage);
  check(
    "the probe reports each video's box and playback state",
    topResult.videos.length === 2 &&
      topResult.videos[1].x === 200 &&
      topResult.videos[1].width === 400 &&
      topResult.videos[0].paused === true &&
      topResult.videos[1].readyState === 4,
    JSON.stringify(topResult.videos)
  );
  check(
    "the probe reports the frame's viewport and device pixel ratio",
    topResult.isTop === true && topResult.innerWidth === 1280 && topResult.innerHeight === 720 && topResult.devicePixelRatio === 1.5,
    JSON.stringify(topResult)
  );
  const again = runProbe(topPage);
  check(
    "the same element keeps its id across probes (found again when the capture ends)",
    again.videos[1].id === topResult.videos[1].id && again.videos[0].id !== again.videos[1].id,
    JSON.stringify([topResult.videos.map((v) => v.id), again.videos.map((v) => v.id)])
  );
  const picked = choose([{ frameId: 0, result: topResult }]);
  check(
    "a paused banner does not block a playing smaller video",
    picked.info && picked.info.paused === false && picked.rect && picked.rect.x === 200 && picked.rect.y === 520,
    JSON.stringify(picked)
  );
  check(
    "the chosen video's clip length still drives the plan (looping 8 s at 2 s -> 5 s)",
    globalThis.__trueBeatPlanCapture(20, picked.info).seconds === 5,
    JSON.stringify(picked.info)
  );
  check(
    "the tab's viewport travels with the choice",
    picked.view && picked.view.innerWidth === 1280 && picked.view.devicePixelRatio === 1.5,
    JSON.stringify(picked.view)
  );
  const onlyPaused = choose([{ frameId: 0, result: runProbe(page({ videos: [banner] })) }]);
  check(
    "with nothing playing, the paused video is reported as paused",
    onlyPaused.info && onlyPaused.info.paused === true && globalThis.__trueBeatPlanCapture(20, onlyPaused.info).seconds === 0,
    JSON.stringify(onlyPaused)
  );
  const nothing = choose([{ frameId: 0, result: runProbe(page({})) }]);
  check(
    "no video: no info, no box, but the viewport is still known",
    nothing.info === null && nothing.rect === null && nothing.view && nothing.view.innerWidth === 1280,
    JSON.stringify(nothing)
  );
  const clipped = choose([{ frameId: 0, result: runProbe(page({ videos: [fakeVideo({ x: -100, y: 600, w: 640, h: 360 })] })) }]);
  check(
    "a video partly off screen is cut to its visible part",
    clipped.rect && clipped.rect.x === 0 && clipped.rect.y === 600 && clipped.rect.width === 540 && clipped.rect.height === 120,
    JSON.stringify(clipped.rect)
  );

  // Open shadow roots are searched too.
  const shadowVideo = fakeVideo({ x: 50, y: 60, w: 320, h: 180 });
  const shadowHost = { shadowRoot: { querySelectorAll: (sel) => (sel === "video" ? [shadowVideo] : []) } };
  const shadowResult = runProbe(page({ shadowHosts: [shadowHost] }));
  check("a <video> inside an open shadow root is found", shadowResult.videos.length === 1 && shadowResult.videos[0].x === 50);

  // A video in a child frame, placed through the one <iframe> that matches.
  const embedTop = runProbe(page({ iframes: [fakeIframe({ x: 300, y: 200, w: 560, h: 315 })] }));
  check(
    "the top frame reports its iframes' content boxes",
    embedTop.iframes.length === 1 && embedTop.iframes[0].x === 300 && embedTop.iframes[0].width === 560 && embedTop.iframes[0].scaled === false,
    JSON.stringify(embedTop.iframes)
  );
  const child = runProbe(page({ top: false, w: 560, h: 315, videos: [fakeVideo({ x: 0, y: 0, w: 560, h: 315 })] }));
  check("a child frame reports itself as a direct child of the top", child.isTop === false && child.parentIsTop === true);
  const embedded = choose([
    { frameId: 0, result: embedTop },
    { frameId: 5, result: child },
  ]);
  check(
    "a playing video inside an iframe gets its box in the tab's viewport",
    embedded.rect && embedded.rect.x === 300 && embedded.rect.y === 200 && embedded.rect.width === 560 && embedded.rect.height === 315 &&
      embedded.target.frameId === 5,
    JSON.stringify(embedded)
  );
  const bordered = runProbe(page({ iframes: [fakeIframe({ x: 300, y: 200, w: 566, h: 321, border: 3 })] }));
  const borderedPick = choose([{ frameId: 0, result: bordered }, { frameId: 5, result: child }]);
  check(
    "an iframe's border is not counted as part of its content",
    borderedPick.rect && borderedPick.rect.x === 303 && borderedPick.rect.y === 203,
    JSON.stringify(borderedPick.rect)
  );
  const twins = runProbe(page({
    iframes: [fakeIframe({ x: 0, y: 0, w: 560, h: 315 }), fakeIframe({ x: 600, y: 0, w: 560, h: 315 })],
  }));
  const ambiguous = choose([{ frameId: 0, result: twins }, { frameId: 5, result: child }]);
  check(
    "two iframes of the same size: the box is unknown (whole tab), the plan still uses the video",
    ambiguous.rect === null && ambiguous.info && ambiguous.info.paused === false,
    JSON.stringify(ambiguous)
  );
  const scaledTop = runProbe(page({ iframes: [fakeIframe({ x: 0, y: 0, w: 560, h: 315, scale: 1.5 })] }));
  check(
    "a CSS-scaled iframe is not used to place its video",
    choose([{ frameId: 0, result: scaledTop }, { frameId: 5, result: child }]).rect === null
  );
  const nested = runProbe(page({ top: false, parentIsTop: false, w: 560, h: 315, videos: [fakeVideo({ x: 0, y: 0, w: 560, h: 315 })] }));
  check(
    "a video two frames deep is left unplaced (whole tab)",
    choose([{ frameId: 0, result: embedTop }, { frameId: 9, result: nested }]).rect === null
  );
  const offscreenEmbed = choose([
    frameResult({ iframes: [{ x: 300, y: 900, width: 560, height: 315, scaled: false }], videos: [videoInfo({ id: 2, x: 0, y: 0, width: 200, height: 112 })] }),
    { frameId: 5, result: { ...child, videos: [videoInfo({ id: 3, x: 0, y: 0, width: 560, height: 315 })] } },
  ]);
  check(
    "a bigger video in an iframe scrolled off screen does not win over a visible one",
    offscreenEmbed.target && offscreenEmbed.target.frameId === 0 && offscreenEmbed.rect && offscreenEmbed.rect.width === 200,
    JSON.stringify(offscreenEmbed)
  );
  const zoomed = choose([frameResult({ zoomed: true, videos: [videoInfo()] })]);
  check("a pinch-zoomed page leaves the box unknown", zoomed.rect === null && zoomed.info !== null);

  // C-10 stability: the box is read again when the capture ends.
  const start = { rect: { x: 100, y: 80, width: 640, height: 360 }, view: { innerWidth: 1280, innerHeight: 720, devicePixelRatio: 1.5 } };
  const at = (dx, dy, dw = 0, dh = 0, view = start.view) => ({
    rect: { x: 100 + dx, y: 80 + dy, width: 640 + dw, height: 360 + dh },
    view,
  });
  check("an unmoved box is stable", stable(start, at(0, 0)) === true);
  check("a 5% shift is within tolerance", stable(start, at(32, 18)) === true);
  check("a move by more than 10% of its width drops the region", stable(start, at(70, 0)) === false);
  check("scrolling it by more than 10% of its height drops the region", stable(start, at(0, -40)) === false);
  check("a resize by more than 10% drops the region", stable(start, at(0, 0, 100, 56)) === false);
  check(
    "a resized window drops the region",
    stable(start, at(0, 0, 0, 0, { innerWidth: 1000, innerHeight: 720, devicePixelRatio: 1.5 })) === false
  );
  check("a video gone at the end drops the region", stable(start, { rect: null, view: start.view }) === false);
  const endFrames = [frameResult({ videos: [videoInfo({ id: 1, y: 300 }), videoInfo({ id: 2, x: 0, y: 0, width: 1280, height: 700 })] })];
  const found = locateVideo(endFrames, { frameId: 0, id: 1 });
  check(
    "the same element is found again by its id, not by size",
    found.rect && found.rect.y === 300 && found.rect.width === 640,
    JSON.stringify(found)
  );
  const endedAtStop = locateVideo([frameResult({ videos: [videoInfo({ id: 1, paused: true, ended: true })] })], { frameId: 0, id: 1 });
  check("a clip that ended right at the stop is still found where it was", stable(start, endedAtStop) === true, JSON.stringify(endedAtStop));
});

// ------------------------------------------ 4. offscreen: capture and region

section("4a. offscreen.js — capture size, bitrate and the CSS->recording-pixel region");
await guarded(async () => {
  offscreenEnv();
  const size = globalThis.__trueBeatCaptureSize;
  const bitrate = globalThis.__trueBeatBitrate;
  const toCapture = globalThis.__trueBeatRegionToCapture;
  const constraints = globalThis.__trueBeatCaptureConstraints;
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  check("a 1280x720 tab at 1.5x records 1920x1080 device pixels",
    same(size({ innerWidth: 1280, innerHeight: 720, devicePixelRatio: 1.5 }), { width: 1920, height: 1080 }));
  check("a 2560x1440 tab is scaled down to fit 1920x1080",
    same(size({ innerWidth: 2560, innerHeight: 1440, devicePixelRatio: 1 }), { width: 1920, height: 1080 }));
  check("a 1440x789 tab at 2x keeps its shape inside 1920x1080 (even sizes)",
    same(size({ innerWidth: 1440, innerHeight: 789, devicePixelRatio: 2 }), { width: 1920, height: 1052 }),
    JSON.stringify(size({ innerWidth: 1440, innerHeight: 789, devicePixelRatio: 2 })));
  check("a tall tab is fitted by its height",
    same(size({ innerWidth: 1000, innerHeight: 800, devicePixelRatio: 2 }), { width: 1350, height: 1080 }));
  check("a small tab is not upscaled",
    same(size({ innerWidth: 800, innerHeight: 600, devicePixelRatio: 1 }), { width: 800, height: 600 }));
  check("an unknown viewport gives no size (old behaviour)", size(null) === null && size({}) === null);
  check("6 Mb/s at 1080p", bitrate(1920, 1080) === 6_000_000);
  check("proportionally less below it", bitrate(1280, 720) === Math.round((6_000_000 * 1280 * 720) / (1920 * 1080)));
  check("never under 1 Mb/s", bitrate(640, 360) === 1_000_000);

  const view = { innerWidth: 1280, innerHeight: 720, devicePixelRatio: 1.5 };
  const cap = { width: 1920, height: 1080 };
  check(
    "a CSS box is scaled by capture width / innerWidth into recording pixels",
    same(toCapture({ x: 100, y: 80, width: 640, height: 360 }, view, cap), { x0: 150, y0: 120, x1: 1110, y1: 660 }),
    JSON.stringify(toCapture({ x: 100, y: 80, width: 640, height: 360 }, view, cap))
  );
  check(
    "the region is clamped to the frame",
    same(toCapture({ x: -50, y: 600, width: 400, height: 300 }, view, cap), { x0: 0, y0: 900, x1: 525, y1: 1080 }),
    JSON.stringify(toCapture({ x: -50, y: 600, width: 400, height: 300 }, view, cap))
  );
  check(
    "a box covering the whole viewport is the whole frame",
    same(toCapture({ x: 0, y: 0, width: 1280, height: 720 }, view, cap), { x0: 0, y0: 0, x1: 1920, y1: 1080 })
  );
  check(
    "a recording whose shape differs from the viewport's is not guessed at",
    toCapture({ x: 100, y: 80, width: 640, height: 360 }, { innerWidth: 1280, innerHeight: 1000 }, cap) === null
  );
  check("a box under 16 px is no region", toCapture({ x: 10, y: 10, width: 8, height: 8 }, view, cap) === null);
  check("a box entirely outside the frame is no region", toCapture({ x: 2000, y: 10, width: 100, height: 100 }, view, cap) === null);
  check("unknown sizes give no region", toCapture({ x: 0, y: 0, width: 10, height: 10 }, null, cap) === null);
  const pinned = constraints("sid-1", { width: 1920, height: 1080 });
  check(
    "the capture is pinned to that size (min = max) at up to 30 fps",
    same(pinned, {
      audio: false,
      video: { mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: "sid-1", minWidth: 1920, maxWidth: 1920, minHeight: 1080, maxHeight: 1080, maxFrameRate: 30 } },
    }),
    JSON.stringify(pinned)
  );
  check(
    "without a size the constraints are the 1.2 ones",
    same(constraints("sid-1", null), { audio: false, video: { mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: "sid-1" } } })
  );
});

section("4b. offscreen.js — tab-capture wiring");
await guarded(async () => {
  // Static: the real API is navigator.mediaDevices, never chrome.mediaStream.
  const src = readFileSync(join(EXT, "offscreen.js"), "utf-8");
  check("offscreen opens the stream via navigator.mediaDevices.getUserMedia", src.includes("navigator.mediaDevices.getUserMedia"));
  check("offscreen never touches the non-existent chrome.mediaStream", !src.includes("chrome.mediaStream"));

  // With a viewport: the size is requested, the bitrate set, the region mapped.
  const env = offscreenEnv();
  const answer = await env.send({
    type: "OFFSCREEN_START",
    streamId: "sid-123",
    tabId: 7,
    checkId: "c-1",
    seconds: 20,
    backend: "http://127.0.0.1:9",
    note: "",
    view: { innerWidth: 1280, innerHeight: 720, devicePixelRatio: 1.5 },
    rect: { x: 100, y: 80, width: 640, height: 360 },
  });
  const c = env.gumCalls[0] || {};
  const mandatory = (c.video && c.video.mandatory) || {};
  check(
    "the streamId flows through as chromeMediaSourceId",
    c.audio === false && mandatory.chromeMediaSource === "tab" && mandatory.chromeMediaSourceId === "sid-123",
    JSON.stringify(c)
  );
  check(
    "the tab's device-pixel size is requested: min = max 1920x1080 at 30 fps (C-06)",
    mandatory.minWidth === 1920 && mandatory.maxWidth === 1920 && mandatory.minHeight === 1080 &&
      mandatory.maxHeight === 1080 && mandatory.maxFrameRate === 30,
    JSON.stringify(mandatory)
  );
  const rec = env.recorders[0];
  check(
    "MediaRecorder gets 6 Mb/s for 1080p",
    rec && rec.options.videoBitsPerSecond === 6_000_000 && /webm/.test(rec.options.mimeType || ""),
    JSON.stringify(rec && rec.options)
  );
  check(
    "a running capture answers ok with the measured size and the region in recording pixels",
    answer && answer.ok === true &&
      JSON.stringify(answer.size) === JSON.stringify({ width: 1920, height: 1080 }) &&
      JSON.stringify(answer.region) === JSON.stringify({ x0: 150, y0: 120, x1: 1110, y1: 660 }) &&
      !env.sent.some((m) => m.type === "CAPTURE_ERROR"),
    JSON.stringify(answer)
  );
  const video = env.videos[0];
  check(
    "the black-frame sampler plays the tab stream in a muted <video> (C-02)",
    video && video.srcObject === env.stream && video.muted === true && video.playing === true,
    JSON.stringify(video && { muted: video.muted, playing: video.playing })
  );

  // Without a viewport (a page that cannot be probed): the 1.2 behaviour.
  const plain = offscreenEnv();
  await plain.send({ type: "OFFSCREEN_START", streamId: "sid-9", tabId: 7, seconds: 20, backend: "http://127.0.0.1:9" });
  const pm = (plain.gumCalls[0] && plain.gumCalls[0].video.mandatory) || {};
  check(
    "an unknown viewport asks for no size and sets no bitrate",
    !("minWidth" in pm) && !("maxFrameRate" in pm) && plain.recorders[0] && !("videoBitsPerSecond" in plain.recorders[0].options),
    JSON.stringify({ pm, options: plain.recorders[0] && plain.recorders[0].options })
  );

  // A browser that refuses the size constraints still records, at its default size.
  const refusals = [];
  const fallback = offscreenEnv({
    gum: async (constraints) => {
      refusals.push(rtMessage(constraints));
      if (constraints.video.mandatory.minWidth) throw Object.assign(new Error("Constraints not satisfied"), { name: "OverconstrainedError" });
      return fallback.stream;
    },
    settings: { width: 960, height: 540 },
  });
  const fbAnswer = await fallback.send({
    type: "OFFSCREEN_START", streamId: "sid-2", tabId: 7, seconds: 20, backend: "http://127.0.0.1:9",
    view: { innerWidth: 1280, innerHeight: 720, devicePixelRatio: 1.5 }, rect: { x: 100, y: 80, width: 640, height: 360 },
  });
  check(
    "refused size constraints fall back to the default size, region mapped on the real size",
    refusals.length === 2 && fbAnswer && fbAnswer.ok === true &&
      JSON.stringify(fbAnswer.region) === JSON.stringify({ x0: 75, y0: 60, x1: 555, y1: 330 }),
    JSON.stringify({ refusals: refusals.length, fbAnswer })
  );

  // The browser reports one size and delivers another: the region follows the frames.
  const sizeBackend = fakeBackend({ stream: streamOf([{ end: true }]) });
  const liar = offscreenEnv({ fetchImpl: sizeBackend.fetch, settings: { width: 1920, height: 1080 }, delivered: { width: 960, height: 540 } });
  await liar.send({
    type: "OFFSCREEN_START", streamId: "sid-3", tabId: 7, checkId: "c-3", seconds: 4, backend: "http://127.0.0.1:9",
    view: { innerWidth: 1280, innerHeight: 720, devicePixelRatio: 1.5 }, rect: { x: 100, y: 80, width: 640, height: 360 },
  });
  await liar.clock.advance(6000);
  const uploaded = liar.of("CAPTURE_UPLOADED")[0];
  check(
    "the region is mapped onto the frame size really delivered, not the one reported",
    uploaded && JSON.stringify(uploaded.capture) === JSON.stringify({ width: 960, height: 540 }) &&
      JSON.stringify(uploaded.region) === JSON.stringify({ x0: 75, y0: 60, x1: 555, y1: 330 }),
    JSON.stringify(uploaded && { capture: uploaded.capture, region: uploaded.region })
  );
  const lastProgress = liar.of("PROGRESS").slice(-1)[0];
  check(
    "the panel is shown the measured size",
    lastProgress && JSON.stringify(lastProgress.payload.size) === JSON.stringify({ width: 960, height: 540 }),
    JSON.stringify(lastProgress && lastProgress.payload)
  );

  // A second recording while one runs is refused, honestly.
  const busy = await env.send({ type: "OFFSCREEN_START", streamId: "sid-x", tabId: 8, checkId: "c-2", seconds: 20, backend: "http://127.0.0.1:9" });
  check(
    "a second recording while one runs is refused with a readable message",
    busy && busy.ok === false && /Another tab is still being recorded/.test(busy.error || ""),
    JSON.stringify(busy)
  );

  // Guard: a context without mediaDevices must fail with a readable message.
  const bare = offscreenEnv();
  bare.globals.navigator.mediaDevices = undefined;
  const listeners2 = [];
  const sent2 = [];
  loadScript("offscreen.js", {
    ...bare.globals,
    navigator: {},
    chrome: chromeStub({
      runtime: {
        onMessage: { addListener: (fn) => listeners2.push(fn) },
        sendMessage: async (m) => {
          sent2.push(rtMessage(m));
          return {};
        },
        id: "test",
      },
    }),
  });
  let response2 = null;
  listeners2[0]({ type: "OFFSCREEN_START", streamId: "sid-x", tabId: 8, seconds: 20 }, null, (r) => {
    response2 = r;
  });
  await turn();
  check(
    "a missing mediaDevices yields the readable guard message",
    response2 && response2.ok === false && /Tab capture is unavailable/.test(sent2.map((m) => m.message || "").join(" ")) &&
      sent2.every((m) => m.tabId === 8),
    JSON.stringify({ response: response2, sent: sent2 })
  );
});

section("4c. offscreen.js — the black-frame guard reads real frames (C-02)");
await guarded(async () => {
  offscreenEnv();
  const isBlack = globalThis.__trueBeatIsBlackFrame;
  const stats = globalThis.__trueBeatLumaStats;
  check("offscreen exposes the black-frame test", typeof isBlack === "function");
  if (isBlack) {
    check("a fully black frame is black", isBlack(2, 1) === true);
    check("a dark but moving frame is not black", isBlack(2, 40) === false);
    check("a bright frame is not black", isBlack(180, 5) === false);
    check("a normal frame is not black", isBlack(90, 25) === false);
  }
  check("offscreen exposes the luma statistics", typeof stats === "function");
  if (typeof stats === "function") {
    const black = stats(pixels(16, 9, "black"));
    const normal = stats(pixels(16, 9, "normal"));
    check("luma statistics of a black frame: dark and flat", black.mean < 12 && black.std < 6, JSON.stringify(black));
    check("luma statistics of a picture: neither", normal.mean > 12 && normal.std > 6, JSON.stringify(normal));
  }

  // Drive the real sampler: a DRM stream hands the capturer black frames.
  const backend = fakeBackend({ stream: streamOf([{ end: true }]) });
  const env = offscreenEnv({ fetchImpl: backend.fetch });
  env.frame = "black";
  await env.send({ type: "OFFSCREEN_START", streamId: "sid-drm", tabId: 7, checkId: "c-drm", seconds: 20, backend: "http://127.0.0.1:9" });
  await env.clock.advance(1500);
  check("two black samples are not enough to give up", env.recorders[0].state === "recording");
  await env.clock.advance(1000);
  const drm = env.of("CAPTURE_ERROR")[0];
  check(
    "three black samples stop the recorder long before the planned 20 s",
    env.recorders[0].state === "inactive" && env.clock.now() - Date.UTC(2026, 10, 7, 12) < 3000
  );
  check(
    "the sampler drew the <video> element, never the MediaStreamTrack",
    env.drawn.length >= 3 && env.drawn.every((d) => d === env.videos[0] && d.tagName === "VIDEO") && !env.drawn.includes(env.track),
    String(env.drawn.length)
  );
  check(
    "a black capture says DRM or black frames, to the right tab",
    drm && drm.tabId === 7 && drm.checkId === "c-drm" &&
      /protected stream \(DRM\) or black frames/.test(drm.message) && /Nothing was uploaded/.test(drm.message),
    JSON.stringify(drm)
  );
  await env.clock.advance(30000);
  check("nothing is uploaded from a black capture", backend.uploads().length === 0 && env.of("CAPTURE_UPLOADED").length === 0);
  check("the tab stream is released", env.track.stopped === true && env.videos[0].srcObject === null);
  check(
    "without a known video box the guard looks at the whole tab",
    JSON.stringify(env.drawArgs[0]) === JSON.stringify([0, 0, 1920, 1080, 0, 0, 160, 90]),
    JSON.stringify(env.drawArgs[0])
  );

  // With the video's box known, the guard looks at what will be analysed: a
  // protected video is drawn black inside a page that is not.
  const boxed = offscreenEnv({ fetchImpl: fakeBackend({ stream: streamOf([{ end: true }]) }).fetch });
  boxed.frame = "black";
  await boxed.send({
    type: "OFFSCREEN_START", streamId: "sid-box", tabId: 7, checkId: "c-box", seconds: 20, backend: "http://127.0.0.1:9",
    view: { innerWidth: 1280, innerHeight: 720, devicePixelRatio: 1.5 }, rect: { x: 100, y: 80, width: 640, height: 360 },
  });
  await boxed.clock.advance(2500);
  check(
    "with the video's box known, the guard samples that area of the frame",
    JSON.stringify(boxed.drawArgs[0]) === JSON.stringify([150, 120, 960, 540, 0, 0, 160, 90]) &&
      /DRM/.test((boxed.of("CAPTURE_ERROR")[0] || {}).message || ""),
    JSON.stringify(boxed.drawArgs[0])
  );

  // A normal picture records the full plan and is uploaded.
  const backend2 = fakeBackend({ stream: streamOf([{ end: true }]) });
  const env2 = offscreenEnv({ fetchImpl: backend2.fetch });
  await env2.send({ type: "OFFSCREEN_START", streamId: "sid-ok", tabId: 7, checkId: "c-ok", seconds: 6, backend: "http://127.0.0.1:9", note: "Short clip: recording 6 s (what is left of it)." });
  await env2.clock.advance(5000);
  check("a normal picture keeps recording", env2.recorders[0].state === "recording" && env2.of("CAPTURE_ERROR").length === 0);
  await env2.clock.advance(1500);
  const progress = env2.of("PROGRESS");
  check(
    "every PROGRESS carries the plan note, so it stays on screen (C-13)",
    progress.length >= 10 && progress.every((p) => p.payload.message === "Short clip: recording 6 s (what is left of it)."),
    JSON.stringify(progress.slice(-1))
  );
  check(
    "a finished recording reports CAPTURE_STOPPED, then is uploaded with its path",
    env2.of("CAPTURE_STOPPED").length === 1 && backend2.uploads().length === 1 &&
      env2.of("CAPTURE_UPLOADED")[0] && env2.of("CAPTURE_UPLOADED")[0].serverPath === "/tmp/gradio/abc123/truebeat-capture.webm",
    JSON.stringify(env2.sent.map((m) => m.type))
  );
});

section("4d. offscreen.js — a failed or partial capture is never uploaded (C-04)");
await guarded(async () => {
  const start = async (env, extra = {}) =>
    env.send({ type: "OFFSCREEN_START", streamId: "sid", tabId: 7, checkId: "c-1", seconds: 20, backend: "http://127.0.0.1:9", ...extra });

  // The tab's track ends early (tab closed, reloaded).
  const b1 = fakeBackend({ stream: streamOf([{ end: true }]) });
  const ended = offscreenEnv({ fetchImpl: b1.fetch });
  await start(ended);
  await ended.clock.advance(5000);
  ended.track.handlers.ended();
  await ended.clock.advance(30000);
  const e1 = ended.of("CAPTURE_ERROR")[0];
  check(
    "a track that ends early is reported to the right tab, not to null",
    e1 && e1.tabId === 7 && e1.checkId === "c-1" && /stopped being captured/.test(e1.message) && /Nothing was uploaded/.test(e1.message),
    JSON.stringify(e1)
  );
  check("the partial recording is not uploaded", b1.uploads().length === 0 && ended.of("CAPTURE_UPLOADED").length === 0);

  // The recorder errors.
  const b2 = fakeBackend({ stream: streamOf([{ end: true }]) });
  const broken = offscreenEnv({ fetchImpl: b2.fetch });
  await start(broken);
  await broken.clock.advance(4000);
  broken.recorders[0].onerror({ error: { message: "encoder crashed" } });
  await broken.clock.advance(30000);
  const e2 = broken.of("CAPTURE_ERROR")[0];
  check(
    "a recorder error is reported honestly and nothing is uploaded",
    e2 && e2.tabId === 7 && /recorder failed \(encoder crashed\)/.test(e2.message) && b2.uploads().length === 0,
    JSON.stringify(e2)
  );

  // The recorder stops by itself part-way (nobody asked it to).
  const b3 = fakeBackend({ stream: streamOf([{ end: true }]) });
  const quitter = offscreenEnv({ fetchImpl: b3.fetch });
  await start(quitter);
  await quitter.clock.advance(10000);
  quitter.recorders[0].stop();
  await quitter.clock.advance(30000);
  const e3 = quitter.of("CAPTURE_ERROR")[0];
  check(
    "a recording that stops on its own part-way is refused",
    e3 && /stopped on its own after 10\.0 s of the planned 20 s/.test(e3.message) && b3.uploads().length === 0,
    JSON.stringify(e3)
  );

  // A plan under the 3 s minimum is recorded but never uploaded.
  const b4 = fakeBackend({ stream: streamOf([{ end: true }]) });
  const short = offscreenEnv({ fetchImpl: b4.fetch });
  await start(short, { seconds: 2 });
  await short.clock.advance(10000);
  const e4 = short.of("CAPTURE_ERROR")[0];
  check(
    "a recording under MIN_CAPTURE_SECONDS (3 s) is refused",
    e4 && e4.title === "Recording too short" && /at least 3 s/.test(e4.message) && b4.uploads().length === 0,
    JSON.stringify(e4)
  );

  // After a failure the recorder is free again.
  const again = await start(ended, { checkId: "c-2" });
  check("after a failed capture the next one can start", again && again.ok === true, JSON.stringify(again));
});

// ------------------------------------------- 4e. the analysis wait (C-01/C-05)

section("4e. offscreen.js — the analysis wait on Gradio's real event stream");
await guarded(async () => {
  const run = async (stream, extra = {}, opts = {}) => {
    const clock = makeClock();
    const backend = fakeBackend({ clock, stream, endpoints: opts.endpoints });
    const env = offscreenEnv({ clock, fetchImpl: backend.fetch });
    const posted = await analyzeOnce(env, extra);
    return { env, backend, posted };
  };

  // (a) heartbeats, then the result.
  {
    const { backend, posted } = await run(
      streamOf([{ text: HEARTBEAT, delay: 15000 }, { text: HEARTBEAT, delay: 15000 }, { text: completeEvent(VERDICT), delay: 4000 }, { end: true }])
    );
    check(
      "heartbeats are ignored and `event: complete` delivers the verdict",
      posted && posted.type === "ANALYSIS_RESULT" && posted.result.verdict === "fake" &&
        posted.result.headline === "Possible face-swap" && posted.tabId === 7 && posted.checkId === "c-1",
      JSON.stringify(posted)
    );
    check("one call, one stream", backend.posts().length === 1 && backend.gets().length === 1);
  }

  // (b) `event: error` with {"error": null} (show_error=False): terminal.
  {
    const { backend, posted } = await run(streamOf([{ text: HEARTBEAT }, { text: errorEvent({ error: null }) }, { hang: true }]));
    check(
      "`event: error` with no message ends the wait with an honest error (no 'Analysing…' forever)",
      posted && posted.type === "CAPTURE_ERROR" && posted.title === "Backend error" &&
        /^The backend reported an error but gave no details/.test(posted.message),
      JSON.stringify(posted)
    );
    check("an explicit backend error is not retried", backend.posts().length === 1 && backend.gets().length === 1);
  }

  // (c) `event: error` with the message (show_error=True).
  {
    const { posted } = await run(streamOf([{ text: errorEvent({ error: "analysis blew up", visible: true }) }]));
    check(
      "an explicit backend error surfaces its own readable message",
      posted && posted.type === "CAPTURE_ERROR" && posted.message === "The backend reported an error: analysis blew up",
      JSON.stringify(posted)
    );
  }

  // (d) Gradio's UnexpectedErrorMessage: data is a bare JSON string.
  {
    const { posted } = await run(streamOf([{ text: errorEvent("Server stopped unexpectedly.") }]));
    check(
      "an error whose data is a plain string is shown too",
      posted && posted.type === "CAPTURE_ERROR" && /Server stopped unexpectedly\./.test(posted.message),
      JSON.stringify(posted)
    );
  }

  // (e) A stream that never sends a byte: the watchdog, then a re-POST.
  {
    const { env, backend, posted } = await run(({ attempt, signal, clock }) =>
      streamOf(attempt === 1 ? [{ hang: true }] : [{ text: HEARTBEAT, delay: 15000 }, { text: completeEvent(VERDICT), delay: 2000 }, { end: true }])({ signal, clock })
    );
    check(
      "a silent stream is abandoned by the inactivity watchdog and the call POSTed again",
      posted && posted.type === "ANALYSIS_RESULT" && backend.posts().length === 2,
      JSON.stringify({ posted, posts: backend.posts().length })
    );
    const gets = backend.gets().map((g) => g.url);
    check(
      "the dead event is never re-read: the second wait reads the new event",
      gets.length === 2 && gets[0].endsWith("/ev-1") && gets[1].endsWith("/ev-2"),
      JSON.stringify(gets)
    );
    check(
      "the re-POST analyses the same uploaded file",
      backend.posts().every((p) => JSON.parse(p.body).data[0].path === "/tmp/gradio/abc123/truebeat-capture.webm")
    );
    check(
      "the panel is told the analysis is being asked for again",
      env.of("ANALYSIS_STATUS").some((m) => /attempt 2 of 3/.test(m.message)),
      JSON.stringify(env.of("ANALYSIS_STATUS"))
    );
  }

  // (f) Heartbeats forever: alive, so no re-POST, but the deadline holds mid-stream.
  {
    const clockStart = Date.UTC(2026, 10, 7, 12);
    const { env, backend, posted } = await run(streamOf(() => ({ text: HEARTBEAT, delay: 15000 })));
    check(
      "heartbeats keep the stream alive (no watchdog re-POST)",
      backend.posts().length === 1 && backend.gets().length === 1,
      String(backend.posts().length)
    );
    check(
      "the overall deadline is enforced mid-stream",
      posted && posted.type === "CAPTURE_ERROR" && /no result within 6 minutes/.test(posted.message) &&
        env.clock.now() - clockStart <= 6 * 60 * 1000 + 1000,
      JSON.stringify({ posted, elapsed: env.clock.now() - clockStart })
    );
  }

  // (g) A stream that drops mid-way: re-POST, then the verdict.
  {
    const { backend, posted } = await run(({ attempt, signal, clock }) =>
      streamOf(attempt === 1 ? [{ text: HEARTBEAT }, { drop: true }] : [{ text: completeEvent(VERDICT) }, { end: true }])({ signal, clock })
    );
    check(
      "a dropped stream is retried by POSTing again, and the verdict arrives",
      posted && posted.type === "ANALYSIS_RESULT" && posted.result.verdict === "fake" && backend.posts().length === 2,
      JSON.stringify(posted)
    );
  }

  // (h) Every attempt drops: a readable message, never Chrome's raw error.
  {
    const { backend, posted } = await run(streamOf([{ drop: true }]));
    check(
      "retries exhausted produce the readable drop message",
      posted && posted.type === "CAPTURE_ERROR" && posted.title === "Backend error" &&
        posted.message ===
          "The connection to your backend dropped while analysing (3 attempts). Is `trubeat serve` still running?" &&
        backend.posts().length === 3,
      JSON.stringify(posted)
    );
    check("the readable message never carries the raw network error text", posted && !/network error/i.test(posted.message || ""));
  }

  // (i) The retry finds the file gone (the dead first run deleted it).
  {
    const gone = {
      verdict: "inconclusive",
      headline: "Inconclusive: could not read the video",
      layers: { file: "not a readable file: truebeat-capture.webm" },
      warnings: ["E_UNREADABLE"],
    };
    const { posted } = await run(({ attempt, signal, clock }) =>
      streamOf(attempt === 1 ? [{ drop: true }] : [{ text: completeEvent(gone) }])({ signal, clock })
    );
    check(
      "a re-run that finds the recording deleted says so instead of 'could not read the video'",
      posted && posted.type === "CAPTURE_ERROR" && /no longer had the recording/.test(posted.message),
      JSON.stringify(posted)
    );
  }

  // (i2) The retry's POST fails in the backend (Gradio cannot find the upload any
  // more: kept about two minutes, or the server restarted): the same readable text.
  {
    const { posted } = await run(({ attempt, signal, clock }) =>
      streamOf(attempt === 1 ? [{ drop: true }] : [{ text: errorEvent({ error: "File not found", visible: true }) }])({
        signal,
        clock,
      })
    );
    check(
      "an error on the re-POST says the recording is gone, with the backend's own message",
      posted && posted.type === "CAPTURE_ERROR" && /no longer had the recording/.test(posted.message) &&
        /File not found/.test(posted.message),
      JSON.stringify(posted)
    );
  }

  // (j) A chunk boundary in the middle of an event, CRLF line endings.
  {
    const text = completeEvent(VERDICT).replace(/\n/g, "\r\n");
    const cut = Math.floor(text.length / 2);
    const { posted } = await run(streamOf([{ text: text.slice(0, cut) }, { text: text.slice(cut) }, { end: true }]));
    check(
      "an event split across chunks (with CRLF) is reassembled",
      posted && posted.type === "ANALYSIS_RESULT" && posted.result.score === 0.91,
      JSON.stringify(posted)
    );
  }

  // (k) The endpoint: the region goes to /analyze_video_region, else /analyze_video.
  {
    const region = { x0: 150, y0: 120, x1: 1110, y1: 660 };
    const ok = streamOf([{ text: completeEvent(VERDICT) }, { end: true }]);
    const withRegion = await run(ok, { region, capture: { width: 1920, height: 1080 } });
    const post = withRegion.backend.posts()[0];
    const data = post && JSON.parse(post.body).data;
    check(
      "a known region is sent to /analyze_video_region as [FileData, region JSON]",
      post && post.url === "http://127.0.0.1:9/gradio_api/call/analyze_video_region" && data.length === 2 &&
        data[0].path === "/tmp/gradio/abc123/truebeat-capture.webm" && data[0].meta._type === "gradio.FileData" &&
        JSON.stringify(JSON.parse(data[1])) === JSON.stringify(region),
      JSON.stringify(post && { url: post.url, data })
    );
    check(
      "the result says the video's region was analysed",
      withRegion.posted && JSON.stringify(withRegion.posted.capture) === JSON.stringify({ width: 1920, height: 1080, region }),
      JSON.stringify(withRegion.posted && withRegion.posted.capture)
    );
    const noRegion = await run(ok, { region: null, capture: { width: 1920, height: 1080 } });
    const p2 = noRegion.backend.posts()[0];
    check(
      "no region: /analyze_video with the file alone",
      p2 && p2.url.endsWith("/gradio_api/call/analyze_video") && JSON.parse(p2.body).data.length === 1 &&
        noRegion.posted.capture.region === null
    );
    const old = await run(ok, { region, capture: { width: 1920, height: 1080 } }, { endpoints: { "/analyze_video": {} } });
    const p3 = old.backend.posts()[0];
    check(
      "an older backend without the region endpoint gets /analyze_video, and the panel is told",
      p3 && p3.url.endsWith("/gradio_api/call/analyze_video") && JSON.parse(p3.body).data.length === 1 &&
        /no analyze_video_region endpoint/.test(old.posted.note) && old.posted.capture.region === null,
      JSON.stringify(old.posted)
    );
    const onlyRegion = await run(ok, { region: null }, { endpoints: { "/analyze_video_region": {} } });
    const p4 = onlyRegion.backend.posts()[0];
    check(
      "a backend with only the region endpoint gets \"\" (the whole frame)",
      p4 && p4.url.endsWith("/analyze_video_region") && JSON.parse(p4.body).data[1] === "",
      JSON.stringify(p4 && p4.body)
    );
    const none = await run(ok, {}, { endpoints: { "/other": {} } });
    check(
      "a backend without either endpoint is named as such",
      none.posted && none.posted.type === "CAPTURE_ERROR" && /does not expose the analyze_video endpoint \(found: \/other\)/.test(none.posted.message),
      JSON.stringify(none.posted)
    );
  }

  // (l) The SSE parser on its own.
  {
    const parse = globalThis.__trueBeatSseParser();
    const evs = [
      ...parse.push(": comment\nevent: heartbeat\ndata: null\n\nevent: comp"),
      ...parse.push("lete\ndata: [\"{}\"]\n"),
      ...parse.push("\n"),
    ];
    check(
      "the SSE parser names events and joins split lines",
      evs.length === 2 && evs[0].event === "heartbeat" && evs[1].event === "complete" && evs[1].data === '["{}"]',
      JSON.stringify(evs)
    );
    const tail = globalThis.__trueBeatSseParser();
    tail.push("event: complete\ndata: [1]");
    const flushed = tail.end();
    check("an event cut off by the end of the stream is still read", flushed.length === 1 && flushed[0].data === "[1]");
  }
});

// ------------------------------------------------ 5. background: per-tab lock

/** background.js alone: what it sends out and pushes into the panel. */
function backgroundEnv({
  answer = { ok: true },
  frames = [],
  clock = makeClock(),
  chrome: shared = null,
  rejectSends = 0,
  refuseInject = false,
  refuseAllFrames = false,
} = {}) {
  const listeners = [];
  const out = [];
  const overlays = [];
  const tabCalls = [];
  const removed = [];
  const insertCSS = [];
  let rejections = rejectSends;
  const chrome =
    shared ||
    chromeStub({
      runtime: {
        id: "test",
        onMessage: { addListener: (fn) => listeners.push(fn) },
        sendMessage: async (m) => {
          out.push(rtMessage(m));
          if (m.type === "OFFSCREEN_START") return typeof answer === "function" ? answer(m) : answer;
          return { ok: true };
        },
      },
      tabs: {
        sendMessage: async (tabId, m) => {
          tabCalls.push({ kind: "send", tabId, state: m.payload.state });
          if (rejections > 0) {
            rejections -= 1;
            throw new Error("Could not establish connection. Receiving end does not exist.");
          }
          overlays.push({ tabId, ...rtMessage(m.payload) });
          return { ok: true };
        },
        query: async () => [],
        onRemoved: { addListener: (fn) => removed.push(fn) },
      },
      scripting: {
        executeScript: async (details) => {
          if (details.files) {
            tabCalls.push({ kind: "inject", tabId: details.target.tabId, files: details.files, allFrames: !!details.target.allFrames });
            if (refuseInject) throw new Error("Cannot access contents of the page.");
            return [{ frameId: 0, result: null }];
          }
          tabCalls.push({ kind: "probe", tabId: details.target.tabId, allFrames: !!details.target.allFrames });
          if (refuseAllFrames && details.target.allFrames) throw new Error("Cannot access a chrome-extension:// URL of different extension");
          const all = rtMessage(typeof frames === "function" ? frames() : frames);
          return details.target.allFrames ? all : all.filter((f) => f.frameId === 0);
        },
        insertCSS: async (d) => {
          insertCSS.push(d);
          throw new Error("No source code or file specified.");
        },
      },
    });
  if (shared) {
    // a second worker instance on the same browser state
    const prev = shared.runtime.onMessage.addListener;
    shared.runtime.onMessage.addListener = (fn) => listeners.push(fn);
    loadScript("background.js", { chrome: shared, fetch, console, Date: clock.Date, AbortSignal });
    shared.runtime.onMessage.addListener = prev;
  } else {
    loadScript("background.js", { chrome, fetch, console, Date: clock.Date, AbortSignal });
  }
  const dispatch = (m) =>
    new Promise((resolve) => {
      listeners[listeners.length - 1](rtMessage(m), { id: "test" }, (r) => resolve(rtMessage(r)));
    });
  return {
    chrome,
    clock,
    dispatch,
    out,
    overlays,
    tabCalls,
    removed,
    insertCSS,
    starts: () => out.filter((m) => m.type === "OFFSCREEN_START"),
  };
}

section("5a. background.js — one check per tab, locked until its result or an error (C-03, C-19)");
await guarded(async () => {
  const frames = [frameResult({ videos: [videoInfo()] })];
  const env = backgroundEnv({ frames });
  const first = await env.dispatch({ type: "CHECK_TAB", tabId: 7 });
  check("a check starts a capture", first && first.ok === true && env.starts().length === 1, JSON.stringify(first));
  const start = env.starts()[0];
  check(
    "the capture request carries the tab's viewport, the video's box and a check id",
    start.view && start.view.innerWidth === 1280 && start.rect && start.rect.x === 100 && typeof start.checkId === "string" &&
      start.target && start.target.id === 1,
    JSON.stringify(start)
  );
  const dup = await env.dispatch({ type: "CHECK_TAB", tabId: 7 });
  check(
    "a second click while capturing says a check is running (popup), without a second capture",
    dup && dup.ok === false && /already running in this tab/.test(dup.error) && env.starts().length === 1,
    JSON.stringify(dup)
  );
  const racing = backgroundEnv({ frames });
  const both = await Promise.all([
    racing.dispatch({ type: "CHECK_TAB", tabId: 7 }),
    racing.dispatch({ type: "CHECK_TAB", tabId: 7 }),
  ]);
  check(
    "two clicks at once start one capture, not two",
    racing.starts().length === 1 && both.filter((r) => r && r.ok).length === 1,
    JSON.stringify(both)
  );
  const other = await env.dispatch({ type: "CHECK_TAB", tabId: 8 });
  check(
    "another tab cannot start recording while this one records",
    other && other.ok === false && /Another tab is being recorded/.test(other.error),
    JSON.stringify(other)
  );
  await env.dispatch({ type: "CAPTURE_ERROR", tabId: 7, checkId: start.checkId, title: "Can't capture this tab", message: "The track ended." });
  check(
    "CAPTURE_ERROR shows the error in the tab's panel",
    env.overlays.some((o) => o.tabId === 7 && o.state === "error" && o.message === "The track ended.")
  );
  const retry = await env.dispatch({ type: "CHECK_TAB", tabId: 7 });
  check(
    "after CAPTURE_ERROR a second CHECK_TAB starts a new capture (no stuck 'Already running')",
    retry && retry.ok === true && env.starts().length === 2,
    JSON.stringify(retry)
  );
  const second = env.starts()[1];
  check("each check gets its own id", second.checkId !== start.checkId);

  // The capture ends; the analysis runs; the tab stays locked until the result.
  await env.dispatch({ type: "CAPTURE_STOPPED", tabId: 7, checkId: second.checkId });
  await env.dispatch({
    type: "CAPTURE_UPLOADED", tabId: 7, checkId: second.checkId, serverPath: "/tmp/gradio/x/truebeat-capture.webm",
    capture: { width: 1920, height: 1080 }, region: { x0: 150, y0: 120, x1: 1110, y1: 660 },
    rect: start.rect, view: start.view, target: start.target,
  });
  const analyze = env.out.find((m) => m.type === "ANALYZE");
  check(
    "the upload is handed to the offscreen document for analysis, with the region (the video did not move)",
    analyze && analyze.serverPath === "/tmp/gradio/x/truebeat-capture.webm" && analyze.checkId === second.checkId &&
      JSON.stringify(analyze.region) === JSON.stringify({ x0: 150, y0: 120, x1: 1110, y1: 660 }),
    JSON.stringify(analyze)
  );
  const during = await env.dispatch({ type: "CHECK_TAB", tabId: 7 });
  check(
    "the lock holds while the backend analyses (not released by the upload)",
    during && during.ok === false && /already running/.test(during.error),
    JSON.stringify(during)
  );
  const elsewhere = await env.dispatch({ type: "CHECK_TAB", tabId: 8 });
  check("another tab may start recording once this one has stopped recording", elsewhere && elsewhere.ok === true, JSON.stringify(elsewhere));
  const tab8 = env.starts()[2];
  await env.dispatch({ type: "ANALYSIS_RESULT", tabId: 7, checkId: start.checkId, result: { verdict: "real" } });
  check(
    "a late result from a replaced check is ignored",
    !env.overlays.some((o) => o.tabId === 7 && o.state === "result")
  );
  await env.dispatch({ type: "ANALYSIS_RESULT", tabId: 7, checkId: second.checkId, result: { verdict: "real" }, capture: { width: 1920, height: 1080, region: null } });
  check(
    "the check's own result reaches the panel",
    env.overlays.some((o) => o.tabId === 7 && o.state === "result" && o.result.verdict === "real" && o.capture.width === 1920)
  );
  await env.dispatch({ type: "CAPTURE_ERROR", tabId: 8, checkId: tab8.checkId, title: "x", message: "y" });
  const after = await env.dispatch({ type: "CHECK_TAB", tabId: 7 });
  check("ANALYSIS_RESULT releases the tab", after && after.ok === true, JSON.stringify(after));

  // A closed tab's lock is dropped.
  const closing = await env.dispatch({ type: "CHECK_TAB", tabId: 7 });
  check("(the tab is locked again)", closing && closing.ok === false);
  env.removed.forEach((fn) => fn(7));
  await turn();
  const reopened = await env.dispatch({ type: "CHECK_TAB", tabId: 7 });
  check("chrome.tabs.onRemoved clears the tab's state", reopened && reopened.ok === true, JSON.stringify(reopened));
});

section("5b. background.js — a refused start and a restarted worker");
await guarded(async () => {
  const refused = backgroundEnv({
    frames: [frameResult({ videos: [videoInfo()] })],
    answer: { ok: false, error: "The browser refused the tab stream (Error starting tab capture)." },
  });
  const r1 = await refused.dispatch({ type: "CHECK_TAB", tabId: 7 });
  check(
    "an OFFSCREEN_START answered {ok:false} clears the lock and says why",
    r1 && r1.ok === false && /refused the tab stream/.test(r1.error) &&
      refused.overlays.some((o) => o.state === "error" && /refused the tab stream/.test(o.message)),
    JSON.stringify(r1)
  );
  const r2 = await refused.dispatch({ type: "CHECK_TAB", tabId: 7 });
  check("the next click tries again", refused.starts().length === 2, JSON.stringify(r2));

  // The lock lives in chrome.storage.session: a worker idled out mid-analysis
  // and woken by the next click still knows the tab is busy.
  const first = backgroundEnv({ frames: [frameResult({ videos: [videoInfo()] })] });
  await first.dispatch({ type: "CHECK_TAB", tabId: 7 });
  const f0 = first.starts()[0];
  await first.dispatch({ type: "CAPTURE_STOPPED", tabId: 7, checkId: f0.checkId });
  await first.dispatch({
    type: "CAPTURE_UPLOADED", tabId: 7, checkId: f0.checkId, serverPath: "/tmp/gradio/x/c.webm",
    capture: { width: 1920, height: 1080 }, region: null, rect: null, view: f0.view, target: f0.target,
  });
  const woken = backgroundEnv({ chrome: first.chrome, clock: first.clock });
  const busy = await woken.dispatch({ type: "CHECK_TAB", tabId: 7 });
  check(
    "a worker restarted mid-analysis keeps the lock (chrome.storage.session)",
    busy && busy.ok === false && /already running/.test(busy.error),
    JSON.stringify(busy)
  );
  await first.clock.advance(5 * 60 * 1000);
  const still = await woken.dispatch({ type: "CHECK_TAB", tabId: 7 });
  check("an analysis may take minutes: still locked at 5", still && still.ok === false, JSON.stringify(still));
  await first.clock.advance(6 * 60 * 1000);
  const stale = await woken.dispatch({ type: "CHECK_TAB", tabId: 7 });
  check("a check stuck for over 10 minutes no longer blocks the tab", stale && stale.ok === true, JSON.stringify(stale));

  // A recording that never reported back blocks other tabs for 2 minutes, not 10.
  const stuck = backgroundEnv({ frames: [frameResult({ videos: [videoInfo()] })] });
  await stuck.dispatch({ type: "CHECK_TAB", tabId: 7 });
  const blocked = await stuck.dispatch({ type: "CHECK_TAB", tabId: 8 });
  await stuck.clock.advance(3 * 60 * 1000);
  const freed = await stuck.dispatch({ type: "CHECK_TAB", tabId: 8 });
  check(
    "a recording silent for over 2 minutes stops blocking other tabs",
    blocked && blocked.ok === false && freed && freed.ok === true,
    JSON.stringify({ blocked, freed })
  );

  // The box could not be mapped onto the recording: the panel says why.
  const unmapped = backgroundEnv({ frames: [frameResult({ videos: [videoInfo()] })] });
  await unmapped.dispatch({ type: "CHECK_TAB", tabId: 7 });
  const s0 = unmapped.starts()[0];
  await unmapped.dispatch({
    type: "CAPTURE_UPLOADED", tabId: 7, checkId: s0.checkId, serverPath: "/tmp/gradio/x/c.webm",
    capture: { width: 960, height: 540 }, region: null, rect: s0.rect, view: s0.view, target: s0.target,
  });
  const az = unmapped.out.find((m) => m.type === "ANALYZE");
  check(
    "a box that could not be mapped onto the recording is reported, and the whole tab analysed",
    az && az.region === null && /could not be mapped/.test(az.note),
    JSON.stringify(az)
  );

  // The page cannot be read again when the recording ends: say so plainly.
  let probes = 0;
  const unreadable = backgroundEnv({ frames: () => (++probes === 1 ? [frameResult({ videos: [videoInfo()] })] : []) });
  await unreadable.dispatch({ type: "CHECK_TAB", tabId: 7 });
  const u0 = unreadable.starts()[0];
  await unreadable.dispatch({
    type: "CAPTURE_UPLOADED", tabId: 7, checkId: u0.checkId, serverPath: "/tmp/gradio/x/c.webm",
    capture: { width: 1920, height: 1080 }, region: { x0: 150, y0: 120, x1: 1110, y1: 660 },
    rect: u0.rect, view: u0.view, target: u0.target,
  });
  const az2 = unreadable.out.find((m) => m.type === "ANALYZE");
  check(
    "a page that cannot be re-read after the recording gets the whole tab, said plainly (not 'moved')",
    az2 && az2.region === null && /could not be checked again/.test(az2.note),
    JSON.stringify(az2)
  );
});

section("5c. background.js — the panel is injected on click; the fallback works (C-07, C-08)");
await guarded(async () => {
  const env = backgroundEnv({ frames: [frameResult({ videos: [videoInfo()] })] });
  await env.dispatch({ type: "CHECK_TAB", tabId: 7 });
  const firstSend = env.tabCalls.findIndex((c) => c.kind === "send");
  const firstInject = env.tabCalls.findIndex((c) => c.kind === "inject");
  check(
    "overlay.js is injected (top frame) before the first state is pushed",
    firstInject > -1 && firstInject < firstSend && env.tabCalls[firstInject].files[0] === "overlay.js" &&
      env.tabCalls[firstInject].allFrames === false,
    JSON.stringify(env.tabCalls)
  );
  check(
    "the video is probed in every frame (allFrames)",
    env.tabCalls.some((c) => c.kind === "probe" && c.allFrames === true),
    JSON.stringify(env.tabCalls)
  );

  // tabs.sendMessage rejects once: inject with executeScript, then deliver.
  const flaky = backgroundEnv({ rejectSends: 1 });
  await flaky.dispatch({ type: "CAPTURE_ERROR", tabId: 9, title: "Backend error", message: "boom" });
  check(
    "a panel that is missing is injected with executeScript and the state delivered",
    flaky.tabCalls.map((c) => c.kind).join(",") === "send,inject,send" &&
      flaky.overlays.length === 1 && flaky.overlays[0].message === "boom",
    JSON.stringify(flaky.tabCalls)
  );
  check("the fallback never calls insertCSS({files: []})", flaky.insertCSS.length === 0);

  // A paused video: nothing is recorded, the panel says why.
  const paused = backgroundEnv({ frames: [frameResult({ videos: [videoInfo({ paused: true })] })] });
  const p = await paused.dispatch({ type: "CHECK_TAB", tabId: 7 });
  check(
    "a paused video starts no capture and the panel says why",
    p && p.ok === false && paused.starts().length === 0 &&
      paused.overlays.some((o) => o.state === "error" && /paused/.test(o.message)),
    JSON.stringify(p)
  );
  const again = await paused.dispatch({ type: "CHECK_TAB", tabId: 7 });
  check("(and the tab is not left locked)", again && again.ok === false && !/already running/.test(again.error || ""));

  // A page Chrome keeps extensions out of (chrome://, the Web Store): no
  // panel, so no recording and no upload whose result would go nowhere.
  const sealed = backgroundEnv({ refuseInject: true, frames: [frameResult({ videos: [videoInfo()] })] });
  const s = await sealed.dispatch({ type: "CHECK_TAB", tabId: 7 });
  check(
    "a page that cannot show the panel is not recorded, and the popup says why",
    s && s.ok === false && /cannot show a result here/.test(s.error) && sealed.starts().length === 0,
    JSON.stringify(s)
  );
  const s2 = await sealed.dispatch({ type: "CHECK_TAB", tabId: 7 });
  check("(nor left locked)", s2 && !/already running/.test(s2.error || ""), JSON.stringify(s2));

  // allFrames refused as a whole: the top frame alone still gives the size.
  const topOnly = backgroundEnv({ refuseAllFrames: true, frames: [frameResult({ videos: [videoInfo()] })] });
  await topOnly.dispatch({ type: "CHECK_TAB", tabId: 7 });
  const t0 = topOnly.starts()[0];
  check(
    "if the all-frames probe is refused, the top frame alone is probed",
    t0 && t0.view && t0.view.innerWidth === 1280 && t0.rect && t0.rect.x === 100 &&
      topOnly.tabCalls.filter((c) => c.kind === "probe").map((c) => c.allFrames).join(",") === "true,false",
    JSON.stringify({ t0, calls: topOnly.tabCalls })
  );
});

// ---------------------------------------------- 6. the whole bus, end to end

section("6. end to end: click -> probe -> capture -> upload -> re-probe -> analysis -> panel");
await guarded(async () => {
  const startFrames = [frameResult({ videos: [videoInfo({ duration: NaN })] })];
  const ok = streamOf([{ text: HEARTBEAT, delay: 15000 }, { text: completeEvent(VERDICT), delay: 3000 }, { end: true }]);

  // (a) The video stays put: the region goes to the backend.
  {
    const clock = makeClock();
    const backend = fakeBackend({ clock, stream: ok });
    const bus = busEnv({ clock, backend, frames: startFrames });
    const res = await bus.check(7);
    check("the click starts a check", res && res.ok === true, JSON.stringify(res));
    check(
      "the panel is injected before anything is pushed into it",
      bus.tabCalls[0] && bus.tabCalls[0].kind === "inject",
      JSON.stringify(bus.tabCalls.slice(0, 3))
    );
    await runUntil(clock, bus.finished);
    const final = bus.last();
    const gum = bus.off.gumCalls[0] && bus.off.gumCalls[0].video.mandatory;
    check(
      "the recording is requested at the tab's size in device pixels",
      gum && gum.minWidth === 1920 && gum.maxHeight === 1080 && gum.maxFrameRate === 30,
      JSON.stringify(gum)
    );
    check(
      "capture progress shows the size",
      bus.overlays.some((o) => o.state === "capturing" && o.size && o.size.width === 1920),
      JSON.stringify(bus.overlays.filter((o) => o.state === "capturing").slice(-1))
    );
    check(
      "the overlay was told the analysis is running before any verdict",
      bus.overlays.some((o) => o.state === "analyzing"),
      JSON.stringify(bus.overlays.map((o) => o.state))
    );
    const post = backend.posts()[0];
    check(
      "the analysis call used the uploaded server path and the video's region",
      post && post.url === "http://127.0.0.1:9/gradio_api/call/analyze_video_region" &&
        JSON.parse(post.body).data[0].path === "/tmp/gradio/abc123/truebeat-capture.webm" &&
        JSON.parse(post.body).data[1] === JSON.stringify({ x0: 150, y0: 120, x1: 1110, y1: 660 }),
      JSON.stringify(post && post.body)
    );
    check(
      "background relays the posted result into the overlay",
      final && final.state === "result" && final.result.verdict === "fake" &&
        JSON.stringify(final.capture) === JSON.stringify({ width: 1920, height: 1080, region: { x0: 150, y0: 120, x1: 1110, y1: 660 } }),
      JSON.stringify(final)
    );
    check(
      "no blob ever crossed the runtime",
      bus.deliveries.every((d) => !("blob" in d)) &&
        bus.sent("CAPTURE_UPLOADED").some((d) => typeof d.serverPath === "string") &&
        bus.sent("ANALYZE").length === 1 && bus.sent("ANALYSIS_RESULT").length === 1,
      JSON.stringify(bus.deliveries.map((d) => d.type))
    );
    check("the end-of-capture probe ran (two probes in all)", bus.probe.count === 2, String(bus.probe.count));
    const again = await bus.check(7);
    check("after the result the tab can be checked again", again && again.ok === true, JSON.stringify(again));
  }

  // (b) The video moved during the recording: the whole tab, and the panel says so.
  {
    const clock = makeClock();
    const backend = fakeBackend({ clock, stream: ok });
    const moved = [frameResult({ videos: [videoInfo({ duration: NaN, y: 400 })] })];
    const bus = busEnv({ clock, backend, frames: (n) => (n === 1 ? startFrames : moved) });
    await bus.check(7);
    await runUntil(clock, bus.finished);
    const post = backend.posts()[0];
    check(
      "a video that moved during the recording is analysed as the whole tab",
      post && post.url.endsWith("/gradio_api/call/analyze_video") && JSON.parse(post.body).data.length === 1,
      JSON.stringify(post && post.url)
    );
    check(
      "the panel says why, while analysing and with the result",
      bus.overlays.some((o) => o.state === "analyzing" && /moved, changed size or went away/.test(o.note || "")) &&
        bus.last().state === "result" && /moved, changed size or went away/.test(bus.last().note) && bus.last().capture.region === null,
      JSON.stringify(bus.last())
    );
  }

  // (c) A DRM stream: black frames stop the capture, nothing is uploaded, the tab is free again.
  {
    const clock = makeClock();
    const backend = fakeBackend({ clock, stream: ok });
    const bus = busEnv({ clock, backend, frames: startFrames });
    bus.off.frame = "black";
    await bus.check(7);
    await runUntil(clock, bus.finished);
    check(
      "a black capture ends in the DRM message with nothing uploaded",
      bus.last().state === "error" && /protected stream \(DRM\) or black frames/.test(bus.last().message) &&
        backend.uploads().length === 0,
      JSON.stringify(bus.last())
    );
    const again = await bus.check(7);
    check("and the tab can be checked again", again && again.ok === true, JSON.stringify(again));
  }
});

// --------------------------------------------- 7. a real HTTP backend (junk, drops)

section("7. a real HTTP backend: junk is not a verdict, a dropped connection is re-POSTed");
await guarded(async () => {
  let posts = 0;
  let mode = "junk";
  const server = createServer((req, res) => {
    if (req.url.endsWith("/gradio_api/info")) {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ named_endpoints: { "/analyze_video": {} } }));
      return;
    }
    if (req.method === "POST" && /\/gradio_api\/call\/analyze_video$/.test(req.url)) {
      posts += 1;
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ event_id: "ev-" + posts }));
      return;
    }
    if (req.url.includes("/gradio_api/call/analyze_video/")) {
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      if (mode === "junk") {
        res.end('event: complete\ndata: ["not json"]\n\n');
      } else if (req.url.endsWith("/ev-1") || req.url.endsWith("/ev-3")) {
        // the connection dies mid-stream
        res.write(HEARTBEAT);
        setTimeout(() => res.socket.destroy(), 20);
      } else {
        res.write(HEARTBEAT);
        setTimeout(() => res.end(completeEvent(VERDICT)), 20);
      }
      return;
    }
    res.writeHead(404);
    res.end();
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const base = `http://127.0.0.1:${server.address().port}`;
  const listeners = [];
  const sent = [];
  loadScript("offscreen.js", {
    chrome: chromeStub({
      runtime: {
        onMessage: { addListener: (fn) => listeners.push(fn) },
        sendMessage: async (m) => {
          sent.push(rtMessage(m));
          return {};
        },
        id: "test",
      },
    }),
    fetch,
    navigator: {},
    window: {},
    document: { createElement: () => ({}) },
    FormData,
    Blob,
    TextDecoder,
    console,
  });
  const analyse = async () => {
    const before = sent.length;
    listeners[0]({ type: "ANALYZE", tabId: 7, serverPath: "/tmp/gradio/fake/fake.webm", backend: base }, null, () => {});
    const startedAt = Date.now();
    for (;;) {
      const done = sent.slice(before).find((m) => m.type === "ANALYSIS_RESULT" || m.type === "CAPTURE_ERROR");
      if (done || Date.now() - startedAt > 10000) return done || null;
      await new Promise((r) => setTimeout(r, 25));
    }
  };
  const junk = await analyse();
  check(
    "an unreadable backend reply becomes the honest inconclusive verdict",
    junk && junk.type === "ANALYSIS_RESULT" && junk.result && junk.result.verdict === "inconclusive" &&
      junk.result.headline === "Unreadable backend reply",
    JSON.stringify(junk)
  );
  // ev-1 dies mid-stream; the re-POST (ev-2) completes.
  mode = "drop";
  posts = 0;
  const before = posts;
  const dropped = await analyse();
  check(
    "a connection dropped mid-stream is re-POSTed over a real socket and the verdict arrives",
    dropped && dropped.type === "ANALYSIS_RESULT" && dropped.result.verdict === "fake" && posts - before === 2,
    JSON.stringify({ dropped, posts: posts - before })
  );
  server.close();
});

// ------------------------------------------------ 8. static privacy audit

section("8. static audit — permissions, no remote code, no telemetry, one destination");
await guarded(async () => {
  const manifest = JSON.parse(readFileSync(join(EXT, "manifest.json"), "utf-8"));
  check("manifest v3", manifest.manifest_version === 3);
  check("version 1.3.0", manifest.version === "1.3.0", manifest.version);
  check(
    "no remote code in the manifest",
    !JSON.stringify(manifest).includes("http://cdn") &&
      !JSON.stringify(manifest).includes("cdn.jsdelivr") &&
      !JSON.stringify(manifest).includes("unpkg")
  );
  check(
    "no content script on every page: the panel is injected on click (C-08)",
    !("content_scripts" in manifest),
    JSON.stringify(manifest.content_scripts)
  );
  check("no web-accessible resources (a fingerprinting surface)", !("web_accessible_resources" in manifest));
  check(
    "permissions are exactly what the flow needs",
    JSON.stringify([...manifest.permissions].sort()) ===
      JSON.stringify(["activeTab", "offscreen", "scripting", "storage", "tabCapture"]),
    JSON.stringify(manifest.permissions)
  );
  check(
    "host permissions are local-only by default",
    manifest.host_permissions.every((h) => /^https?:\/\/(127\.0\.0\.1|localhost)\/\*$/.test(h)),
    JSON.stringify(manifest.host_permissions)
  );
  check(
    "other origins are optional (asked for at save time)",
    Array.isArray(manifest.optional_host_permissions) && manifest.optional_host_permissions.length > 0
  );
  check("the manifest description fits the store's 132 characters", manifest.description.length <= 132, String(manifest.description.length));

  const files = ["background.js", "offscreen.js", "popup.js", "options.js", "overlay.js"];
  const read = (f) => readFileSync(join(EXT, f), "utf-8");
  // Comments out of the way, so a URL or "fetch(" in prose counts for nothing.
  const code = (f) =>
    read(f)
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .split("\n")
      .filter((l) => !/^\s*\/\//.test(l))
      .join("\n");
  const telemetryHosts = [
    "google-analytics", "googletagmanager", "segment.io", "sentry.io",
    "amplitude", "mixpanel", "facebook.com/tr", "doubleclick",
  ];
  const hostHits = [];
  const remoteCodeHits = [];
  const inlineScriptHits = [];
  const primitiveHits = [];
  for (const f of files) {
    const src = read(f);
    for (const h of telemetryHosts) if (src.includes(h)) hostHits.push(`${f}:${h}`);
    for (const m of src.matchAll(/https?:\/\/[^\s"'`)]+/g)) {
      if (/cdn|jsdelivr|unpkg|googleapis|cloudflare/i.test(m[0])) remoteCodeHits.push(`${f}:${m[0]}`);
    }
    for (const m of code(f).matchAll(/\b(XMLHttpRequest|WebSocket|EventSource|sendBeacon|importScripts|RTCPeerConnection)\b|new\s+Image\s*\(/g)) {
      primitiveHits.push(`${f}:${m[0]}`);
    }
  }
  for (const f of ["popup.html", "options.html", "offscreen.html"]) {
    const html = read(f);
    // any <script> without a src= is inline -> blocked by the MV3 CSP
    for (const m of html.matchAll(/<script\b([^>]*)>/g)) if (!/src=/.test(m[1])) inlineScriptHits.push(f);
    for (const m of html.matchAll(/<script[^>]*src=["']([^"']+)["']/g)) {
      if (/^https?:/.test(m[1])) remoteCodeHits.push(`${f}:${m[1]}`);
    }
  }
  check("no telemetry/analytics hosts referenced", hostHits.length === 0, hostHits.join(", "));
  check("no remote scripts or CDN references", remoteCodeHits.length === 0, remoteCodeHits.join(", "));
  check("no inline <script> (MV3 CSP would block it)", inlineScriptHits.length === 0, inlineScriptHits.join(", "));
  check("no network primitive but fetch (no XHR, sockets, beacons, image pings)", primitiveHits.length === 0, primitiveHits.join(", "));

  // Every fetch must go to the configured backend: the URL expression is
  // exactly `base + "/gradio_api/…"` (optionally + identifiers), and every
  // `base` is bound to the backend setting. No substring passes.
  const firstArgs = (src, name) => {
    const out = [];
    const re = new RegExp("\\b" + name + "\\(", "g");
    let m;
    while ((m = re.exec(src))) {
      let i = m.index + m[0].length;
      const start = i;
      let depth = 0;
      let quote = null;
      for (; i < src.length; i++) {
        const c = src[i];
        if (quote) {
          if (c === "\\") i += 1;
          else if (c === quote) quote = null;
          continue;
        }
        if (c === '"' || c === "'" || c === "`") quote = c;
        else if ("([{".includes(c)) depth += 1;
        else if (")]}".includes(c)) {
          if (depth === 0) break;
          depth -= 1;
        } else if (c === "," && depth === 0) break;
      }
      out.push(src.slice(start, i).replace(/\s+/g, " ").trim());
    }
    return out;
  };
  const backendUrl = /^base \+ "\/gradio_api\/[a-z_/]*"(?: \+ [A-Za-z_$][\w$]*(?: \+ "[^"]*")?)*$/;
  const fetches = {};
  for (const f of files) fetches[f] = firstArgs(code(f), "fetch");
  check(
    "background.js and offscreen.js both fetch (the audit cannot pass vacuously)",
    fetches["background.js"].length >= 1 && fetches["offscreen.js"].length >= 4,
    JSON.stringify(fetches)
  );
  const strays = Object.entries(fetches).flatMap(([f, list]) => list.filter((u) => !backendUrl.test(u)).map((u) => `${f}: fetch(${u}`));
  check("every fetch in every script targets base + \"/gradio_api/…\"", strays.length === 0, strays.join(" | "));
  check(
    "only the service worker and the offscreen document talk to the network",
    fetches["popup.js"].length === 0 && fetches["options.js"].length === 0 && fetches["overlay.js"].length === 0
  );
  const bindings = [];
  for (const f of ["background.js", "offscreen.js"]) {
    for (const m of code(f).matchAll(/\b(?:const|let|var)\s+base\s*=\s*([^;]+);|(?<![.\w])base\s*=(?!=)\s*([^;]+);/g)) {
      bindings.push(`${f}: ${(m[1] || m[2]).trim()}`);
    }
  }
  const badBindings = bindings.filter((b) => !/: (normaliseBackend\(\w+\)|cap\.backend|msg\.backend)$/.test(b));
  check(
    "`base` is only ever the backend setting (normaliseBackend / the message's backend)",
    bindings.length >= 3 && badBindings.length === 0,
    JSON.stringify(bindings)
  );
  check("the offscreen document holds no URL of its own", !/https?:\/\//.test(code("offscreen.js")));

  // The 1.0.2 / 1.0.3 guards: the upload and the SSE wait live in the offscreen document.
  const bgSource = read("background.js");
  const offSource = read("offscreen.js");
  check("background owns no upload: no FormData in background.js", !bgSource.includes("FormData"));
  check("the offscreen document performs the upload (gradio_api/upload)", offSource.includes("gradio_api/upload"));
  check("no SSE reader is left behind in background.js (no getReader)", !bgSource.includes("getReader"));
  check("no call endpoints left in background.js (the offscreen doc owns them)", !bgSource.includes("/gradio_api/call"));
  check("the offscreen document owns the ANALYZE handler", offSource.includes('msg.type === "ANALYZE"'));
  check(
    "the offscreen document re-POSTs a dropped wait (bounded) instead of re-reading the event",
    /const ANALYZE_ATTEMPTS = \d+;/.test(offSource) && /const SSE_IDLE_MS = /.test(offSource) && !offSource.includes("SSE_RETRIES"),
  );
  check("no insertCSS left in the overlay fallback (C-07)", !code("background.js").includes("insertCSS"));
  check("the sampler no longer draws the track (C-02)", !/drawImage\(\s*track/.test(offSource));

  // The overlay reads only keys the backend sends (tests/js/api_keys.json).
  const used = [...new Set([...code("overlay.js").matchAll(/\br\.([a-z_]+)/g)].map((m) => m[1]))];
  const unknown = used.filter((k) => !API_KEYS.includes(k));
  check(
    "every result key the overlay reads is in the backend contract (api_keys.json)",
    used.length >= 15 && unknown.length === 0,
    `unknown: ${unknown.join(", ")}; used: ${used.join(", ")}`
  );
  check(
    "the overlay renders pulse_measurable and warnings (C-09)",
    used.includes("pulse_measurable") && used.includes("warnings")
  );

  // Texts that 1.3.0 made false.
  const readme = readFileSync(join(EXT, "README.md"), "utf-8");
  const popupHtml = read("popup.html");
  const offscreenHtml = read("offscreen.html");
  check("README no longer claims Gradio replays a completed event (C-05)", !/replays? a completed event/i.test(readme));
  check("README no longer claims a 1280 px width cap (C-06)", !/max width 1280|up to 1280 px/i.test(readme));
  check("README documents the region endpoint and its fallback", readme.includes("/analyze_video_region") && readme.includes("/analyze_video"));
  check("README's verdict table uses the app's headlines", !readme.includes("or manipulated") && readme.includes("Possible face-swap"));
  check(
    "README lists the JSON fields the backend sends",
    API_KEYS.every((k) => readme.includes("`" + k + "`")),
    API_KEYS.filter((k) => !readme.includes("`" + k + "`")).join(", ")
  );
  check("the popup no longer describes the 1.0 face-swap-only tool (C-16)", !/Detects face-swaps\./.test(popupHtml));
  check("offscreen.html no longer says it never touches the network (C-16)", !/never\s+touches the network/i.test(offscreenHtml));
});

// ----------------------------------------------------------------- summary

console.log(`\n${passed} passed, ${failed} failed`);
if (failed) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
}
