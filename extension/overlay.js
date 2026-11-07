/**
 * TrueBeat extension — in-page result panel.
 *
 * Injected into the tab's top frame when you press "Check this tab" (no
 * content script is declared). A shadow-DOM panel in the bottom-right
 * corner; it moves into the fullscreen element when the page goes
 * fullscreen, so it stays visible there (once the recording has ended, so
 * it is never recorded over the video). It renders whatever the service
 * worker sends it:
 *   capturing | analyzing | result | error
 *
 * It never computes a verdict: the headline, detail, numbers, warnings and
 * caveat come from the backend. The labels below are used only when a
 * reply lacks its own headline or caveat, and they are the app's wording.
 */

(() => {
  if (window.__trueBeatOverlay) return;
  window.__trueBeatOverlay = true;

  const VERDICT_STYLE = {
    real: {
      accent: "#22c55e",
      label: "No sign of AI generation found",
    },
    fake: {
      accent: "#ef4444",
      label: "Likely AI-generated",
    },
    inconclusive: {
      accent: "#f59e0b",
      label: "Inconclusive",
    },
  };

  // The app's headline for each layer that can decide "fake"
  // (app.verdict_for), for a reply that carries no headline of its own.
  const FAKE_HEADLINE = {
    provenance: "AI-generated: declared by its Content Credentials",
    ai_video: "Likely AI-generated",
    face_swap: "Possible face-swap",
    pulse: "Possible face-swap",
  };

  // The backend's own caveat (app.CAVEAT) as of 1.3.0, shown only when a
  // reply carries none; every TrueBeat backend sends its current one.
  const STANDING_CAVEAT =
    "Checks for fully AI-generated video with a whole-frame detector trained " +
    "on 21 generator families. On held-out test sets it caught about 4 in 5 clips from " +
    "2024-era tools (Sora, Kling, Runway, Luma, Pika), about half from the " +
    "newest (Veo 3.1, Sora 2, Kling 2.6) and from some open-source ones, and " +
    "wrongly flagged about 5% of real test videos (reports/aigen_detector_v2.md). " +
    "Clips under 2 s are not checked. It also reads C2PA Content Credentials " +
    "when present. Face-swaps: a weak pulse check can decide (it catches about a " +
    "quarter of Deepfakes-style swaps and is blind to reenactment fakes such as " +
    "Face2Face and NeuralTextures); a face-swap detector trained on FaceForensics++ " +
    "is shown as evidence only, because it wrongly flagged 38% of real faces from " +
    "other sources (docs/12). 'No sign of AI generation' is not proof a video is real.";

  function fallbackHeadline(r) {
    if (r && r.verdict === "fake") return FAKE_HEADLINE[r.decided_by] || VERDICT_STYLE.fake.label;
    return (VERDICT_STYLE[r && r.verdict] || VERDICT_STYLE.inconclusive).label;
  }

  // Exposed for the automated tests (tests/js/extension_test.mjs).
  globalThis.__trueBeatVerdictStyles = VERDICT_STYLE;
  globalThis.__trueBeatFakeHeadlines = FAKE_HEADLINE;
  globalThis.__trueBeatFallbackHeadline = fallbackHeadline;
  globalThis.__trueBeatStandingCaveat = STANDING_CAVEAT;

  // Elements that do not draw their children: when one of these is the
  // fullscreen element, the panel stays in the page.
  const NO_CHILDREN = new Set(["VIDEO", "AUDIO", "IFRAME", "CANVAS", "IMG", "EMBED", "OBJECT", "SVG"]);

  let host = null;
  let root = null;
  let panel = null;
  let shownState = "";

  /**
   * Where the panel lives: the fullscreen element when the page has one
   * (else it is not painted there), the page otherwise. While the tab is
   * being recorded it stays out of a fullscreen player: painted over the
   * video, it would be recorded into the frames the backend analyses.
   */
  function mountPoint() {
    const fs = document.fullscreenElement;
    if (fs && shownState !== "capturing" && !NO_CHILDREN.has(String(fs.tagName || "").toUpperCase())) {
      return fs;
    }
    return document.body || document.documentElement;
  }

  /** Keep the panel inside whatever the page shows: the fullscreen element, or the page. */
  function place() {
    if (!host) return;
    const parent = mountPoint();
    if (parent && host.parentNode !== parent) parent.appendChild(host);
  }

  if (typeof document.addEventListener === "function") {
    document.addEventListener("fullscreenchange", place);
  }

  function ensurePanel() {
    if (host && host.isConnected) {
      place();
      return;
    }
    host = document.createElement("div");
    host.id = "truebeat-overlay-host";
    host.style.cssText = "position:fixed;z-index:2147483647;right:16px;bottom:16px;";
    root = host.attachShadow({ mode: "closed" });
    const style = document.createElement("style");
    style.textContent = `
      .panel {
        font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #e6e6e6; background: #14161a; border: 1px solid #2a2e36;
        border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,.45);
        width: 320px; padding: 14px 16px; box-sizing: border-box;
        max-height: calc(100vh - 32px); overflow-y: auto;
      }
      .head { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
      .dot { width:9px; height:9px; border-radius:50%; flex:0 0 auto; }
      .title { font-weight:600; font-size:14px; }
      .close { margin-left:auto; background:none; border:0; color:#8b93a1;
               font-size:17px; line-height:1; cursor:pointer; padding:2px 4px; }
      .close:hover { color:#e6e6e6; }
      .msg { color:#b9c0cc; }
      .bar { height:4px; background:#242832; border-radius:2px; margin:10px 0 4px; overflow:hidden; }
      .bar > i { display:block; height:100%; background:#f97316; width:0%; transition:width .25s; }
      .timer { font-variant-numeric: tabular-nums; color:#8b93a1; font-size:12px; }
      table { width:100%; border-collapse:collapse; margin:10px 0 0; }
      td { padding:2px 0; vertical-align:top; }
      td.k { color:#8b93a1; padding-right:10px; white-space:nowrap; }
      td.v { text-align:right; font-variant-numeric: tabular-nums; }
      .warn { margin:8px 0 0; padding:0 0 0 14px; color:#8b93a1; font-size:11px; line-height:1.4; }
      .note { color:#9aa3b2; font-size:11px; margin-top:8px; }
      .caveat { margin-top:10px; padding-top:8px; border-top:1px solid #262a33;
                color:#8b93a1; font-size:11px; line-height:1.4; }
      .detail { color:#9aa3b2; font-size:12px; margin-top:8px; }
      .sr { position:absolute; width:1px; height:1px; margin:-1px; padding:0; border:0;
            overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }
    `;
    panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `
      <div class="head">
        <span class="dot" style="background:#f97316"></span>
        <span class="title">TrueBeat</span>
        <button class="close" title="Close" aria-label="Close">&times;</button>
      </div>
      <div class="body"></div>
      <div class="sr" role="status" aria-live="polite"></div>
    `;
    panel.querySelector(".close").addEventListener("click", () => {
      host.remove();
      host = null;
    });
    root.append(style, panel);
    place();
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[c]));
  }

  function num(v, digits = 1) {
    if (v == null || !isFinite(v)) return "n/a";
    return Number(v).toFixed(digits);
  }

  /** Screen readers hear the verdict (and errors) once, politely. */
  function announce(text) {
    const live = panel && panel.querySelector(".sr");
    if (live) live.textContent = text;
  }

  function sizeText(size) {
    const w = Math.round(Number(size && size.width));
    const h = Math.round(Number(size && size.height));
    return w > 0 && h > 0 ? `${w}×${h} px` : "";
  }

  /** What was recorded and which part of it was analysed. */
  function captureText(capture) {
    const size = sizeText(capture);
    if (!size) return "";
    const box = capture.region;
    const w = box ? Math.round(Number(box.x1) - Number(box.x0)) : 0;
    const h = box ? Math.round(Number(box.y1) - Number(box.y0)) : 0;
    return w > 0 && h > 0
      ? `${size} tab; analysed the video's ${w}×${h} px area`
      : `${size}; analysed the whole tab`;
  }

  function warningsOf(r) {
    const list = Array.isArray(r.warnings)
      ? r.warnings
      : r.qc && Array.isArray(r.qc.warnings)
        ? r.qc.warnings
        : [];
    return list.filter((w) => w != null && String(w).trim()).map(String);
  }

  function render(payload) {
    shownState = String(payload.state || "");
    ensurePanel();
    const body = panel.querySelector(".body");
    const dot = panel.querySelector(".dot");
    const title = panel.querySelector(".title");

    if (payload.state === "capturing") {
      dot.style.background = "#f97316";
      title.textContent = "TrueBeat";
      const elapsed = Math.round(Number(payload.elapsed) || 0);
      const total = Math.round(Number(payload.total) || 0);
      const pct = total ? Math.min(100, (elapsed / total) * 100) : 0;
      const size = sizeText(payload.size);
      body.innerHTML = `
        <div class="msg">Capturing… ${elapsed}s${total ? " / " + total + "s" : ""}</div>
        <div class="bar"><i style="width:${pct}%"></i></div>
        <div class="timer">${esc(payload.message || "Recording the tab…")}</div>
        ${size ? `<div class="timer">${esc("Recording the tab at " + size + ".")}</div>` : ""}`;
      return;
    }

    if (payload.state === "analyzing") {
      dot.style.background = "#f97316";
      title.textContent = "TrueBeat";
      body.innerHTML = `
        <div class="msg">${esc(payload.message || "Analysing…")}</div>
        <div class="bar"><i style="width:100%"></i></div>
        ${payload.note ? `<div class="note">${esc(payload.note)}</div>` : ""}`;
      return;
    }

    if (payload.state === "error") {
      dot.style.background = "#f59e0b";
      title.textContent = payload.title || "TrueBeat";
      body.innerHTML = `
        <div class="msg">${esc(payload.message || "")}</div>`;
      announce(["TrueBeat", payload.title, payload.message].filter(Boolean).join(": "));
      return;
    }

    if (payload.state === "result") {
      const r = payload.result && typeof payload.result === "object" ? payload.result : {};
      const style = VERDICT_STYLE[r.verdict] || VERDICT_STYLE.inconclusive;
      const headline =
        typeof r.headline === "string" && r.headline.trim() ? r.headline : fallbackHeadline(r);
      dot.style.background = style.accent;
      title.textContent = "TrueBeat";
      const rows = [];
      if (r.ai_score != null) {
        rows.push([
          "AI-video detector",
          "score " + num(r.ai_score, 2) + " (flags at " + num(r.ai_threshold, 2) + ")" +
            (r.ai_real_percentile != null
              ? "; above " + Math.round(r.ai_real_percentile * 100) + "% of real clips"
              : ""),
        ]);
      }
      if (r.fsd_score != null) {
        rows.push([
          "face-swap detector",
          "score " + num(r.fsd_score, 2) + " (flags at " + num(r.fsd_threshold, 2) + ")" +
            (r.fsd_mode === "decisive"
              ? ""
              : ", evidence only" +
                (r.fsd_outside_fpr != null
                  ? " (flagged " + Math.round(r.fsd_outside_fpr * 100) + "% of real faces outside its training data)"
                  : "")) +
            (r.fsd_n_sampled ? "; face in " + r.fsd_n_faces + " of " + r.fsd_n_sampled + " frames" : ""),
        ]);
      }
      if (r.decided_by) {
        rows.push(["decided by", String(r.decided_by).replace("_", " ")]);
      }
      if (r.hr_bpm != null) {
        rows.push([
          "heart rate",
          num(r.hr_bpm) + " bpm" +
            (r.pulse_measurable === false ? " (not reliable: pulse not measurable)" : ""),
        ]);
      }
      if (r.region_agreement_bpm != null) {
        rows.push([
          "region agreement",
          num(r.region_agreement_bpm) + " bpm spread",
        ]);
      }
      if (r.snr != null) {
        rows.push(["pulse SNR", num(r.snr) + " dB"]);
      }
      if (r.face_valid_frac != null) {
        rows.push(["face visible", Math.round(r.face_valid_frac * 100) + "%"]);
      }
      if (r.score != null) {
        rows.push(["pulse face-swap score", num(r.score, 3)]);
      }
      const recorded = captureText(payload.capture);
      if (recorded) rows.push(["recorded", recorded]);
      const warnings = warningsOf(r);
      const caveat = typeof r.caveat === "string" && r.caveat.trim() ? r.caveat : STANDING_CAVEAT;
      body.innerHTML = `
        <div class="msg" style="color:${style.accent};font-weight:600">
          ${esc(headline)}
        </div>
        ${r.detail ? `<div class="detail">${esc(r.detail)}</div>` : ""}
        ${
          rows.length
            ? `<table>${rows
                .map(([k, v]) => `<tr><td class="k">${esc(k)}</td><td class="v">${esc(v)}</td></tr>`)
                .join("")}</table>`
            : ""
        }
        ${
          warnings.length
            ? `<ul class="warn">${warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>`
            : ""
        }
        ${payload.note ? `<div class="note">${esc(payload.note)}</div>` : ""}
        <div class="caveat">${esc(caveat)}</div>`;
      announce("TrueBeat: " + headline);
      return;
    }
  }

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg && msg.type === "TRUEBEAT_STATE") {
      render(msg.payload || {});
      if (typeof sendResponse === "function") sendResponse({ ok: true });
    }
  });
})();
