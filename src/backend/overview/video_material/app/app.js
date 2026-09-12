import { Renderer } from "./renderer.js";
import {
  DEFAULTS,
  SHOTS,
  DURATION,
  clamp,
  clean,
  shotAt,
  formatTime,
  keyOf,
  methodFor,
  weights,
} from "./model.js";

const $ = (id) => document.getElementById(id);
const state = {
  ...DEFAULTS,
  reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
};
let renderer,
  snapshot,
  last = performance.now(),
  toastTimer,
  dirty = true,
  lastChapter = -1;
const change = () => {
  dirty = true;
};
function toast(text) {
  $("toast").textContent = text;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ($("toast").hidden = true), 3200);
}
function option(value, text) {
  const el = document.createElement("option");
  el.value = value;
  el.textContent = text;
  return el;
}
function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function go(time, pause = false) {
  state.time = clamp(time, 0, DURATION - 0.001);
  if (pause) state.playing = false;
  sync();
  change();
}
function sync() {
  for (const key of [
    "caseId",
    "speed",
    "cycle",
    "zoom",
    "branch",
    "density",
    "depth",
    "compare",
    "budget",
  ]) {
    const el = $(key === "caseId" ? "case" : key);
    el.value = state[key];
  }
  for (const key of [
    "autoCamera",
    "labels",
    "particles",
    "captions",
    "reducedMotion",
  ])
    $(key).checked = state[key];
  const d = snapshot.cases.find((c) => c.id === state.caseId);
  const r = state.readiness ?? d.readiness;
  $("readiness").value = r;
  $("readiness-value").textContent = `${r} / 100`;
  $("zoom-value").textContent = `${Math.round(state.zoom * 100)}%`;
  $("depth-value").textContent = `${state.depth} levels`;
  $("budget-value").textContent = `${state.budget} slots`;
  $("case-note").textContent = `${d.age} · ${d.role}. ${d.theme}.`;
  $("case-transcript").textContent = d.complaint;
  $("plan-transcript").textContent =
    `Authored proposal: ${d.hapa.when}. ${d.hapa.action} Coping alternative: ${d.hapa.coping}`;
  $("download-mp4").href = `renders/mp4/${d.id}_phoenix_full_workflow.mp4`;
  $("play-icon").textContent = state.playing ? "Ⅱ" : "▶";
  $("play").setAttribute(
    "aria-label",
    state.playing ? "Pause film" : "Play film",
  );
}
function updateChapter() {
  const shot = shotAt(state.time);
  $("scene-title").textContent = shot.title;
  $("time-value").textContent = formatTime(state.time);
  $("timeline").value = state.time;
  $("timeline").style.background =
    `linear-gradient(to right, #c3b1e8 ${(state.time / DURATION) * 100}%, #364154 ${(state.time / DURATION) * 100}%)`;
  if (lastChapter !== shot.index) {
    lastChapter = shot.index;
    $("chapter").value = shot.index;
    $("chapter-count").textContent =
      `${String(shot.index + 1).padStart(2, "0")} / ${SHOTS.length}`;
    $("chapter-heading").textContent = shot.title;
    $("scene-description").textContent = shot.subtitle;
    document
      .querySelectorAll("#stage-chips button")
      .forEach((b, i) =>
        b.classList.toggle("active", SHOTS[i].index === shot.index),
      );
    $("stage").setAttribute(
      "aria-label",
      `${snapshot.cases.find((c) => c.id === state.caseId).name}: ${shot.title}. ${shot.subtitle}`,
    );
  }
}
function inspect(row, type = "CANDIDATE") {
  const path = row.path || [];
  $("inspect-title").textContent = clean(row.label || path.at(-1));
  $("inspect-path").textContent = path.map(clean).join(" › ");
  $("node-type").textContent = type;
  $("inspect-note").textContent =
    row.score !== undefined
      ? `Illustrative score ${row.score.toFixed(3)}. ${row.links.length ? `Linked concepts: ${row.links.map((i) => renderer.caseData.criteria[i].label).join(", ")}.` : "No authored criterion links for this background candidate."}`
      : "Exact parent-child lineage from the repository snapshot. Membership does not establish clinical suitability.";
}
function resetInspect() {
  renderer.prepare(state);
  $("inspect-candidate").replaceChildren(
    option("", "Choose a featured candidate…"),
  );
  renderer.model.seeds.forEach((r, i) =>
    $("inspect-candidate").append(option(i, `${r.root} · ${r.label}`)),
  );
  inspect(renderer.model.seeds[0]);
  $("inspect-candidate").value = "0";
}
function loadHash() {
  try {
    const p = new URLSearchParams(location.hash.slice(1));
    if (snapshot.cases.some((c) => c.id === p.get("case")))
      state.caseId = p.get("case");
    const number = (key, min, max, target = key) => {
      if (p.has(key) && Number.isFinite(Number(p.get(key))))
        state[target] = clamp(Number(p.get(key)), min, max);
    };
    number("t", 0, DURATION - 0.001, "time");
    number("zoom", 0.6, 2.4);
    number("depth", 2, 10);
    number("budget", 12, 60);
    number("readiness", 0, 100);
    number("cycle", 1, 3);
    number("panX", -650, 650);
    number("panY", -450, 450);
    number("speed", 0.5, 2);
    if (["ALL", "BIO", "PSYCHO", "SOCIAL"].includes(p.get("branch")))
      state.branch = p.get("branch");
    if (["psychodynamic", "cbt", "integrative"].includes(p.get("compare")))
      state.compare = p.get("compare");
    if (["320", "720", "1440", "8000"].includes(p.get("density")))
      state.density = Number(p.get("density"));
    for (const k of [
      "autoCamera",
      "labels",
      "particles",
      "captions",
      "reducedMotion",
    ])
      if (p.has(k)) state[k] = p.get(k) === "1";
  } catch {
    toast("The shared view could not be read. The default view is ready.");
  }
}
function shareHash() {
  const p = new URLSearchParams({
    case: state.caseId,
    t: state.time.toFixed(2),
    zoom: state.zoom,
    depth: state.depth,
    branch: state.branch,
    density: state.density,
    budget: state.budget,
    compare: state.compare,
    readiness: state.readiness ?? renderer.caseData.readiness,
    cycle: state.cycle,
    panX: Math.round(state.panX),
    panY: Math.round(state.panY),
    speed: state.speed,
  });
  for (const k of [
    "autoCamera",
    "labels",
    "particles",
    "captions",
    "reducedMotion",
  ])
    p.set(k, state[k] ? "1" : "0");
  return `${location.origin}${location.pathname}#${p}`;
}
function setPresentation(active) {
  document.body.classList.toggle("presenting", active);
  window.scrollTo(0, 0);
  change();
}
function bind() {
  $("case").addEventListener("change", () => {
    state.caseId = $("case").value;
    state.readiness = null;
    state.time = 0;
    state.panX = state.panY = 0;
    lastChapter = -1;
    sync();
    resetInspect();
    change();
  });
  $("chapter").addEventListener("change", () =>
    go(SHOTS[Number($("chapter").value)].start + 0.65, true),
  );
  for (const key of [
    "speed",
    "cycle",
    "density",
    "depth",
    "zoom",
    "budget",
    "readiness",
  ])
    $(key).addEventListener("input", () => {
      state[key] = Number($(key).value);
      sync();
      change();
      if (key === "budget" || key === "cycle" || key === "readiness")
        resetInspect();
    });
  for (const key of ["branch", "compare"])
    $(key).addEventListener("change", () => {
      state[key] = $(key).value;
      sync();
      change();
    });
  for (const key of [
    "autoCamera",
    "labels",
    "particles",
    "captions",
    "reducedMotion",
  ])
    $(key).addEventListener("change", () => {
      state[key] = $(key).checked;
      change();
    });
  $("timeline").max = DURATION - 0.001;
  $("timeline").addEventListener("input", () =>
    go(Number($("timeline").value), true),
  );
  const play = () => {
    if (state.time >= DURATION - 0.1) state.time = 0;
    state.playing = !state.playing;
    sync();
    change();
  };
  $("play").addEventListener("click", play);
  const step = (delta) => {
    const shot = shotAt(state.time);
    go(
      SHOTS[clamp(shot.index + delta, 0, SHOTS.length - 1)].start + 0.65,
      true,
    );
  };
  $("previous").addEventListener("click", () => step(-1));
  $("next").addEventListener("click", () => step(1));
  $("restart").addEventListener("click", () => go(0));
  $("film").addEventListener("click", () => {
    state.autoCamera = true;
    state.playing = true;
    go(0);
    $("film").classList.add("selected");
    $("explore").classList.remove("selected");
  });
  $("explore").addEventListener("click", () => {
    state.autoCamera = false;
    go(SHOTS.find((s) => s.id === "ontology").start + 6, true);
    $("explore").classList.add("selected");
    $("film").classList.remove("selected");
  });
  $("reset-camera").addEventListener("click", () => {
    state.zoom = 1;
    state.panX = state.panY = 0;
    sync();
    change();
  });
  $("reset").addEventListener("click", () => {
    Object.assign(state, DEFAULTS, {
      reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)")
        .matches,
    });
    $("loop").checked = false;
    history.replaceState(null, "", location.pathname);
    lastChapter = -1;
    sync();
    resetInspect();
    change();
    toast("Studio controls reset.");
  });
  $("fullscreen").addEventListener("click", () => setPresentation(true));
  $("exit-present").addEventListener("click", () => setPresentation(false));
  $("toggle-controls").addEventListener("click", () => {
    document.body.classList.toggle("controls-hidden");
    $("toggle-controls").setAttribute(
      "aria-expanded",
      String(!document.body.classList.contains("controls-hidden")),
    );
  });
  $("inspect-candidate").addEventListener("change", () => {
    if ($("inspect-candidate").value !== "")
      inspect(renderer.model.seeds[Number($("inspect-candidate").value)]);
  });
  $("save-frame").addEventListener("click", () => {
    renderer.draw(state);
    $("stage").toBlob((blob) => {
      if (blob) {
        download(
          blob,
          `${state.caseId}_${shotAt(state.time).id}_${formatTime(state.time).replace(":", "-")}.png`,
        );
        toast("The 1920 × 1080 frame is ready.");
      } else toast("Frame export could not be completed.");
    }, "image/png");
  });
  $("save-session").addEventListener("click", () => {
    const payload = {
      version: 2,
      notice: snapshot.notice,
      state: { ...state, playing: false },
      case: renderer.caseData,
      selectedCandidates: renderer.model.selected.map(
        ({ path, score, phase, links }) => ({ path, score, phase, links }),
      ),
      sourceProvenance: snapshot.provenance,
    };
    download(
      new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      }),
      `phoenix_${state.caseId}_session.json`,
    );
    toast("Session and path provenance exported.");
  });
  $("copy-link").addEventListener("click", async () => {
    const url = shareHash();
    history.replaceState(null, "", url);
    try {
      await navigator.clipboard.writeText(url);
      toast("A link to this view is copied.");
    } catch {
      toast("The address bar now contains this view. Copy it to share.");
    }
  });
  $("download-mp4").addEventListener("click", () =>
    toast(
      "Downloading the default case film. Custom controls affect PNG and session exports.",
    ),
  );
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      setPresentation(false);
      return;
    }
    if (/INPUT|SELECT|TEXTAREA/.test(event.target.tagName)) return;
    if (event.code === "Space" && event.target.tagName === "BUTTON") return;
    if (event.code === "Space") {
      event.preventDefault();
      play();
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      step(1);
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      step(-1);
    }
    if (event.key.toLowerCase() === "f")
      setPresentation(!document.body.classList.contains("presenting"));
    if (event.key === "Escape") setPresentation(false);
  });
  let drag = null,
    moved = false;
  const point = (event) => {
    const r = $("stage").getBoundingClientRect();
    return {
      x: ((event.clientX - r.left) * 1920) / r.width,
      y: ((event.clientY - r.top) * 1080) / r.height,
    };
  };
  const pick = (p) =>
    renderer.hitRegions
      .filter((r) => Math.hypot(p.x - r.x, p.y - r.y) < r.r)
      .sort(
        (a, b) =>
          Math.hypot(p.x - a.x, p.y - a.y) - Math.hypot(p.x - b.x, p.y - b.y),
      )[0];
  $("stage").addEventListener("pointerdown", (e) => {
    const p = point(e);
    drag = { ...p, panX: state.panX, panY: state.panY };
    moved = false;
    $("stage").setPointerCapture(e.pointerId);
  });
  $("stage").addEventListener("pointermove", (e) => {
    const p = point(e);
    if (drag) {
      if (Math.hypot(p.x - drag.x, p.y - drag.y) > 5) moved = true;
      if (moved) {
        state.panX = clamp(drag.panX + p.x - drag.x, -650, 650);
        state.panY = clamp(drag.panY + p.y - drag.y, -450, 450);
        state.autoCamera = false;
        sync();
        change();
      }
      $("tooltip").hidden = true;
      return;
    }
    const hit = pick(p);
    $("tooltip").hidden = !hit;
    if (hit) {
      $("tooltip").textContent = clean(hit.node.label);
      const rect = $("canvas-wrap").getBoundingClientRect();
      $("tooltip").style.left =
        `${Math.min(e.clientX - rect.left + 12, rect.width - 285)}px`;
      $("tooltip").style.top = `${Math.max(8, e.clientY - rect.top - 44)}px`;
    }
  });
  $("stage").addEventListener("pointerup", (e) => {
    if (!moved) {
      const hit = pick(point(e));
      if (hit) inspect(hit.node, "ONTOLOGY NODE");
    }
    drag = null;
  });
  $("stage").addEventListener("pointercancel", () => (drag = null));
  $("stage").addEventListener(
    "pointerleave",
    () => ($("tooltip").hidden = true),
  );
  $("stage").addEventListener(
    "wheel",
    (e) => {
      if (!["ontology", "search", "lineage"].includes(shotAt(state.time).id))
        return;
      e.preventDefault();
      state.zoom = clamp(state.zoom - e.deltaY * 0.001, 0.6, 2.4);
      state.autoCamera = false;
      sync();
      change();
    },
    { passive: false },
  );
  window.addEventListener("hashchange", () => {
    loadHash();
    lastChapter = -1;
    sync();
    resetInspect();
    change();
  });
}
function frame(now) {
  const elapsed = Math.min((now - last) / 1000, 0.2);
  last = now;
  if (state.playing) {
    state.time += elapsed * state.speed;
    if (state.time >= DURATION) {
      if ($("loop").checked) state.time %= DURATION;
      else {
        state.time = DURATION - 0.001;
        state.playing = false;
        sync();
      }
    }
    dirty = true;
  }
  if (dirty) {
    renderer.draw(state);
    updateChapter();
    $("sample-count").textContent =
      `${renderer.hierarchy.displayedLeaves.toLocaleString()} shown / ${renderer.hierarchy.totalLeaves.toLocaleString()} leaves`;
    dirty = false;
  }
  requestAnimationFrame(frame);
}
async function init() {
  try {
    const response = await fetch("data/snapshot.json");
    if (!response.ok)
      throw new Error("The ontology snapshot could not be loaded.");
    snapshot = await response.json();
    await document.fonts.load("500 32px Manrope");
    await document.fonts.load('14px "IBM Plex Mono"');
    await document.fonts.ready;
    snapshot.cases.forEach((c) =>
      $("case").append(option(c.id, `${c.name} · ${c.theme}`)),
    );
    SHOTS.forEach((s, i) => {
      $("chapter").append(
        option(i, `${String(i + 1).padStart(2, "0")} · ${s.title}`),
      );
      const tick = document.createElement("i");
      tick.style.left = `${(s.start / DURATION) * 100}%`;
      $("chapter-ticks").append(tick);
      const chip = document.createElement("button");
      chip.textContent = `${i + 1}`;
      chip.title = s.title;
      chip.setAttribute("aria-label", `Go to ${s.title}`);
      chip.addEventListener("click", () => go(s.start + 0.65, true));
      $("stage-chips").append(chip);
    });
    $("time-value").nextElementSibling.textContent =
      `/ ${formatTime(DURATION)}`;
    loadHash();
    renderer = new Renderer($("stage"), snapshot);
    bind();
    sync();
    resetInspect();
    $("loading").remove();
    requestAnimationFrame(frame);
  } catch (error) {
    $("loading").textContent =
      `${error.message} Start the local server with npm start, then open its local address.`;
    console.error(error);
  }
}
init();
