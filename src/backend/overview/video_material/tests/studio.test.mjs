import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createCanvas, GlobalFonts } from "@napi-rs/canvas";
import { Renderer } from "../app/renderer.js";
import {
  DEFAULTS,
  SHOTS,
  DURATION,
  flatten,
  makeCaseModel,
  buildHierarchy,
  breadthSelect,
  baselineSelect,
  coverage,
  methodFor,
  weights,
  keyOf,
} from "../app/model.js";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repo = path.resolve(root, "../../../..");
const snapshot = JSON.parse(
  readFileSync(path.join(root, "data/snapshot.json"), "utf8"),
);
for (const style of ["Regular", "Medium", "SemiBold", "Bold"])
  GlobalFonts.registerFromPath(
    path.join(root, `assets/fonts/Manrope-${style}.ttf`),
    "Manrope",
  );
GlobalFonts.registerFromPath(
  path.join(root, "assets/fonts/IBMPlexMono-Regular.ttf"),
  "IBM Plex Mono",
);

test("all five source fingerprints still match the repository", () => {
  for (const info of Object.values(snapshot.provenance))
    assert.equal(
      createHash("sha256")
        .update(readFileSync(path.join(repo, info.source)))
        .digest("hex"),
      info.sha256,
      info.source,
    );
});
test("five distinct stories retain exact quoted spans and valid ontology leaves", () => {
  const criterion = JSON.parse(
    readFileSync(path.join(repo, snapshot.provenance.criterion.source), "utf8"),
  );
  const criterionKeys = new Set(flatten(criterion).map(keyOf)),
    predictorKeys = new Set(flatten(snapshot.predictor).map(keyOf));
  assert.equal(snapshot.cases.length, 5);
  assert.equal(new Set(snapshot.cases.map((c) => c.complaint)).size, 5);
  for (const c of snapshot.cases) {
    assert.equal(c.criteria.length, 6);
    assert.equal(c.candidates.length, 8);
    for (const cr of c.criteria) {
      assert.ok(c.complaint.includes(cr.span));
      assert.ok(criterionKeys.has(keyOf(cr.path)));
    }
    for (const p of c.candidates) {
      assert.ok(predictorKeys.has(keyOf(p.path)));
      assert.ok(p.links.every((i) => i >= 0 && i < c.criteria.length));
    }
  }
});
test("full catalogue preserves every leaf; sampled nodes preserve real parent-child edges", () => {
  const m = makeCaseModel(snapshot, "lana", 36),
    all = buildHierarchy(m, 8000),
    sample = buildHierarchy(m, 720);
  assert.equal(all.displayedLeaves, 7265);
  assert.equal(all.nodes.filter((n) => !n.children.length).length, 7265);
  assert.ok(sample.displayedLeaves < all.displayedLeaves);
  for (const n of sample.nodes) {
    if (n.depth) assert.equal(keyOf(n.path.slice(0, -1)), n.parent.id);
    assert.ok(Number.isFinite(n.x) && Number.isFinite(n.y));
  }
  const psycho = buildHierarchy(m, 720, "PSYCHO");
  assert.ok(
    psycho.nodes.filter((n) => n.depth).every((n) => n.path[0] === "PSYCHO"),
  );
});
test("BFS honours its domain coverage, round-robin and score-refinement allocation", () => {
  const rows = [];
  for (let domain = 0; domain < 4; domain++)
    for (let j = 0; j < 7; j++)
      rows.push({
        path: ["PSYCHO", `d${domain}`, `leaf${j}`],
        score: 1 - domain * 0.1 - j * 0.01,
      });
  const selected = breadthSelect(rows, 16);
  assert.equal(selected.length, 16);
  assert.equal(new Set(selected.map((r) => keyOf(r.path))).size, 16);
  assert.equal(new Set(selected.slice(0, 4).map((r) => r.path[1])).size, 4);
  assert.ok(selected.slice(0, 4).every((r) => r.phase === 0));
  assert.ok(selected.slice(4, 12).every((r) => r.phase === 1));
  assert.ok(selected.slice(12).every((r) => r.phase === 2));
  assert.equal(breadthSelect(rows, 2).length, 2);
  assert.equal(breadthSelect(rows, 200).length, rows.length);
});
test("comparison counts come from the selected candidates and change with policy", () => {
  for (const c of snapshot.cases) {
    const m = makeCaseModel(snapshot, c.id, 36),
      narrow = baselineSelect(m, "psychodynamic", 36),
      wide = baselineSelect(m, "integrative", 36);
    assert.ok(narrow.every((r) => r.path[1] === "Insight_Oriented_Therapies"));
    assert.ok(coverage(wide).domains > coverage(narrow).domains);
    assert.equal(
      coverage(m.selected).criteria,
      new Set(m.selected.flatMap((r) => r.links)).size,
    );
    assert.ok(m.selected.length <= 36);
  }
});
test("sparse examples do not receive invented personal impact anchors", () => {
  for (const id of ["maya", "elias"])
    assert.ok(
      makeCaseModel(snapshot, id, 36).rows.every((r) => r.anchor === 0),
    );
  assert.equal(methodFor(0).tier, 0);
  assert.equal(methodFor(28).code, "DESCRIPTIVES");
  assert.equal(methodFor(42).directed, false);
  assert.equal(methodFor(61).directed, false);
  assert.equal(methodFor(84, "TIME_VARYING_gVAR").code, "TIME_VARYING_gVAR");
});
test("readiness weights remain complementary and within source bounds", () => {
  for (const r of [-5, 0, 28, 42, 61, 78, 100, 150]) {
    const w = weights(r);
    assert.ok(w.idiographic >= 0.3 && w.idiographic <= 0.8);
    assert.ok(Math.abs(w.idiographic + w.nomothetic - 1) < 1e-9);
  }
  assert.equal(weights(78).idiographic, 0.69);
});
test("cycle simulation changes personal anchors without forcing improvement", () => {
  const first = makeCaseModel(snapshot, "lana", 36, 1),
    next = makeCaseModel(snapshot, "lana", 36, 2);
  const differences = next.rows.map((r, i) => r.anchor - first.rows[i].anchor);
  assert.ok(differences.some((x) => x > 0));
  assert.ok(differences.some((x) => x < 0));
});
test("the timeline includes the full closed loop and has no gaps", () => {
  for (let i = 1; i < SHOTS.length; i++)
    assert.equal(SHOTS[i].start, SHOTS[i - 1].start + SHOTS[i - 1].duration);
  assert.equal(DURATION, 174);
  assert.ok(
    [
      "intake",
      "criteria",
      "ontology",
      "lineage",
      "model",
      "ema",
      "readiness",
      "network",
      "search",
      "compare",
      "targets",
      "fusion",
      "hapa",
      "loop",
    ].every((id) => SHOTS.some((s) => s.id === id)),
  );
});
test("all five cases render every scene, boundaries and alternate readiness states without off-screen text", () => {
  const canvas = createCanvas(960, 540),
    renderer = new Renderer(canvas, snapshot),
    overflow = [];
  const original = renderer.text.bind(renderer);
  renderer.text = (
    str,
    x,
    y,
    size = 22,
    color,
    weight = 400,
    align = "left",
    mono = false,
  ) => {
    renderer.font(size, weight, mono);
    const w = renderer.ctx.measureText(String(str)).width;
    let left = align === "right" ? x - w : align === "center" ? x - w / 2 : x;
    let right = left + w;
    if (left < 5 || right > 1915 || y > 1078)
      overflow.push(
        `${renderer.caseData?.id}/${renderer.shot?.id}: ${String(str)} [${left.toFixed(0)},${right.toFixed(0)},${y}]`,
      );
    original(str, x, y, size, color, weight, align, mono);
  };
  for (const c of snapshot.cases)
    for (const s of SHOTS)
      for (const fraction of [0.05, 0.58, 0.95])
        renderer.draw({
          ...DEFAULTS,
          caseId: c.id,
          time: s.start + s.duration * fraction,
        });
  for (const r of [0, 42, 61, 95])
    for (const id of ["readiness", "network", "fusion"]) {
      const s = SHOTS.find((s) => s.id === id);
      renderer.draw({ ...DEFAULTS, readiness: r, time: s.start + 5 });
    }
  assert.deepEqual([...new Set(overflow)], []);
});
test("scrubbing is deterministic and independent of previous scenes", () => {
  const canvas = createCanvas(480, 270),
    r = new Renderer(canvas, snapshot),
    state = { ...DEFAULTS, time: 102.2 };
  r.draw(state);
  const first = createHash("sha256").update(canvas.data()).digest("hex");
  r.draw({ ...state, caseId: "noor", time: 46 });
  r.draw(state);
  assert.equal(createHash("sha256").update(canvas.data()).digest("hex"), first);
});
