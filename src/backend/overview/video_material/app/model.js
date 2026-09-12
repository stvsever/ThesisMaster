export const COLORS = {
  bg: "#0b101a",
  panel: "#121b28",
  line: "#263246",
  ink: "#f4f3f0",
  muted: "#9daec5",
  faint: "#61748e",
  BIO: "#87cfff",
  PSYCHO: "#bba5ff",
  SOCIAL: "#8ddfc1",
  gold: "#edc48d",
  rose: "#f1a7b5",
};
export const SHOTS = [
  ["opening", "The person", "Every story has a different structure.", 7],
  [
    "intake",
    "Listen & separate",
    "The complaint leads. The person and context stay visible.",
    13,
  ],
  [
    "criteria",
    "Ground the meaning",
    "Atomic concepts, traceable spans, ontology-constrained leaves.",
    11,
  ],
  [
    "ontology",
    "Reveal the hierarchy",
    "Explore a real, deeply nested biopsychosocial solution space.",
    12,
  ],
  [
    "lineage",
    "Follow one possibility",
    "A candidate is the endpoint of a traceable hierarchy, not a detached suggestion.",
    10,
  ],
  [
    "model",
    "Build a first model",
    "HyDE retrieval proposes predictor-criterion links for review.",
    11,
  ],
  [
    "ema",
    "Observe daily life",
    "The initial model determines what to measure over time.",
    10,
  ],
  [
    "readiness",
    "Respect the evidence",
    "Data quality determines which analytical method is supportable.",
    10,
  ],
  [
    "network",
    "Estimate & prioritise",
    "Quantify associations and predictive impact within this person.",
    11,
  ],
  [
    "search",
    "Breadth before depth",
    "Cover domains, rotate through them, then refine by score.",
    14,
  ],
  [
    "compare",
    "Compare the search",
    "Hold the input and candidate space constant. Change the policy.",
    12,
  ],
  [
    "targets",
    "Select with scrutiny",
    "Ranked candidates become reviewed, feasible treatment targets.",
    10,
  ],
  [
    "fusion",
    "Update the model",
    "Fuse population knowledge with readiness-weighted personal evidence.",
    10,
  ],
  [
    "hapa",
    "Make action possible",
    "Turn a target into a phase-aware plan with a coping alternative.",
    14,
  ],
  [
    "loop",
    "Remember & adapt",
    "Communicate the plan, retain its lineage, and begin the next cycle.",
    12,
  ],
  [
    "closing",
    "Keep learning",
    "A broader search. A personal model. An adaptive next step.",
    7,
  ],
].map(([id, title, subtitle, duration], index, list) => ({
  id,
  title,
  subtitle,
  duration,
  index,
  start: list.slice(0, index).reduce((s, x) => s + x[3], 0),
}));
export const DURATION = SHOTS.reduce((s, x) => s + x.duration, 0);
export const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => {
  t = clamp(t);
  return t * t * (3 - 2 * t);
};
export const range = (t, a, b) => ease((t - a) / (b - a));
export const clean = (s) => s.replaceAll("_", " ").replaceAll("\u2014", ", ");
export function hash(s) {
  let h = 2166136261;
  for (const c of s) h = Math.imul(h ^ c.charCodeAt(0), 16777619);
  return (h >>> 0) / 4294967296;
}
export const keyOf = (p) => p.join(" / ");
export function flatten(tree, path = [], out = []) {
  for (const [label, child] of Object.entries(tree)) {
    const p = [...path, label];
    if (Object.keys(child).length) flatten(child, p, out);
    else out.push(p);
  }
  return out;
}
export function shotAt(time) {
  return SHOTS.find((s) => time < s.start + s.duration) || SHOTS.at(-1);
}
export function formatTime(s) {
  s = Math.max(0, Math.floor(s));
  return `${Math.floor(s / 60)
    .toString()
    .padStart(2, "0")}:${(s % 60).toString().padStart(2, "0")}`;
}
export function weights(readiness) {
  const idiographic = clamp(0.3 + (0.5 * readiness) / 100, 0.3, 0.8);
  return { idiographic, nomothetic: 1 - idiographic };
}
export function methodFor(readiness, preferred = "STATIC_gVAR") {
  if (readiness < 35)
    return {
      tier: 0,
      name: "Descriptive monitoring",
      code: "DESCRIPTIVES",
      note: "Summaries and trajectories only. No estimated network.",
      directed: false,
    };
  if (readiness < 55)
    return {
      tier: 1,
      name: "Exploratory correlations",
      code: "CORRELATION",
      note: "A low-complexity association view. No temporal direction.",
      directed: false,
    };
  if (readiness < 72)
    return {
      tier: 2,
      name: "Reduced baseline analysis",
      code: "GGM",
      note: "Illustrative partial correlations; no lagged inference.",
      directed: false,
    };
  return {
    tier: 3,
    name:
      preferred === "TIME_VARYING_gVAR"
        ? "Time-varying gVAR"
        : "Stationary gVAR",
    code: preferred === "TIME_VARYING_gVAR" ? preferred : "STATIC_gVAR",
    note: "Lagged dynamics, conditional on full method diagnostics.",
    directed: true,
  };
}
export function breadthSelect(rows, budget = 18) {
  const domains = new Map();
  for (const row of rows) {
    const d = keyOf(row.path.slice(0, 2));
    if (!domains.has(d)) domains.set(d, []);
    domains.get(d).push(row);
  }
  for (const group of domains.values())
    group.sort(
      (a, b) => b.score - a.score || keyOf(a.path).localeCompare(keyOf(b.path)),
    );
  const ordered = [...domains.values()].sort((a, b) => b[0].score - a[0].score);
  const out = [],
    seen = new Set();
  const add = (r, phase) => {
    if (r && out.length < budget && !seen.has(keyOf(r.path))) {
      out.push({ ...r, phase });
      seen.add(keyOf(r.path));
    }
  };
  for (const group of ordered) add(group[0], 0);
  const breadthLimit = Math.min(budget, domains.size * 3);
  for (let pos = 1; out.length < breadthLimit && pos < rows.length; pos++) {
    let any = false;
    for (const group of ordered)
      if (group[pos]) {
        add(group[pos], 1);
        any = true;
        if (out.length >= breadthLimit) break;
      }
    if (!any) break;
  }
  for (const row of [...rows].sort((a, b) => b.score - a.score)) add(row, 2);
  return out;
}
export function makeCaseModel(
  snapshot,
  caseId,
  budget = 24,
  cycle = 1,
  readiness = null,
) {
  const caseData =
    snapshot.cases.find((c) => c.id === caseId) || snapshot.cases[0];
  const allLeaves = flatten(snapshot.predictor);
  const seeds = caseData.candidates;
  const domains = new Set(seeds.map((s) => keyOf(s.path.slice(0, 2))));
  const pool = new Map();
  for (const seed of seeds) pool.set(keyOf(seed.path), seed);
  for (const domain of domains) {
    const options = allLeaves
      .filter((p) => keyOf(p.slice(0, 2)) === domain)
      .sort((a, b) => hash(keyOf(a)) - hash(keyOf(b)));
    for (const path of options.slice(0, 9))
      if (!pool.has(keyOf(path)))
        pool.set(keyOf(path), {
          path,
          label: clean(path.at(-1)),
          links: [],
          impact: 0.1 + hash(keyOf(path)) * 0.4,
        });
  }
  const rows = [...pool.values()].map((s, i) => {
    const seeded = seeds.includes(s),
      h = hash(caseId + keyOf(s.path));
    const mapping = seeded ? 0.74 + h * 0.22 : 0.24 + h * 0.39;
    const hyde = seeded ? 0.69 + h * 0.23 : 0.16 + h * 0.48;
    const effectiveReadiness = readiness ?? caseData.readiness;
    const anchor =
      effectiveReadiness < 55
        ? 0
        : clamp(s.impact + Math.sin(h * 11 + cycle) * (cycle - 1) * 0.07);
    const bonus = seeded ? 1 : 0.75;
    const score = 0.45 * mapping + 0.25 * hyde + 0.2 * anchor + 0.1 * bonus;
    return {
      ...s,
      index: i,
      seeded,
      mapping,
      hyde,
      anchor,
      bonus,
      score,
      root: s.path[0],
    };
  });
  const selected = breadthSelect(rows, budget);
  return {
    caseData,
    allLeaves,
    rows,
    selected,
    seeds: rows.filter((r) => r.seeded),
    maxDepth: snapshot.maxPredictorDepth,
    snapshot,
  };
}
export function baselineSelect(model, policy, budget = 24) {
  const rows = [...model.rows].sort((a, b) => b.score - a.score);
  if (policy === "integrative") return rows.slice(0, budget);
  const domain =
    policy === "cbt"
      ? "Cognitive_Behavioral_Therapies"
      : "Insight_Oriented_Therapies";
  return rows
    .filter((r) => r.path[0] === "PSYCHO" && r.path[1] === domain)
    .slice(0, budget);
}
export function coverage(rows) {
  return {
    domains: new Set(rows.map((r) => keyOf(r.path.slice(0, 2)))).size,
    roots: new Set(rows.map((r) => r.root)).size,
    criteria: new Set(rows.flatMap((r) => r.links)).size,
  };
}
export function buildHierarchy(model, density = 720, branch = "ALL") {
  const source = model.allLeaves.filter(
    (p) => branch === "ALL" || p[0] === branch,
  );
  const keep = new Map(
    model.rows
      .filter((r) => branch === "ALL" || r.root === branch)
      .map((r) => [keyOf(r.path), r.path]),
  );
  const groups = new Map();
  for (const p of source) {
    const d = keyOf(p.slice(0, 2));
    if (!groups.has(d)) groups.set(d, []);
    groups.get(d).push(p);
  }
  if (density >= source.length) for (const p of source) keep.set(keyOf(p), p);
  const perGroup = Math.max(1, Math.floor(density / groups.size));
  for (const paths of groups.values()) {
    const sorted = [...paths].sort((a, b) => hash(keyOf(a)) - hash(keyOf(b)));
    for (const p of sorted.slice(0, perGroup)) keep.set(keyOf(p), p);
  }
  const nodes = new Map();
  const root = {
    id: "",
    label: "PREDICTOR",
    depth: 0,
    path: [],
    children: [],
    x: 0,
    y: 0,
    root: "PSYCHO",
  };
  nodes.set("", root);
  for (const path of keep.values())
    for (let depth = 1; depth <= path.length; depth++) {
      const id = keyOf(path.slice(0, depth));
      if (nodes.has(id)) continue;
      const parent = nodes.get(keyOf(path.slice(0, depth - 1)));
      const n = {
        id,
        label: path[depth - 1],
        path: path.slice(0, depth),
        depth,
        root: path[0],
        children: [],
        parent,
      };
      nodes.set(id, n);
      parent.children.push(n);
    }
  function count(n) {
    n.children.sort((a, b) => a.label.localeCompare(b.label));
    n.leafCount = n.children.length
      ? n.children.reduce((s, c) => s + count(c), 0)
      : 1;
    return n.leafCount;
  }
  count(root);
  const branches = ["BIO", "PSYCHO", "SOCIAL"].filter((b) =>
    root.children.some((c) => c.label === b),
  );
  function layout(n, lo, hi) {
    n.angle = (lo + hi) / 2;
    const r = 55 + n.depth * 45;
    n.x = Math.cos(n.angle) * r;
    n.y = Math.sin(n.angle) * r;
    let pos = lo;
    for (const child of n.children) {
      const span = ((hi - lo) * child.leafCount) / n.leafCount;
      layout(child, pos, pos + span);
      pos += span;
    }
  }
  branches.forEach((b, i) => {
    const lo = -Math.PI / 2 + (i * Math.PI * 2) / branches.length + 0.045;
    layout(
      root.children.find((c) => c.label === b),
      lo,
      lo + (Math.PI * 2) / branches.length - 0.09,
    );
  });
  return {
    nodes: [...nodes.values()],
    lookup: nodes,
    root,
    displayedLeaves: keep.size,
    totalLeaves: source.length,
  };
}
export function simulatedSeries(caseData, signal = 0, n = 64) {
  const phase = hash(caseData.id) * 8 + signal * 1.7;
  return Array.from({ length: n }, (_, i) =>
    clamp(
      0.5 +
        0.19 * Math.sin(i * 0.32 + phase) +
        0.1 * Math.sin(i * 0.89 + phase * 2) +
        0.08 * Math.cos(i * 0.15 + signal),
      0.08,
      0.92,
    ),
  );
}
export const DEFAULTS = {
  caseId: "lana",
  time: 3,
  playing: false,
  speed: 1,
  density: 720,
  depth: 10,
  branch: "ALL",
  zoom: 1,
  panX: 0,
  panY: 0,
  autoCamera: true,
  labels: true,
  particles: true,
  captions: true,
  compare: "psychodynamic",
  budget: 36,
  readiness: null,
  cycle: 1,
  reducedMotion: false,
};
