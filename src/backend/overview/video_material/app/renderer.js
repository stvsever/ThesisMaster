import {
  COLORS as C,
  SHOTS,
  DURATION,
  clamp,
  lerp,
  ease,
  range,
  clean,
  hash,
  keyOf,
  shotAt,
  formatTime,
  weights,
  methodFor,
  makeCaseModel,
  buildHierarchy,
  baselineSelect,
  coverage,
  simulatedSeries,
} from "./model.js";

const TAU = Math.PI * 2;
export class Renderer {
  constructor(canvas, snapshot) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.snapshot = snapshot;
    this.hitRegions = [];
    this.cacheKey = "";
  }
  font(size = 22, weight = 400, mono = false) {
    this.ctx.font = `${weight} ${size}px "${mono ? "IBM Plex Mono" : "Manrope"}", sans-serif`;
  }
  text(
    str,
    x,
    y,
    size = 22,
    color = C.ink,
    weight = 400,
    align = "left",
    mono = false,
  ) {
    const c = this.ctx;
    this.font(size, weight, mono);
    c.fillStyle = color;
    c.textAlign = align;
    c.textBaseline = "alphabetic";
    c.fillText(String(str), x, y);
  }
  wrap(
    str,
    x,
    y,
    width,
    size = 22,
    color = C.muted,
    lineHeight = 1.5,
    weight = 400,
    maxLines = 20,
  ) {
    this.font(size, weight);
    const words = String(str).split(/\s+/);
    let line = "",
      count = 0;
    for (const word of words) {
      const next = line ? `${line} ${word}` : word;
      if (this.ctx.measureText(next).width > width && line) {
        this.text(line, x, y + count * size * lineHeight, size, color, weight);
        count++;
        line = word;
        if (count >= maxLines) return count * size * lineHeight;
      } else line = next;
    }
    if (line) {
      this.text(line, x, y + count * size * lineHeight, size, color, weight);
      count++;
    }
    return count * size * lineHeight;
  }
  fit(str, width, size = 20) {
    this.font(size);
    if (this.ctx.measureText(str).width <= width) return str;
    while (str.length && this.ctx.measureText(str + "…").width > width)
      str = str.slice(0, -1);
    return str + "…";
  }
  rect(x, y, w, h, fill = C.panel, stroke = null, r = 18) {
    const c = this.ctx;
    c.beginPath();
    c.roundRect(x, y, w, h, r);
    if (fill) {
      c.fillStyle = fill;
      c.fill();
    }
    if (stroke) {
      c.strokeStyle = stroke;
      c.lineWidth = 1;
      c.stroke();
    }
  }
  line(x, y, x2, y2, color = C.line, width = 1) {
    const c = this.ctx;
    c.beginPath();
    c.moveTo(x, y);
    c.lineTo(x2, y2);
    c.strokeStyle = color;
    c.lineWidth = width;
    c.stroke();
  }
  circle(x, y, r, color, stroke = null, width = 1) {
    const c = this.ctx;
    c.beginPath();
    c.arc(x, y, r, 0, TAU);
    if (color) {
      c.fillStyle = color;
      c.fill();
    }
    if (stroke) {
      c.strokeStyle = stroke;
      c.lineWidth = width;
      c.stroke();
    }
  }
  pill(label, x, y, color = C.PSYCHO, size = 14) {
    this.font(size, 500, true);
    const w = this.ctx.measureText(label).width + 28;
    this.rect(x, y - 22, w, 34, color + "16", color + "45", 17);
    this.text(label, x + 14, y + 1, size, color, 500, "left", true);
    return w;
  }
  bezier(a, b, color, width = 1, progress = 1, bend = 0) {
    const c = this.ctx;
    const dx = b.x - a.x;
    c.beginPath();
    c.moveTo(a.x, a.y);
    c.bezierCurveTo(
      a.x + dx * 0.48,
      a.y + bend,
      b.x - dx * 0.48,
      b.y + bend,
      b.x,
      b.y,
    );
    c.strokeStyle = color;
    c.lineWidth = width;
    c.stroke();
  }
  arrow(a, b, color, width = 1.5) {
    this.line(a.x, a.y, b.x, b.y, color, width);
    const angle = Math.atan2(b.y - a.y, b.x - a.x);
    this.line(
      b.x,
      b.y,
      b.x - 10 * Math.cos(angle - 0.45),
      b.y - 10 * Math.sin(angle - 0.45),
      color,
      width,
    );
    this.line(
      b.x,
      b.y,
      b.x - 10 * Math.cos(angle + 0.45),
      b.y - 10 * Math.sin(angle + 0.45),
      color,
      width,
    );
  }
  saveAlpha(alpha, fn) {
    this.ctx.save();
    this.ctx.globalAlpha *= clamp(alpha);
    fn();
    this.ctx.restore();
  }
  badge(label, x, y, index, color = C.PSYCHO) {
    this.circle(x, y, 22, color + "17", color + "55");
    this.text(index, x, y + 6, 15, color, 600, "center", true);
    this.text(label, x + 38, y + 7, 21, C.ink, 500);
  }
  panelHeading(label, x, y, detail = "") {
    this.text(label.toUpperCase(), x, y, 15, C.muted, 500, "left", true);
    if (detail) this.text(detail, x, y + 30, 19, C.ink, 500);
  }
  logo(x, y, scale = 1, color = C.ink) {
    const c = this.ctx;
    c.save();
    c.translate(x, y);
    c.scale(scale, scale);
    c.strokeStyle = color;
    c.lineWidth = 2.6;
    c.lineCap = "round";
    c.beginPath();
    c.moveTo(-15, 13);
    c.bezierCurveTo(-33, -5, -14, -17, -10, -30);
    c.bezierCurveTo(-13, -8, 10, -8, 1, 13);
    c.moveTo(-3, 14);
    c.bezierCurveTo(19, 3, 24, -13, 17, -26);
    c.bezierCurveTo(39, -3, 13, 24, -3, 23);
    c.stroke();
    c.restore();
  }
  prepare(state) {
    const modelKey = `${state.caseId}:${state.budget}:${state.cycle}:${state.readiness}`;
    if (this.modelKey !== modelKey) {
      this.model = makeCaseModel(
        this.snapshot,
        state.caseId,
        state.budget,
        state.cycle,
        state.readiness,
      );
      this.modelKey = modelKey;
      this.cacheKey = "";
    }
    const k = `${modelKey}:${state.density}:${state.branch}`;
    if (k !== this.cacheKey) {
      this.hierarchy = buildHierarchy(this.model, state.density, state.branch);
      this.cacheKey = k;
    }
    this.caseData = this.model.caseData;
  }
  draw(state) {
    this.prepare(state);
    this.state = state;
    this.time = state.time;
    this.shot = shotAt(state.time);
    this.local = state.time - this.shot.start;
    this.p = clamp(this.local / this.shot.duration);
    this.hitRegions = [];
    const c = this.ctx;
    c.setTransform(
      this.canvas.width / 1920,
      0,
      0,
      this.canvas.height / 1080,
      0,
      0,
    );
    c.globalAlpha = 1;
    c.fillStyle = C.bg;
    c.fillRect(0, 0, 1920, 1080);
    const gradient = c.createRadialGradient(1340, 430, 20, 1340, 430, 1000);
    gradient.addColorStop(0, this.caseData.color + "0e");
    gradient.addColorStop(1, C.bg + "00");
    c.fillStyle = gradient;
    c.fillRect(0, 0, 1920, 1080);
    c.save();
    c.globalAlpha = 0.28;
    for (let x = 30; x < 1920; x += 40)
      for (let y = 30; y < 1080; y += 40) this.circle(x, y, 0.6, C.faint);
    c.restore();
    this.logo(107, 59, 0.65);
    this.text("PHOENIX", 141, 67, 23, C.ink, 700);
    this.text(
      "THE ADAPTIVE CARE LOOP",
      308,
      65,
      12,
      C.faint,
      400,
      "left",
      true,
    );
    this.text(
      `${this.caseData.name.toUpperCase()}  /  ${this.caseData.theme.toUpperCase()}`,
      1830,
      52,
      13,
      C.muted,
      400,
      "right",
      true,
    );
    this.text(
      "FICTIONAL CASE · SIMULATED EVIDENCE",
      1830,
      77,
      11,
      C.faint,
      400,
      "right",
      true,
    );
    this.line(88, 102, 1832, 102, C.line);
    if (!["opening", "closing"].includes(this.shot.id)) {
      this.text(
        String(this.shot.index).padStart(2, "0"),
        88,
        166,
        15,
        this.caseData.color,
        400,
        "left",
        true,
      );
      this.text(this.shot.title, 141, 175, 48, C.ink, 600);
      this.text(this.shot.subtitle, 88, 218, 22, C.muted);
      this.text(
        `${String(this.shot.index).padStart(2, "0")} / ${SHOTS.length - 2}`,
        1832,
        170,
        15,
        C.faint,
        400,
        "right",
        true,
      );
    }
    // Deterministic transition: headings stay fixed while the scene eases into place.
    const entrance =
        state.reducedMotion || this.shot.id === "opening"
          ? 1
          : range(this.local, 0, 0.6),
      outro =
        state.reducedMotion || this.shot.id === "closing"
          ? 1
          : 1 - range(this.local, this.shot.duration - 0.4, this.shot.duration);
    c.save();
    c.globalAlpha *= entrance * outro;
    c.translate(0, state.reducedMotion ? 0 : (1 - entrance) * 18);
    this[this.shot.id === "model" ? "observationModel" : this.shot.id]();
    c.restore();
    this.footer();
    return {
      shot: this.shot,
      caseData: this.caseData,
      regions: this.hitRegions,
    };
  }
  footer() {
    if (this.state.captions) {
      this.rect(88, 956, 1744, 66, "#131d2b", null, 13);
      this.circle(115, 989, 4, this.caseData.color);
      this.text(this.caption || this.shot.subtitle, 137, 996, 20, C.ink, 400);
    }
    this.caption = "";
    SHOTS.forEach((s, i) => {
      const w = (1730 * s.duration) / DURATION;
      const x = 88 + (1730 * s.start) / DURATION;
      this.rect(
        x,
        1047,
        w - 4,
        3,
        i < this.shot.index
          ? this.caseData.color
          : i === this.shot.index
            ? C.ink
            : C.line,
        null,
        1,
      );
      if (i === this.shot.index)
        this.rect(x, 1047, (w - 4) * this.p, 3, this.caseData.color, null, 1);
    });
    this.text(
      "STIJN VAN SEVEREN  ·  GHENT UNIVERSITY",
      88,
      1039,
      10,
      C.faint,
      400,
      "left",
      true,
    );
    this.text(
      `${formatTime(this.time)} / ${formatTime(DURATION)}`,
      1832,
      1039,
      11,
      C.faint,
      400,
      "right",
      true,
    );
  }
  graph(
    x,
    y,
    scale = 1,
    {
      active = [],
      reveal = 1,
      labels = true,
      rotate = 0,
      allowCamera = false,
      fade = 0.8,
    } = {},
  ) {
    const c = this.ctx,
      visibleDepth = this.state.depth,
      h = this.hierarchy;
    const activeIds = new Set();
    for (const row of active)
      for (let d = 1; d <= row.path.length; d++)
        activeIds.add(keyOf(row.path.slice(0, d)));
    const angle = rotate;
    const cs = Math.cos(angle),
      sn = Math.sin(angle);
    let cameraScale = 1,
      cameraX = 0,
      cameraY = 0;
    if (allowCamera) {
      const auto = this.state.autoCamera && !this.state.reducedMotion;
      if (auto && this.shot.id === "ontology") {
        const z = Math.sin(this.p * Math.PI) ** 2;
        cameraScale = 1 + z * 0.3;
        cameraX = -z * 70;
        cameraY = z * 15;
      }
      if (auto && this.shot.id === "search") {
        const z = range(this.p, 0.6, 0.85);
        cameraScale = 1 + z * 0.2;
        cameraX = -z * 30;
      }
      cameraScale *= this.state.zoom;
      cameraX += this.state.panX;
      cameraY += this.state.panY;
    }
    const s = scale * cameraScale;
    const coord = (n) => ({
      x: x + (n.x * cs - n.y * sn) * s + cameraX,
      y: y + (n.x * sn + n.y * cs) * s + cameraY,
    });
    c.save();
    c.beginPath();
    c.rect(66, 258, 1788, 675);
    c.clip();
    for (let d = 1; d <= Math.min(visibleDepth, 10); d++) {
      this.circle(
        x + cameraX,
        y + cameraY,
        (55 + d * 45) * s,
        null,
        "#8da7d00a",
      );
    }
    // Quiet structural edges preserve the actual parent-child relationships.
    for (const n of h.nodes) {
      if (!n.depth || n.depth > visibleDepth || n.depth > reveal * 11 + 0.3)
        continue;
      const a = coord(n.parent),
        b = coord(n),
        isActive = activeIds.has(n.id),
        col = C[n.root];
      this.line(
        a.x,
        a.y,
        b.x,
        b.y,
        isActive ? col + "c5" : col + "22",
        isActive ? 1.8 : 0.7,
      );
    }
    for (const n of h.nodes) {
      if (!n.depth || n.depth > visibleDepth || n.depth > reveal * 11 + 0.3)
        continue;
      const b = coord(n),
        isActive = activeIds.has(n.id),
        col = C[n.root];
      const r =
        (n.depth === 1
          ? 7
          : n.depth === 2
            ? 3.7
            : n.children.length
              ? 1.8
              : 1.15) * Math.min(s, 1.4);
      this.circle(
        b.x,
        b.y,
        r,
        isActive ? col : col + (n.depth < 3 ? "b0" : "65"),
      );
      if (isActive && n.depth > 3)
        this.circle(b.x, b.y, r + 5, null, col + "45");
      if (n.depth < 3 || isActive)
        this.hitRegions.push({
          x: b.x,
          y: b.y,
          r: Math.max(10, r + 4),
          type: "node",
          node: n,
        });
    }
    if (this.state.particles && !this.state.reducedMotion)
      active.slice(0, 24).forEach((row, i) => {
        const phase = (this.time * 0.32 + i * 0.17) % 1;
        const path = row.path;
        const depth = Math.max(
          1,
          Math.min(path.length - 1, Math.floor(phase * (path.length - 1)) + 1),
        );
        const a = h.lookup.get(keyOf(path.slice(0, depth))),
          b = h.lookup.get(keyOf(path.slice(0, depth + 1)));
        if (!a || !b || b.depth > visibleDepth) return;
        const aa = coord(a),
          bb = coord(b),
          p = (phase * (path.length - 1)) % 1;
        this.circle(lerp(aa.x, bb.x, p), lerp(aa.y, bb.y, p), 3.2, C[row.root]);
      });
    this.circle(x + cameraX, y + cameraY, 43 * s, C.bg, C.PSYCHO + "50");
    this.logo(x + cameraX + 2, y + cameraY + 4, 0.9 * s, C.ink);
    if (labels && this.state.labels) {
      h.root.children.forEach((n) => {
        const pos = coord(n);
        this.rect(pos.x - 48, pos.y - 40, 96, 28, C.bg);
        this.text(
          n.label,
          pos.x,
          pos.y - 19,
          15,
          C[n.root],
          600,
          "center",
          true,
        );
      });
      const labelled = h.nodes
        .filter((n) => n.depth === 2 && activeIds.has(n.id))
        .slice(0, 5);
      const abbreviate = (label) =>
        ({
          Sleep_Circadian_and_Restoration: "Sleep & circadian",
          Medical_Assessment_and_Physiology_Testing: "Medical assessment",
          Trauma_Memory_Imagery_and_Narrative_Work: "Trauma & memory",
          Change_Process_and_Therapeutic_Phases: "Change & readiness",
          Care_Navigation_Access_and_Coordination: "Care navigation",
          Work_Education_and_Role_Functioning: "Work & role support",
          Cognitive_Behavioral_Therapies: "Cognitive behavioral therapies",
          Insight_Oriented_Therapies: "Insight-oriented therapies",
          Social_Support_and_Belonging: "Support & belonging",
        })[label] || this.fit(clean(label), 215, 14);
      const occupied = [];
      for (const n of labelled) {
        const pos = coord(n),
          label = abbreviate(n.label);
        this.font(14);
        const w = this.ctx.measureText(label).width + 22;
        let lx = pos.x - w / 2,
          ly = pos.y + 14;
        for (
          let attempt = 0;
          attempt < 10 &&
          occupied.some(
            (b) =>
              lx < b.x + b.w + 9 &&
              lx + w > b.x - 9 &&
              ly < b.y + 35 &&
              ly + 29 > b.y - 6,
          );
          attempt++
        )
          ly += 34;
        occupied.push({ x: lx, y: ly, w });
        this.line(pos.x, pos.y, lx + w / 2, ly, C[n.root] + "70");
        this.rect(lx, ly, w, 27, C.bg + "f5", C[n.root] + "45", 7);
        this.text(label, lx + w / 2, ly + 19, 14, C[n.root], 500, "center");
      }
    }
    c.restore();
    return coord;
  }
  opening() {
    const c = this.caseData;
    this.pill("FROM COMPLAINT TO CONTINUOUS LEARNING", 88, 269, c.color, 14);
    this.text(c.headline[0], 88, 418, 78, C.ink, 500);
    this.text(c.headline[1], 88, 514, 78, c.color, 500);
    this.wrap(`${c.name}, ${c.age}. ${c.role}.`, 91, 587, 770, 29, C.ink, 1.5);
    this.wrap(
      "A complete visual journey through the PHOENIX Engine.",
      91,
      643,
      690,
      27,
      C.muted,
      1.5,
    );
    this.line(91, 735, 460, 735, C.line);
    this.text("01", 91, 783, 14, c.color, 400, "left", true);
    this.text("Listen", 132, 784, 20);
    this.text("02", 266, 783, 14, c.color, 400, "left", true);
    this.text("Model", 307, 784, 20);
    this.text("03", 451, 783, 14, c.color, 400, "left", true);
    this.text("Adapt", 492, 784, 20);
    this.graph(1350, 535, 0.9, {
      active: this.model.seeds.slice(0, 6),
      rotate: this.state.reducedMotion ? 0 : -0.09 + this.p * 0.15,
      reveal: range(this.p, 0, 0.42),
    });
    this.pill("7,265 REAL SOLUTION LEAVES", 1210, 897, C.muted, 13);
    this.caption =
      "An explanatory film grounded in the repository. All case data and numerical evidence are illustrative.";
  }
  intake() {
    const d = this.caseData,
      active = Math.min(5, Math.floor(this.p * 7));
    this.rect(88, 271, 1122, 635, "#111a27", C.line, 23);
    this.panelHeading("FREE-TEXT COMPLAINT", 121, 315);
    this.text("“", 119, 399, 80, d.color, 400);
    // Word layout retains exact quoted text and reveals source spans without rewriting the complaint.
    const words = d.complaint.split(/(\s+)/);
    let x = 139,
      y = 400;
    this.font(30, 400);
    let offset = 0;
    for (const word of words) {
      const w = this.ctx.measureText(word).width;
      if (x + w > 1152) {
        x = 139;
        y += 50;
      }
      let ci = -1;
      for (let i = 0; i < d.criteria.length; i++) {
        const start = d.complaint.indexOf(d.criteria[i].span);
        if (
          offset < start + d.criteria[i].span.length &&
          offset + word.length > start
        )
          ci = i;
      }
      if (ci >= 0 && ci <= active) {
        this.rect(x - 1, y - 31, w + 2, 42, d.color + "14", null, 2);
        this.line(
          x,
          y + 6,
          x + w,
          y + 6,
          ci === active ? d.color : d.color + "55",
          2,
        );
      }
      this.text(
        word,
        x,
        y,
        30,
        ci === active ? C.ink : ci >= 0 && ci < active ? d.color : C.muted,
        400,
      );
      x += w;
      offset += word.length;
      this.font(30, 400);
    }
    this.pill(
      `C${active + 1}  ${d.criteria[active].label.toUpperCase()}`,
      125,
      860,
      d.color,
      14,
    );
    this.rect(1240, 271, 592, 282, "#111a27", C.line, 23);
    this.panelHeading(
      "PERSON",
      1276,
      316,
      "Stable characteristics & preferences",
    );
    d.person.forEach((s, i) => {
      this.circle(1280, 380 + i * 49, 3.5, C.PSYCHO);
      this.text(s, 1299, 387 + i * 49, 19, C.muted);
    });
    this.rect(1240, 576, 592, 330, "#111a27", C.line, 23);
    this.panelHeading(
      "CONTEXT",
      1276,
      621,
      "The conditions around this moment",
    );
    d.context.forEach((s, i) => {
      this.circle(1280, 682 + i * 49, 3.5, C.SOCIAL);
      this.text(s, 1299, 689 + i * 49, 21, C.muted);
    });
    this.text(
      "Retained alongside the complaint",
      1276,
      873,
      15,
      C.faint,
      400,
      "left",
      true,
    );
    this.caption =
      "Separate symptoms from circumstances. Contextual barriers remain part of the reasoning throughout.";
  }
  criteria() {
    const d = this.caseData,
      focus = Math.min(5, Math.floor(this.p * 6.7));
    [
      "Decompose",
      "Critic review",
      "Hybrid retrieval",
      "Leaf grounding",
    ].forEach((s, i) => {
      const x = 88 + i * 447;
      this.rect(x, 276, 404, 66, C.panel, C.line, 14);
      this.badge(s, x + 32, 309, String(i + 1), i === 2 ? C.BIO : d.color);
      if (i < 3)
        this.arrow({ x: x + 414, y: 309 }, { x: x + 436, y: 309 }, C.faint);
    });
    d.criteria.forEach((cr, i) => {
      const x = 88 + (i % 3) * 588,
        y = 376 + Math.floor(i / 3) * 172;
      this.saveAlpha(
        0.28 + 0.72 * range(this.p, i * 0.09, i * 0.09 + 0.1),
        () => {
          this.rect(
            x,
            y,
            568,
            149,
            i === focus ? d.color + "13" : C.panel,
            i === focus ? d.color + "80" : C.line,
            17,
          );
          this.text(
            `C${i + 1} · ${cr.path[2]}`,
            x + 24,
            y + 33,
            12,
            i === focus ? d.color : C.muted,
            400,
            "left",
            true,
          );
          this.text(cr.label, x + 24, y + 70, 24, C.ink, 500);
          this.wrap(
            "“" + cr.span + "”",
            x + 24,
            y + 106,
            510,
            17,
            C.muted,
            1.35,
            400,
            2,
          );
        },
      );
    });
    const path = d.criteria[focus].path;
    this.rect(88, 748, 1744, 154, "#0e1825", C.line, 15);
    this.panelHeading(`TRACE C${focus + 1} · EXACT REPOSITORY PATH`, 115, 781);
    this.wrap(
      path.map(clean).join("  ›  "),
      115,
      823,
      1675,
      18,
      d.color,
      1.45,
      400,
      2,
    );
    this.text(
      "A leaf match names a concept. It does not establish a diagnosis. Uncertain live mappings may remain UNMAPPED.",
      115,
      883,
      17,
      C.muted,
    );
    this.caption =
      "Dense embeddings + BM25 + token overlap + fuzzy matching support retrieval after decomposition and critic review.";
  }
  ontology() {
    const d = this.caseData;
    this.graph(1210, 602, 0.74, {
      active: this.model.seeds.slice(0, 6),
      reveal: range(this.p, 0, 0.32),
      allowCamera: true,
    });
    this.rect(88, 281, 530, 628, "#101925f4", C.line, 20);
    this.text("A structured universe", 119, 329, 30, C.ink, 500);
    this.text("7,265", 115, 420, 79, d.color, 500);
    this.text("PREDICTOR leaves in this snapshot", 119, 458, 20, C.muted);
    [
      ["BIO", "Body, sleep and restoration"],
      ["PSYCHO", "Therapies, skills and meaning"],
      ["SOCIAL", "Relationships, access and resources"],
    ].forEach(([k, s], i) => {
      this.circle(125, 524 + i * 71, 5, C[k]);
      this.text(k, 145, 529 + i * 71, 16, C[k], 500, "left", true);
      this.text(s, 145, 556 + i * 71, 18, C.muted);
    });
    this.line(119, 733, 586, 733, C.line);
    this.wrap(
      "Actual paths reach 10 levels. Depth varies by branch; no invented micro-options.",
      119,
      771,
      450,
      21,
      C.ink,
      1.5,
    );
    this.text(
      `${this.hierarchy.displayedLeaves.toLocaleString()} sampled leaves shown`,
      119,
      878,
      14,
      C.faint,
      400,
      "left",
      true,
    );
    this.caption =
      "Five ontologies, distinct roles: CRITERION · PREDICTOR · PERSON · CONTEXT · HAPA.";
  }
  lineage() {
    const candidate = [...this.model.seeds]
      .filter((r) => r.path[1] !== "Insight_Oriented_Therapies")
      .sort((a, b) => b.path.length - a.path.length)[0];
    const zoom = this.state.reducedMotion
      ? 0.74
      : lerp(0.58, 0.84, range(this.p, 0, 0.58));
    this.graph(548, 600, zoom, {
      active: [candidate],
      labels: false,
      allowCamera: true,
    });
    this.pill("ONE PATH · REAL ANCESTRY", 89, 287, C[candidate.root], 13);
    this.text(
      `${candidate.path.length} levels`,
      89,
      865,
      35,
      C[candidate.root],
      500,
    );
    this.text("from branch to concrete candidate", 89, 904, 19, C.muted);
    this.rect(1020, 278, 812, 637, "#101925f7", C.line, 22);
    this.panelHeading("THE PATH BEHIND THE OPTION", 1055, 321);
    const gap = Math.min(78, 490 / candidate.path.length),
      startY = 367;
    candidate.path.forEach((segment, i) => {
      const x = 1058 + i * 15,
        y = startY + i * gap;
      const active = range(this.p, 0.04 + i * 0.065, 0.17 + i * 0.065);
      this.saveAlpha(0.24 + 0.76 * active, () => {
        if (i)
          this.line(
            x + 16,
            y - gap + 49,
            x + 16,
            y,
            C[candidate.root] + "70",
            2,
          );
        this.rect(
          x,
          y,
          735 - i * 15,
          gap - 9,
          i === candidate.path.length - 1 ? C[candidate.root] + "1a" : C.panel,
          C[candidate.root] + (i === candidate.path.length - 1 ? "aa" : "35"),
          10,
        );
        this.text(
          String(i + 1).padStart(2, "0"),
          x + 16,
          y + 30,
          13,
          C[candidate.root],
          400,
          "left",
          true,
        );
        this.wrap(
          clean(segment),
          x + 57,
          y + 29,
          645 - i * 15,
          18,
          C.ink,
          1.35,
          500,
          2,
        );
      });
    });
    this.text(
      "Structural membership supports auditability, not a promise of benefit.",
      1056,
      887,
      16,
      C.muted,
    );
    this.caption = `Featured ${candidate.root} option: ${candidate.label}. Every visible edge follows the stored parent-child lineage.`;
  }
  observationModel() {
    const d = this.caseData;
    this.panelHeading("MODIFIABLE PREDICTORS", 113, 298);
    this.panelHeading("OBSERVABLE CRITERIA", 911, 298);
    const predictors = this.model.seeds.slice(0, 6);
    const left = predictors.map((r, i) => ({ x: 449, y: 355 + i * 83 })),
      right = d.criteria.map((r, i) => ({ x: 869, y: 355 + i * 83 }));
    predictors.forEach((r, i) => {
      r.links.forEach((target, j) =>
        this.saveAlpha(range(this.p, 0.06 + i * 0.045, 0.2 + i * 0.045), () => {
          this.bezier(
            left[i],
            right[target],
            C[r.root] + "80",
            1 + r.impact * 2,
          );
          if (this.state.particles && !this.state.reducedMotion) {
            const t = (this.time * 0.21 + j * 0.18 + i * 0.11) % 1;
            this.circle(
              lerp(left[i].x, right[target].x, t),
              lerp(left[i].y, right[target].y, ease(t)),
              3,
              C[r.root],
            );
          }
        }),
      );
    });
    predictors.forEach((r, i) => {
      this.rect(88, 322 + i * 83, 363, 60, C.panel, C[r.root] + "60", 12);
      this.circle(110, 351 + i * 83, 4, C[r.root]);
      this.text(this.fit(r.label, 308, 20), 125, 358 + i * 83, 20);
    });
    d.criteria.forEach((r, i) => {
      this.rect(869, 322 + i * 83, 370, 60, C.panel, d.color + "50", 12);
      this.text(`C${i + 1}`, 889, 358 + i * 83, 14, d.color, 400, "left", true);
      this.text(this.fit(r.label, 280, 19), 934, 358 + i * 83, 19);
    });
    this.rect(1280, 277, 552, 625, "#141c29", C.line, 20);
    this.panelHeading("ACTOR ↔ CRITIC", 1316, 321);
    this.wrap(
      "Propose. Challenge. Refine.",
      1316,
      376,
      454,
      32,
      C.ink,
      1.4,
      500,
    );
    const revised = this.p > 0.43,
      passed = this.p > 0.72;
    this.pill(
      passed
        ? "PASS · PROCEED"
        : revised
          ? "REVISE · FEEDBACK"
          : "DRAFT · HYPOTHESIS",
      1316,
      490,
      passed ? C.SOCIAL : C.gold,
      14,
    );
    this.wrap(
      revised
        ? d.revision
        : "Generate hypothetical evidence with HyDE, retrieve ontology candidates and construct a first observation model.",
      1316,
      553,
      454,
      23,
      C.muted,
      1.55,
    );
    this.line(1316, 744, 1795, 744, C.line);
    this.wrap(
      "Check grounding, continuity, ontology validity and evidence quality.",
      1316,
      787,
      445,
      21,
      C.muted,
      1.45,
    );
    this.caption =
      "These first links are hypotheses for measurement. The critic can require revision before collection begins.";
  }
  ema() {
    const d = this.caseData;
    this.rect(88, 279, 548, 623, C.panel, C.line, 23);
    this.panelHeading("ECOLOGICAL MOMENTARY ASSESSMENT", 120, 323);
    this.wrap(
      "A small window into daily life.",
      120,
      389,
      440,
      40,
      C.ink,
      1.3,
      500,
    );
    d.signals.forEach((s, i) => {
      this.text(s, 121, 526 + i * 101, 20, C.muted);
      for (let j = 0; j < 8; j++)
        this.rect(
          121 + j * 57,
          546 + i * 101,
          46,
          17,
          j < 3 + i ? d.color + "cc" : C.line,
          null,
          6,
        );
    });
    this.pill("BRIEF · REPEATED · MODEL-GUIDED", 119, 871, d.color, 12);
    this.panelHeading("SIMULATED LONGITUDINAL RECORD", 698, 311);
    this.text("Watch the pattern develop.", 698, 363, 32, C.ink, 500);
    d.signals.forEach((s, i) => {
      const y = 436 + i * 145;
      this.text(s, 699, y, 18, [C.PSYCHO, C.BIO, C.SOCIAL][i]);
      this.sparkline(
        simulatedSeries(d, i),
        699,
        y + 21,
        1100,
        77,
        [C.PSYCHO, C.BIO, C.SOCIAL][i],
        range(this.p, 0.03, 0.85),
      );
    });
    this.text("Day 01", 699, 882, 13, C.faint, 400, "left", true);
    this.text(
      `Day ${Math.ceil(d.observations / 3)}`,
      1800,
      882,
      13,
      C.faint,
      400,
      "right",
      true,
    );
    this.caption =
      "Measurement follows the initial model. The moving traces are deterministic synthetic data, not patient observations.";
  }
  sparkline(values, x, y, w, h, color, progress = 1, fill = true) {
    const c = this.ctx;
    this.line(x, y + h, x + w, y + h, C.line);
    const count = Math.max(2, Math.floor(values.length * progress));
    c.beginPath();
    values.slice(0, count).forEach((v, i) => {
      const xx = x + (i * w) / (values.length - 1),
        yy = y + (1 - v) * h;
      if (!i) c.moveTo(xx, yy);
      else c.lineTo(xx, yy);
    });
    c.strokeStyle = color;
    c.lineWidth = 2.5;
    c.stroke();
    const endX = x + ((count - 1) * w) / (values.length - 1),
      endY = y + (1 - values[count - 1]) * h;
    this.circle(endX, endY, 4, color);
    if (fill) {
      c.lineTo(endX, y + h);
      c.lineTo(x, y + h);
      c.closePath();
      const g = c.createLinearGradient(0, y, 0, y + h);
      g.addColorStop(0, color + "30");
      g.addColorStop(1, color + "00");
      c.fillStyle = g;
      c.fill();
    }
  }
  readiness() {
    const d = this.caseData,
      r = this.state.readiness ?? d.readiness,
      m = methodFor(r, d.method);
    this.rect(88, 281, 573, 624, C.panel, C.line, 20);
    this.panelHeading("HUA · READINESS CLASSIFIER", 121, 326);
    this.circle(374, 551, 137, null, C.line, 16);
    const c = this.ctx;
    c.beginPath();
    c.arc(
      374,
      551,
      137,
      -Math.PI / 2,
      -Math.PI / 2 + ((TAU * r) / 100) * range(this.p, 0, 0.3),
    );
    c.strokeStyle = d.color;
    c.lineWidth = 16;
    c.lineCap = "round";
    c.stroke();
    this.text(
      Math.round(r * range(this.p, 0, 0.3)),
      374,
      574,
      90,
      C.ink,
      500,
      "center",
    );
    this.text("READINESS / 100", 374, 618, 13, C.muted, 400, "center", true);
    this.text(`${d.observations} observations`, 121, 777, 21, C.muted);
    this.text(`${d.missing}% missing in this scenario`, 121, 819, 21, C.muted);
    this.text(
      "Illustrative diagnostics and tier assignment",
      121,
      871,
      14,
      C.faint,
      400,
      "left",
      true,
    );
    const tiers = [
      ["0", "Descriptives", "A short or insufficient record"],
      ["1", "Correlations", "Exploratory associations"],
      ["2", "Reduced baseline", "Constrained model complexity"],
      ["3", "gVAR / tv-gVAR", "Subject to full diagnostics"],
    ];
    tiers.forEach(([n, title, sub], i) => {
      const y = 280 + i * 126,
        active = i === m.tier;
      this.rect(
        699,
        y,
        1133,
        109,
        active ? d.color + "15" : C.panel,
        active ? d.color + "90" : C.line,
        16,
      );
      this.text(
        n,
        739,
        y + 65,
        33,
        active ? d.color : C.faint,
        500,
        "left",
        true,
      );
      this.text(title, 803, y + 46, 26, C.ink, 500);
      this.text(sub, 803, y + 79, 18, C.muted);
      if (active) this.pill("SELECTED", 1652, y + 58, d.color, 12);
    });
    this.wrap(
      "Stationarity · collinearity · effective sample size · missingness",
      701,
      835,
      1090,
      23,
      C.ink,
      1.5,
    );
    this.wrap(
      "The slider illustrates a policy. Runtime selection uses the diagnostic execution plan, not one score alone.",
      701,
      883,
      1090,
      17,
      C.muted,
      1.4,
    );
    this.caption = `Selected in this scenario: ${m.name}. ${m.note}`;
  }
  network() {
    const d = this.caseData,
      r = this.state.readiness ?? d.readiness,
      m = methodFor(r, d.method);
    this.pill(m.code, 88, 281, d.color, 13);
    const nodes = [
      ...d.signals,
      ...d.criteria.slice(0, 3).map((c) => c.label),
    ].map((label, i) => ({
      label,
      x: 536 + Math.cos(-Math.PI / 2 + (i * TAU) / 6) * 245,
      y: 600 + Math.sin(-Math.PI / 2 + (i * TAU) / 6) * 222,
    }));
    if (m.tier > 0)
      for (let i = 0; i < 6; i++)
        for (let j = i + 1; j < 6; j++) {
          const h = hash(d.id + i + j);
          if (h < 0.38) continue;
          this.saveAlpha(range(this.p, 0.04, 0.35), () => {
            const color =
              (i % 2 ? C.BIO : C.PSYCHO) +
              Math.floor((0.3 + h * 0.6) * 255)
                .toString(16)
                .padStart(2, "0");
            if (m.directed) this.arrow(nodes[i], nodes[j], color, 1 + h * 3);
            else
              this.line(
                nodes[i].x,
                nodes[i].y,
                nodes[j].x,
                nodes[j].y,
                color,
                1 + h * 3,
              );
          });
        }
    nodes.forEach((n, i) => {
      this.circle(n.x, n.y, 45, C.bg, i < 3 ? C.BIO : d.color, 2);
      this.circle(n.x, n.y, 31, (i < 3 ? C.BIO : d.color) + "11");
      this.text(
        i < 3 ? `P${i + 1}` : `C${i - 2}`,
        n.x,
        n.y + 7,
        19,
        i < 3 ? C.BIO : d.color,
        500,
        "center",
        true,
      );
      this.rect(n.x - 131, n.y + 55, 262, 33, C.bg + "f0", null, 6);
      this.text(
        this.fit(n.label, 248, 16),
        n.x,
        n.y + 78,
        16,
        C.ink,
        400,
        "center",
      );
    });
    this.text(
      m.tier === 0
        ? "OBSERVATIONS ONLY"
        : m.directed
          ? "LAGGED ASSOCIATIONS"
          : "UNDIRECTED ASSOCIATIONS",
      536,
      894,
      14,
      C.muted,
      400,
      "center",
      true,
    );
    this.rect(1034, 278, 798, 626, C.panel, C.line, 21);
    this.panelHeading("MOMENTARY IMPACT QUANTIFICATION", 1069, 324);
    this.wrap(
      m.tier === 0
        ? "More data before network impact."
        : "Which predictors matter here?",
      1069,
      383,
      690,
      34,
      C.ink,
      1.35,
      500,
    );
    this.model.seeds
      .slice(0, 5)
      .sort((a, b) => b.impact - a.impact)
      .forEach((row, i) => {
        const y = 492 + i * 65;
        this.text(row.label, 1069, y, 19, C.muted);
        this.rect(1440, y - 16, 242, 10, C.line, null, 5);
        this.rect(
          1440,
          y - 16,
          242 * row.impact * range(this.p, 0.16 + i * 0.04, 0.42 + i * 0.04),
          10,
          C[row.root],
          null,
          5,
        );
        this.text(
          m.tier === 0 ? "n/a" : row.impact.toFixed(2),
          1770,
          y,
          17,
          m.tier === 0 ? C.faint : C[row.root],
          400,
          "right",
          true,
        );
      });
    this.wrap(
      m.tier === 0
        ? "Bars show authored prior relevance only. Network-derived impact is unavailable in this scenario."
        : "Illustrative impact combines leave-one-predictor-out error change with coefficient magnitude.",
      1069,
      843,
      690,
      18,
      C.muted,
      1.45,
    );
    this.caption =
      m.tier === 0
        ? "Sparse data: retain descriptive monitoring and prior evidence. Do not invent a fitted network."
        : "Predictive associations and impact scores do not establish causal effects or treatment effectiveness.";
  }
  search() {
    const phase = Math.min(2, Math.floor(this.p * 3)),
      labels = [
        "01  Cover domains",
        "02  Round-robin breadth",
        "03  Refine by score",
      ];
    const selected = this.model.selected.filter((r) => r.phase <= phase);
    this.graph(1165, 594, 0.74, {
      active: selected,
      reveal: 1,
      allowCamera: true,
    });
    this.rect(88, 277, 535, 633, "#101925f7", C.line, 21);
    this.panelHeading("BFS CANDIDATE SELECTOR", 121, 320);
    this.wrap(
      "Breadth is a deliberate choice.",
      121,
      377,
      448,
      36,
      C.ink,
      1.35,
      500,
    );
    labels.forEach((label, i) => {
      const y = 492 + i * 86;
      this.rect(
        119,
        y - 32,
        469,
        60,
        i === phase ? this.caseData.color + "18" : "#121c29",
        i === phase ? this.caseData.color + "85" : C.line,
        12,
      );
      this.text(label, 140, y + 5, 22, i === phase ? C.ink : C.muted, 500);
    });
    this.wrap(
      [
        "Take one top candidate from each eligible secondary domain.",
        "Return across domains until the breadth allocation is filled.",
        "Fill remaining slots by global score while retaining the earlier breadth.",
      ][phase],
      121,
      752,
      443,
      23,
      C.muted,
      1.5,
    );
    this.text(
      `${selected.length} of ${this.model.selected.length} candidate slots`,
      121,
      876,
      14,
      this.caseData.color,
      400,
      "left",
      true,
    );
    this.pill(
      "0.45 MAPPING + 0.25 HYDE + 0.20 PERSONAL + 0.10 DOMAIN",
      824,
      891,
      C.muted,
      12,
    );
    this.caption =
      "The three selection phases follow target_refinement.py. Candidate component scores here are authored simulation fixtures.";
  }
  compare() {
    const baseline = baselineSelect(
        this.model,
        this.state.compare,
        this.state.budget,
      ),
      broad = this.model.selected;
    const covA = coverage(baseline),
      covB = coverage(broad);
    const label =
      this.state.compare === "psychodynamic"
        ? "Psychodynamic focus"
        : this.state.compare === "cbt"
          ? "CBT branch focus"
          : "Integrative score-first example";
    [
      [88, label, C.gold, baseline, covA],
      [979, "PHOENIX breadth-first", this.caseData.color, broad, covB],
    ].forEach(([x, title, color, rows, cov]) => {
      this.rect(x, 278, 853, 630, "#101925", C.line, 22);
      this.pill(
        x === 88 ? "ILLUSTRATIVE COMPARATOR" : "ONTOLOGY-GUIDED SEARCH",
        x + 30,
        317,
        color,
        11,
      );
      this.text(title, x + 30, 373, 29, C.ink, 500);
      this.graph(x + 426, 588, 0.4, {
        active: rows.slice(0, Math.ceil(rows.length * range(this.p, 0, 0.55))),
        labels: false,
      });
      this.rect(x + 24, 798, 805, 89, "#172131", null, 12);
      this.text(String(cov.domains), x + 52, 839, 32, color, 500);
      this.text("domains explored", x + 102, 836, 16, C.muted);
      this.text(`${cov.criteria} / 6`, x + 354, 839, 32, color, 500);
      this.text("linked criteria", x + 465, 836, 16, C.muted);
      this.text(
        `${rows.length} candidates within a shared ${this.state.budget}-slot budget`,
        x + 52,
        870,
        14,
        C.faint,
        400,
        "left",
        true,
      );
    });
    this.caption =
      this.state.compare === "integrative"
        ? "Integrative care may already span domains. Compare policies without assuming all psychotherapy is confined to one branch."
        : "Same complaint, criteria and candidate pool. The scoped comparator illustrates a search constraint, not therapy in general.";
  }
  targets() {
    const top = [...this.model.seeds]
      .filter((r) => !r.path.includes("Insight_Oriented_Therapies"))
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);
    this.panelHeading("RANKED, CROSS-DOMAIN CANDIDATES", 89, 288);
    top.forEach((r, i) => {
      const y = 325 + i * 174;
      this.saveAlpha(range(this.p, i * 0.07, 0.15 + i * 0.07), () => {
        this.rect(88, y, 1034, 149, C.panel, C[r.root] + "55", 18);
        this.text(`0${i + 1}`, 116, y + 50, 25, C[r.root], 500, "left", true);
        this.text(r.label, 177, y + 50, 28, C.ink, 500);
        this.text(
          `${r.root}  ›  ${this.fit(clean(r.path[1]), 745, 18)}`,
          177,
          y + 86,
          18,
          C.muted,
        );
        this.text(
          `Illustrative candidate score ${r.score.toFixed(3)} · linked to ${r.links.length} criteria`,
          177,
          y + 122,
          15,
          C.faint,
          400,
          "left",
          true,
        );
      });
    });
    this.rect(1162, 279, 670, 626, "#141d2a", C.line, 21);
    this.panelHeading("TARGET SELECTION + CRITIC", 1196, 323);
    this.wrap(
      "Relevance is the beginning.",
      1196,
      387,
      565,
      36,
      C.ink,
      1.35,
      500,
    );
    [
      "Ontology lineage",
      "Evidence and data leakage",
      "Suitability and feasibility",
      "Safety and personal preferences",
    ].forEach((s, i) => {
      this.circle(1205, 501 + i * 57, 11, C.SOCIAL + "17", C.SOCIAL + "55");
      this.text("✓", 1205, 506 + i * 57, 14, C.SOCIAL, 400, "center");
      this.text(s, 1230, 509 + i * 57, 21, C.muted);
    });
    this.wrap(this.caseData.gate, 1196, 763, 563, 22, C.gold, 1.5);
    this.caption =
      "These illustrative targets remain proposals for review. No candidate is guaranteed to work or to fit the person.";
  }
  fusion() {
    const r = this.state.readiness ?? this.caseData.readiness,
      w = weights(r);
    this.rect(88, 281, 1040, 623, C.panel, C.line, 22);
    this.panelHeading("STEP 04 · UPDATED OBSERVATION MODEL", 120, 324);
    this.wrap(
      "Let personal evidence earn its weight.",
      120,
      391,
      900,
      40,
      C.ink,
      1.35,
      500,
    );
    const bx = 121,
      by = 513,
      bw = 937;
    this.rect(bx, by, bw, 75, C.line, null, 14);
    this.rect(bx, by, bw * w.nomothetic, 75, C.BIO, null, 14);
    this.rect(
      bx + bw * w.nomothetic,
      by,
      bw * w.idiographic,
      75,
      this.caseData.color,
      null,
      14,
    );
    this.text(
      `${Math.round(w.nomothetic * 100)}%`,
      bx + 25,
      by + 51,
      36,
      C.bg,
      600,
    );
    this.text(
      `${Math.round(w.idiographic * 100)}%`,
      bx + bw - 25,
      by + 51,
      36,
      C.bg,
      600,
      "right",
    );
    this.text("Nomothetic", 121, 641, 27, C.BIO, 500);
    this.text("Population-level knowledge", 121, 678, 20, C.muted);
    this.text("Idiographic", 1058, 641, 27, this.caseData.color, 500, "right");
    this.text("Within-person evidence", 1058, 678, 20, C.muted, 400, "right");
    this.rect(120, 748, 936, 107, "#0b1421", C.line, 12);
    this.text(
      "w_personal = 0.30 + 0.50 × readiness / 100",
      150,
      791,
      21,
      C.ink,
      400,
      "left",
      true,
    );
    this.text(
      `Readiness ${r}/100 → personal weight ${w.idiographic.toFixed(2)} · bounded to 0.30–0.80`,
      150,
      827,
      15,
      C.muted,
      400,
      "left",
      true,
    );
    this.rect(1166, 281, 666, 623, C.panel, C.line, 22);
    this.panelHeading("REFINE, RETAIN, RECHECK", 1201, 325);
    const before = [0.4, 0.66, 0.47, 0.6],
      after = [0.73, 0.57, 0.79, 0.51];
    [
      "Sleep & restoration",
      "Thinking & emotion",
      "Access & support",
      "Daily routines",
    ].forEach((s, i) => {
      const y = 410 + i * 95;
      this.text(s, 1201, y, 22, C.muted);
      this.rect(1201, y + 24, 529 * before[i], 9, C.faint + "70", null, 4);
      this.rect(
        1201,
        y + 41,
        529 * lerp(before[i], after[i], range(this.p, 0.15, 0.65)),
        9,
        this.caseData.color,
        null,
        4,
      );
    });
    this.pill("GRAY: PRIOR  /  COLOR: UPDATED", 1201, 834, C.muted, 12);
    this.text(
      "CRITIC · LINEAGE + CONTINUITY · PASS / REVISE",
      1201,
      885,
      13,
      C.gold,
      400,
      "left",
      true,
    );
    this.caption =
      "The formula is from the engine. The illustrated rank changes are synthetic; more data does not automatically mean greater certainty.";
  }
  hapa() {
    const h = this.caseData.hapa;
    this.pill(
      "STEP 05 · HEALTH ACTION PROCESS APPROACH",
      88,
      283,
      this.caseData.color,
      13,
    );
    const cards = [
      ["PHASE", h.phase, C.PSYCHO],
      ["BARRIER", h.barrier, C.gold],
      ["COPING PLAN", h.coping, C.SOCIAL],
      ["SUPPORT", h.support, C.BIO],
    ];
    cards.forEach(([title, body, col], i) => {
      const x = 88 + (i % 2) * 474,
        y = 338 + Math.floor(i / 2) * 220;
      this.rect(x, y, 452, 196, C.panel, C.line, 17);
      this.text(title, x + 25, y + 38, 13, col, 400, "left", true);
      this.wrap(
        body,
        x + 25,
        y + 85,
        399,
        i === 0 ? 30 : 21,
        C.ink,
        1.5,
        i === 0 ? 500 : 400,
        3,
      );
    });
    this.text("Barrier relevance", 88, 840, 18, C.muted);
    this.text(
      "0.60 target + 0.20 person + 0.15 context + 0.05 complaint",
      88,
      878,
      15,
      C.faint,
      400,
      "left",
      true,
    );
    // Product-style delivery preview, with a readable action and a real coping alternative.
    this.rect(1158, 264, 581, 654, "#080d15", "#3f4b61", 48);
    this.rect(1171, 277, 555, 628, "#f3f1ec", null, 38);
    this.rect(1375, 289, 145, 21, "#111b29", null, 15);
    this.text("P H O E N I X", 1212, 355, 14, "#586173", 600);
    this.text(
      `A next step for ${this.caseData.name}`,
      1212,
      408,
      27,
      "#172336",
      600,
    );
    this.text(
      "YOUR CHOICE · YOUR PACE",
      1212,
      445,
      11,
      "#728097",
      400,
      "left",
      true,
    );
    this.line(1212, 470, 1684, 470, "#d9dddF");
    this.wrap(h.when, 1212, 515, 460, 19, "#566579", 1.4, 500);
    this.wrap(h.action, 1212, 580, 455, 28, "#172336", 1.43, 500, 5);
    this.rect(1209, 776, 474, 64, "#263a47", null, 16);
    this.text("Save my next step", 1446, 817, 21, "#f5f4f0", 500, "center");
    this.text(
      "Preview of an authored plan, not a live prescription",
      1446,
      873,
      12,
      "#687487",
      400,
      "center",
    );
    this.pill(
      "CRITIC · HAPA CONSISTENCY + SAFETY · PASS / REVISE",
      88,
      928,
      C.gold,
      12,
    );
    this.caption =
      "HAPA connects phase, barriers, action planning, coping planning and monitoring. A relevant target still needs a feasible plan.";
  }
  loop() {
    const names = [
      "Observation model",
      "Daily-life observations",
      "Readiness & analysis",
      "Impact & targets",
      "HAPA plan",
      "History ledger",
    ];
    const cols = [C.PSYCHO, C.BIO, C.BIO, C.gold, C.SOCIAL, C.PSYCHO];
    const nodes = names.map((label, i) => ({
      label,
      x: 606 + Math.cos(-Math.PI / 2 + (i * TAU) / 6) * 300,
      y: 601 + Math.sin(-Math.PI / 2 + (i * TAU) / 6) * 245,
    }));
    nodes.forEach((n, i) => {
      const next = nodes[(i + 1) % 6],
        angle = Math.atan2(next.y - n.y, next.x - n.x);
      const a = {
          x: n.x + Math.cos(angle) * 70,
          y: n.y + Math.sin(angle) * 52,
        },
        b = {
          x: next.x - Math.cos(angle) * 70,
          y: next.y - Math.sin(angle) * 52,
        };
      this.arrow(a, b, cols[i] + "9c", 2);
      if (this.state.particles && !this.state.reducedMotion) {
        const t = (this.time * 0.3 + i / 6) % 1;
        this.circle(lerp(a.x, b.x, t), lerp(a.y, b.y, t), 4, cols[i]);
      }
      this.rect(n.x - 131, n.y - 40, 262, 80, C.panel, cols[i] + "66", 15);
      this.text(n.label, n.x, n.y + 8, 19, C.ink, 500, "center");
    });
    this.text(
      `CYCLE ${this.state.cycle}`,
      606,
      589,
      15,
      this.caseData.color,
      400,
      "center",
      true,
    );
    this.text("N → N + 1", 606, 638, 43, C.ink, 500, "center");
    this.rect(1093, 277, 739, 631, C.panel, C.line, 21);
    this.panelHeading("COMMUNICATE · RECORD · CARRY FORWARD", 1128, 321);
    this.wrap(
      "The next cycle starts with memory.",
      1128,
      382,
      645,
      38,
      C.ink,
      1.35,
      500,
    );
    [
      ["Communicate", this.caseData.hapa.check],
      [
        "Retain lineage",
        "Store selected paths, impact summaries, critic decisions and cycle history.",
      ],
      ["Adapt the next observation model", this.caseData.next],
    ].forEach(([title, body], i) => {
      const y = 503 + i * 121;
      this.text(title, 1128, y, 22, [C.SOCIAL, C.PSYCHO, C.BIO][i], 500);
      this.wrap(body, 1128, y + 37, 635, 21, C.muted, 1.4, 400, 2);
    });
    this.caption =
      "Cycle N+1 carries the updated model into new collection, readiness and analysis. Research reporting remains a support flow.";
  }
  closing() {
    this.graph(1422, 565, 0.7, {
      active: this.model.selected,
      labels: false,
      rotate: this.state.reducedMotion ? 0 : this.p * 0.08,
    });
    this.pill(
      "PHOENIX · A COMPLEMENTARY REASONING LAYER",
      88,
      289,
      this.caseData.color,
      13,
    );
    this.text("Broader possibilities.", 88, 421, 74, C.ink, 500);
    this.text(
      "More personal questions.",
      88,
      516,
      74,
      this.caseData.color,
      500,
    );
    this.text("A next step that can adapt.", 88, 611, 65, C.ink, 500);
    this.wrap(this.caseData.question, 92, 713, 900, 30, C.muted, 1.4);
    this.line(91, 795, 767, 795, C.line);
    this.text(
      "Listen → ground → observe → learn → act → repeat",
      92,
      843,
      22,
      C.muted,
    );
    this.caption =
      "The demonstrated advantage is structural breadth, traceability and adaptation. Clinical superiority is not established by this film.";
  }
}
