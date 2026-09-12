#!/usr/bin/env node
import { createCanvas, GlobalFonts } from "@napi-rs/canvas";
import { readFile, writeFile, mkdir, rename, unlink } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { Renderer } from "../app/renderer.js";
import { DEFAULTS, SHOTS, DURATION, formatTime } from "../app/model.js";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const value = (key, fallback) =>
  args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const width = Number(value("--width", "1920")),
  height = (width * 9) / 16,
  fps = Number(value("--fps", "24"));
if (
  !Number.isInteger(width) ||
  !Number.isInteger(height) ||
  width % 2 ||
  height % 2 ||
  fps < 1 ||
  fps > 60
)
  throw new Error("Use an even 16:9 resolution and 1 to 60 fps.");
for (const style of ["Regular", "Medium", "SemiBold", "Bold"])
  GlobalFonts.registerFromPath(
    path.join(root, `assets/fonts/Manrope-${style}.ttf`),
    "Manrope",
  );
GlobalFonts.registerFromPath(
  path.join(root, "assets/fonts/IBMPlexMono-Regular.ttf"),
  "IBM Plex Mono",
);
const snapshot = JSON.parse(
  await readFile(path.join(root, "data/snapshot.json"), "utf8"),
);
const session = args.includes("--session")
  ? JSON.parse(await readFile(path.resolve(value("--session")), "utf8"))
  : null;
const sessionState = session?.state || {};
const selectedCase = value("--case", sessionState.caseId || "all");
const cases =
  selectedCase === "all"
    ? snapshot.cases
    : snapshot.cases.filter((c) => c.id === selectedCase);
if (!cases.length)
  throw new Error(
    "Unknown case. Use lana, maarten, maya, noor, elias, or all.",
  );
const posterOnly = args.includes("--posters");
const from = Number(value("--from", "0"));
const to = Number(value("--to", String(DURATION)));
const seconds = to - from;
if (from < 0 || to > DURATION || seconds <= 0)
  throw new Error("Invalid film interval.");
const out = path.join(root, "renders");
for (const d of ["mp4", "posters", "contact-sheets", "qa"])
  await mkdir(path.join(out, d), { recursive: true });
const canvas = createCanvas(width, height);
const renderer = new Renderer(canvas, snapshot);
const manifest = {
  renderer: "Shared Canvas 2D scene engine",
  width,
  height,
  fps,
  duration: seconds,
  from,
  to,
  audio: "silent",
  cases: [],
  notice: snapshot.notice,
};
for (const d of cases) {
  const state = { ...DEFAULTS, ...sessionState, caseId: d.id };
  const customTag = session ? "_custom" : "";
  const posterTime = SHOTS.find((s) => s.id === "search").start + 10;
  renderer.draw({ ...state, time: posterTime });
  await writeFile(
    path.join(out, "posters", `${d.id}_phoenix_full_workflow${customTag}.png`),
    await canvas.encode("png"),
  );
  const sheet = createCanvas(1600, 1038),
    ctx = sheet.getContext("2d");
  ctx.fillStyle = "#0b101a";
  ctx.fillRect(0, 0, 1600, 1038);
  for (const s of SHOTS) {
    renderer.draw({ ...state, time: s.start + s.duration * 0.6 });
    const col = s.index % 4,
      row = Math.floor(s.index / 4);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.drawImage(canvas, col * 400, row * 255, 400, 225);
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.font = "12px Manrope";
    ctx.fillStyle = "#c8cddd";
    ctx.fillText(
      `${String(s.index + 1).padStart(2, "0")}  ${s.title}  /  ${formatTime(s.start)}`,
      col * 400 + 14,
      row * 255 + 244,
    );
    if (args.includes("--qa"))
      await writeFile(
        path.join(
          out,
          "qa",
          `${d.id}_${String(s.index).padStart(2, "0")}_${s.id}.png`,
        ),
        await canvas.encode("png"),
      );
  }
  await writeFile(
    path.join(out, "contact-sheets", `${d.id}_storyboard${customTag}.jpg`),
    await sheet.encode("jpeg", 88),
  );
  if (posterOnly) {
    console.log(
      `${d.id}: poster and all ${SHOTS.length} storyboard frames ready`,
    );
    continue;
  }
  const suffix =
    from === 0 && to === DURATION
      ? `_phoenix_full_workflow${customTag}`
      : `_preview${customTag}`;
  const destination = path.join(out, "mp4", `${d.id}${suffix}.mp4`),
    temporary = destination.replace(".mp4", ".partial.mp4");
  const ffmpeg = spawn(
    "ffmpeg",
    [
      "-y",
      "-hide_banner",
      "-loglevel",
      "error",
      "-f",
      "rawvideo",
      "-pix_fmt",
      "rgba",
      "-s",
      `${width}x${height}`,
      "-r",
      String(fps),
      "-i",
      "pipe:0",
      "-an",
      "-c:v",
      "libx264",
      "-preset",
      "fast",
      "-crf",
      "20",
      "-pix_fmt",
      "yuv420p",
      "-movflags",
      "+faststart",
      "-metadata",
      `title=PHOENIX | ${d.name} | Complete workflow`,
      "-metadata",
      "artist=Stijn Van Severen",
      "-metadata",
      "comment=Fictional case. Simulated evidence. Ontology-grounded explanatory demonstration.",
      temporary,
    ],
    { stdio: ["pipe", "ignore", "pipe"] },
  );
  let ffError = "";
  ffmpeg.stderr.on("data", (chunk) => (ffError += chunk.toString()));
  let closed = false;
  let ffExit = null;
  const done = new Promise((resolve, reject) => {
    ffmpeg.on("error", reject);
    ffmpeg.on("close", (code) => {
      closed = true;
      ffExit = code;
      code === 0 ? resolve() : reject(new Error(`ffmpeg ${code}: ${ffError}`));
    });
  });
  // Attach immediately so an early encoder failure cannot become an unhandled rejection.
  done.catch(() => {});
  ffmpeg.stdin.on("error", () => {});
  const frames = Math.round(seconds * fps),
    started = Date.now();
  console.log(
    `${d.id}: rendering ${frames} frames at ${width} × ${height}, ${fps} fps`,
  );
  try {
    for (let frame = 0; frame < frames; frame++) {
      if (closed)
        throw new Error(`Encoder closed early (${ffExit}): ${ffError}`);
      renderer.draw({ ...state, time: from + frame / fps });
      if (!ffmpeg.stdin.write(canvas.data())) await once(ffmpeg.stdin, "drain");
      if (frame % (fps * 10) === 0) {
        console.log(
          `${d.id}: ${Math.round((frame / frames) * 100)}% · ${formatTime(frame / fps)} / ${formatTime(seconds)} · ${Math.round((Date.now() - started) / 1000)}s elapsed`,
        );
      }
    }
    ffmpeg.stdin.end();
    await done;
    await rename(temporary, destination);
  } catch (error) {
    ffmpeg.kill("SIGTERM");
    await unlink(temporary).catch(() => {});
    throw error;
  }
  manifest.cases.push({
    id: d.id,
    file: path.relative(root, destination),
    frames,
    renderSeconds: Math.round((Date.now() - started) / 1000),
  });
  console.log(
    `${d.id}: complete in ${Math.round((Date.now() - started) / 1000)} seconds`,
  );
}
if (!posterOnly && from === 0 && to === DURATION)
  await writeFile(
    path.join(
      out,
      session
        ? `manifest-${cases[0].id}-custom.json`
        : cases.length === 5
          ? "manifest.json"
          : `manifest-${cases[0].id}.json`,
    ),
    JSON.stringify(manifest, null, 2) + "\n",
  );
