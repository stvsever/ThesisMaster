#!/usr/bin/env node
import { readFile, stat, writeFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DURATION, SHOTS } from "../app/model.js";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const snapshot = JSON.parse(
  await readFile(path.join(root, "data/snapshot.json"), "utf8"),
);
const results = [];
for (const c of snapshot.cases) {
  const file = path.join(
    root,
    "renders/mp4",
    `${c.id}_phoenix_full_workflow.mp4`,
  );
  const info = JSON.parse(
    execFileSync(
      "ffprobe",
      [
        "-v",
        "error",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames:format=duration,size,tags",
        "-of",
        "json",
        file,
      ],
      { encoding: "utf8" },
    ),
  );
  const video = info.streams.find((s) => s.codec_name === "h264");
  if (
    !video ||
    video.width !== 1920 ||
    video.height !== 1080 ||
    video.pix_fmt !== "yuv420p" ||
    video.r_frame_rate !== "24/1" ||
    Math.abs(Number(info.format.duration) - DURATION) > 0.05 ||
    Number(video.nb_frames) !== DURATION * 24
  )
    throw new Error(`Unexpected media properties: ${c.id}`);
  if (Number(info.format.size) > 95 * 1024 * 1024)
    throw new Error(`Film is too large for ordinary GitHub storage: ${c.id}`);
  execFileSync("ffmpeg", ["-v", "error", "-i", file, "-f", "null", "-"], {
    stdio: ["ignore", "ignore", "pipe"],
  });
  const poster = await stat(
    path.join(root, "renders/posters", `${c.id}_phoenix_full_workflow.png`),
  );
  const sheet = await stat(
    path.join(root, "renders/contact-sheets", `${c.id}_storyboard.jpg`),
  );
  if (!poster.size || !sheet.size)
    throw new Error(`Missing visual companion: ${c.id}`);
  const row = {
    case: c.id,
    width: video.width,
    height: video.height,
    fps: 24,
    duration: Number(info.format.duration),
    frames: Number(video.nb_frames),
    megabytes: Number((Number(info.format.size) / 1024 / 1024).toFixed(2)),
    fullDecode: "passed",
    storyboardFrames: SHOTS.length,
  };
  results.push(row);
  console.log(
    `${c.id}: verified ${row.frames} frames, ${row.duration}s, ${row.megabytes} MiB, full decode passed`,
  );
}
await writeFile(
  path.join(root, "renders/verification.json"),
  JSON.stringify(
    {
      verification: "ffprobe metadata and complete ffmpeg decode",
      films: results,
    },
    null,
    2,
  ) + "\n",
);
