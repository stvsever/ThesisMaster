import http from "node:http";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const port = Number(process.env.PORT || 4173);
const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ttf": "font/ttf",
  ".md": "text/plain; charset=utf-8",
  ".mp4": "video/mp4",
};
const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost");
    const requested = decodeURIComponent(url.pathname);
    const file = path.resolve(
      root,
      "." + (requested === "/" ? "/index.html" : requested),
    );
    if (!file.startsWith(root + path.sep)) {
      res.writeHead(403).end();
      return;
    }
    if (
      requested.split("/").some((p) => p.startsWith(".")) ||
      requested.includes("/node_modules/")
    ) {
      res.writeHead(403).end();
      return;
    }
    const info = await stat(file);
    if (!info.isFile()) {
      res.writeHead(404).end();
      return;
    }
    const headers = {
      "Content-Type": types[path.extname(file)] || "application/octet-stream",
      "Cache-Control": "no-cache",
      "X-Content-Type-Options": "nosniff",
      "Accept-Ranges": "bytes",
    };
    const range = req.headers.range;
    if (range) {
      const match = /^bytes=(\d*)-(\d*)$/.exec(range);
      if (!match) {
        res.writeHead(416, { "Content-Range": `bytes */${info.size}` }).end();
        return;
      }
      let start = match[1]
        ? Number(match[1])
        : Math.max(0, info.size - Number(match[2]));
      let end =
        match[2] && match[1]
          ? Math.min(Number(match[2]), info.size - 1)
          : info.size - 1;
      if (start > end || start >= info.size) {
        res.writeHead(416, { "Content-Range": `bytes */${info.size}` }).end();
        return;
      }
      res.writeHead(206, {
        ...headers,
        "Content-Length": end - start + 1,
        "Content-Range": `bytes ${start}-${end}/${info.size}`,
      });
      if (req.method === "HEAD") res.end();
      else createReadStream(file, { start, end }).pipe(res);
    } else {
      res.writeHead(200, { ...headers, "Content-Length": info.size });
      if (req.method === "HEAD") res.end();
      else createReadStream(file).pipe(res);
    }
  } catch {
    res.writeHead(404, { "Content-Type": "text/plain" }).end("Not found");
  }
});
server.listen(port, "127.0.0.1", () =>
  console.log(`PHOENIX studio: http://127.0.0.1:${port}`),
);
