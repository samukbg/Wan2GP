// Wan2GP downloads ffmpeg into <repo>/ffmpeg_bins and only adds it to PATH inside wgp.py. Put it on PATH
// here too, so manual runs (npm run selftest/render from a plain shell) and every child process
// (ffmpeg spawns, the framefields encoder) find it on servers without a system-wide ffmpeg.
import fs from "node:fs";
import path from "node:path";

const bins = path.resolve(import.meta.dirname, "..", "..", "ffmpeg_bins");
const exe = process.platform === "win32" ? "ffmpeg.exe" : "ffmpeg";
if (fs.existsSync(path.join(bins, exe))) {
	const key = Object.keys(process.env).find((k) => k.toUpperCase() === "PATH") || "PATH";
	const parts = (process.env[key] || "").split(path.delimiter).filter(Boolean);
	if (!parts.some((p) => path.resolve(p) === bins)) process.env[key] = [bins, ...parts].join(path.delimiter);
}
