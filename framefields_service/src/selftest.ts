// `npm run selftest`: renders every template from synthetic inputs, to check a server is ready
// (Node ≥22, WebGPU adapter, ffmpeg, native modules). Prints one line per template.
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const FONT_CANDIDATES = [
	"C:\\Windows\\Fonts\\arialbd.ttf", "C:\\Windows\\Fonts\\segoeuib.ttf",
	"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
	"/System/Library/Fonts/Supplemental/Arial Bold.ttf",
];
const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ff-selftest-"));
const font = FONT_CANDIDATES.find((f) => fs.existsSync(f));
if (!font) { console.error("No bold system font found for the self-test."); process.exit(1); }
fs.copyFileSync(font, path.join(dir, "font.ttf"));
const ff = (args: string[]) => { const r = spawnSync("ffmpeg", ["-v", "error", "-y", ...args], { cwd: dir }); if (r.status !== 0) throw new Error(`ffmpeg: ${r.stderr}`); };
ff(["-f", "lavfi", "-i", "testsrc2=size=720x1280:rate=30:duration=3", "-f", "lavfi", "-i", "sine=frequency=220:duration=3", "-shortest", "-pix_fmt", "yuv420p", "clip.mp4"]);
ff(["-f", "lavfi", "-i", "aevalsrc=0.9*sin(2*PI*55*t)*exp(-25*mod(t\\,0.5)):s=44100:d=6", "music.wav"]);
ff(["-f", "lavfi", "-i", "color=c=white:s=600x600", "-frames:v", "1", "product.png"]);

const jobs: Record<string, any> = {
	beat_grid: { template: "beat_grid", output: "beats.json", params: { audio: "music.wav" } },
	brand_grade: { template: "brand_grade", output: "grade.mp4", params: { video: "clip.mp4", primary: "#E6FF4F", secondary: "#1B3A6B", deflicker: true } },
	chart_card: { template: "chart_card", output: "chart.mp4", params: { font: "font.ttf", title: "Self-test chart", primary: "#E6FF4F", durationSec: 3, chart: { type: "bar", categories: ["A", "B", "C"], series: [{ name: "S", data: [3, 5, 2] }] } } },
	product_carousel: { template: "product_carousel", output: "carousel.mp4", params: { images: ["product.png"], font: "font.ttf", title: "Self-test", primary: "#E6FF4F", durationSec: 3 } },
	kinetic_captions: { template: "kinetic_captions", output: "captions.mp4", params: { video: "clip.mp4", font: "font.ttf", words: [{ text: "hello", start: 0.2, end: 0.8 }, { text: "world", start: 0.9, end: 1.5 }], music: "music.wav" } },
	hook_title: { template: "hook_title", output: "hook.mp4", params: { video: "clip.mp4", font: "font.ttf", title: "Self-test hook", durationSec: 2 } },
};
let failed = 0;
for (const [name, job] of Object.entries(jobs)) {
	const file = path.join(dir, `${name}.json`);
	fs.writeFileSync(file, JSON.stringify({ width: 720, height: 1280, fps: 30, ...job }));
	const r = spawnSync(process.execPath, ["--import", "tsx", path.join(import.meta.dirname, "render.ts"), file], { encoding: "utf8", cwd: path.join(import.meta.dirname, "..") });
	const line = (r.stdout || "").split("\n").find((l) => l.startsWith("FRAMEFIELDS_RESULT ")) || "";
	const res = line ? JSON.parse(line.slice(19)) : { ok: false, error: (r.stderr || "no result").slice(-300) };
	if (!res.ok) failed++;
	console.log(`${res.ok ? "OK  " : "FAIL"} ${name.padEnd(17)} ${res.ok ? `${res.ms} ms` : res.error}`);
}
fs.rmSync(dir, { recursive: true, force: true });
process.exit(failed ? 1 : 0);
