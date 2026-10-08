import { spawn } from "node:child_process";
import path from "node:path";
import { FontManager } from "framefields";
import * as fontkit from "fontkit";

export const FAMILY = "SpreadOutDisplay";

/** Registers the job's display font (passed in by SpreadOut: no system fonts exist headlessly). */
export async function registerFont(fontPath: string): Promise<string> {
	await FontManager.register({ family: FAMILY, source: fontPath });
	return fontPath;
}

export function probeDuration(file: string): Promise<number> {
	return new Promise((resolve, reject) => {
		const p = spawn("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", file], { shell: false });
		let out = "";
		p.stdout.on("data", (c: Buffer) => { out += c.toString(); });
		p.on("error", reject);
		p.on("close", () => {
			const d = Number.parseFloat(out.trim());
			Number.isFinite(d) && d > 0 ? resolve(d) : reject(new Error(`Could not read duration of ${path.basename(file)}`));
		});
	});
}

export const sec = (s: number, fps: number) => Math.max(1, Math.round(s * fps));

const fonts = new Map<string, any>();
function openFont(fontPath: string): any {
	if (!fonts.has(fontPath)) fonts.set(fontPath, (fontkit as any).openSync(fontPath));
	return fonts.get(fontPath);
}

/** Advance width of `text` at `size` px, from the font's real glyph metrics. */
export function textWidth(fontPath: string, text: string, size: number, letterSpacing = 0): number {
	const f = openFont(fontPath);
	const run = f.layout(text);
	return (run.advanceWidth / f.unitsPerEm) * size + letterSpacing * Math.max(0, text.length - 1);
}

/** Greedy word wrap into lines no wider than `width`. */
export function wrapLines(fontPath: string, text: string, size: number, width: number): string[] {
	const lines: string[] = [];
	let cur = "";
	for (const w of text.split(/\s+/).filter(Boolean)) {
		const next = cur ? `${cur} ${w}` : w;
		if (cur && textWidth(fontPath, next, size) > width) { lines.push(cur); cur = w; } else cur = next;
	}
	if (cur) lines.push(cur);
	return lines;
}

/** Largest size (≤ max) at which `text` wraps into ≤ maxLines lines that all fit `width`. */
export function fitText(fontPath: string, text: string, width: number, maxLines: number, max: number, min = 28, maxHeight = Infinity, lineHeight = 1.05): { size: number; lines: string[] } {
	for (let size = max; size >= min; size -= 2) {
		const lines = wrapLines(fontPath, text, size, width);
		if (lines.length <= maxLines && lines.length * size * lineHeight <= maxHeight && lines.every((l) => textWidth(fontPath, l, size) <= width)) return { size, lines };
	}
	return { size: min, lines: wrapLines(fontPath, text, min, width) };
}

/** "#RRGGBB" → [r, g, b] in 0..1 */
export function rgb(hex: string): [number, number, number] {
	const n = Number.parseInt(hex.slice(1), 16);
	return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
