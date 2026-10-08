// Text behind the presenter: background plate → hook title → person cutout (same plate, matte alpha).
// The title sits above the head and the head overlaps its last line. When the framing is too tight
// for that, the plate moves down (≤22% H) and a blurred copy of the clip fills the band above.
import sharp from "sharp";
import { Blur, Composition, HeadlessMediaRenderer, Layer, LayerAnimation, Modulate, TextAnimator } from "framefields";
import { FAMILY, fitText, registerFont, sec } from "../common.js";
import type { z } from "zod";
import type { HookTitle } from "../schema.js";

const KEY = "#00FF00";

/** Top edge of the main subject (px), from the person cutout over a pure-green background. */
async function subjectTop(video: string, W: number, H: number, fps: number): Promise<number | null> {
	const probe = new Composition({ width: W, height: H, fps, durationMs: 1000, backgroundColor: KEY });
	probe.add(Layer.video(video, { position: "absolute", x: 0, y: 0, width: W, height: H, fit: "cover", muted: true })
		.withVision({ mode: "matte", enableSegmentation: true, variant: "s", confidence: 0.25, keyBackground: true }));
	const renderer = new HeadlessMediaRenderer();
	let png: Buffer | null = null;
	// Vision reads the previous frame: render a few in order and keep the last.
	for (let f = 0; f <= 4; f++) png = await probe.renderFrame({ frame: f, renderer });
	if (!png) return null;
	const { data, info } = await sharp(png).removeAlpha().raw().toBuffer({ resolveWithObject: true });
	const { width: w, height: h, channels: c } = info;
	for (let y = 0; y < h; y++) {
		let hits = 0;
		for (let x = 0; x < w; x++) {
			const i = (y * w + x) * c;
			const r = data[i], g = data[i + 1], b = data[i + 2];
			if (!(g > 200 && r < 80 && b < 80)) hits++;
		}
		if (hits > w * 0.08) return Math.round((y * H) / h); // the head's width, not a hair bun's tip
	}
	return null;
}

export async function hookTitle(job: z.infer<typeof HookTitle>, out: string) {
	const { width: W, height: H, fps, params: p } = job;
	const font = await registerFont(p.font);
	const frames = sec(p.durationSec, fps);
	const title = p.title.toUpperCase();
	const boxW = Math.round(W * 0.9);
	// Measure against 88% of the box: the GPU text engine sets display faces a little wider than fontkit.
	const { size, lines } = fitText(font, title, Math.round(boxW * 0.82), 4, Math.round(W * 0.2), Math.round(W * 0.075), H * 0.32);
	const lineH = Math.round(size * 1.05);
	const blockH = lineH * lines.length;
	const margin = Math.round(H * 0.045);

	let shift = 0;
	let titleY = margin;
	let behind = true;
	if (p.position === "top") {
		const top = await subjectTop(p.video, W, H, fps).catch(() => null);
		if (top === null) behind = false; // nobody to put the text behind
		else {
			// The head should cover about 40% of the last line.
			const wantTop = margin + blockH - Math.round(lineH * 0.4);
			shift = Math.max(0, Math.min(Math.round(H * 0.22), wantTop - top));
			titleY = Math.min(margin, top + shift - blockH + Math.round(lineH * 0.4));
			titleY = Math.max(Math.round(H * 0.02), titleY);
		}
	} else titleY = Math.round(H / 2 - blockH / 2);

	const comp = new Composition({ width: W, height: H, fps, durationMs: (frames / fps) * 1000, backgroundColor: "#000000", fonts: [font] });
	if (shift > 0) {
		comp.add(Layer.video(p.video, { position: "absolute", x: 0, y: 0, width: W, height: H, fit: "cover", muted: true })
			.withEffect(new Blur({ strength: 40, blurType: "Gaussian" }))
			.withEffect(new Modulate({ brightness: 0.72, saturation: 0.85 })));
	}
	comp.add(Layer.video(p.video, { position: "absolute", x: 0, y: shift, width: W, height: H, fit: "cover", muted: false }));
	comp.add(Layer.text(lines.join("\n"), {
		id: "hook-title",
		position: "absolute",
		x: Math.round((W - boxW) / 2),
		y: titleY,
		width: boxW,
		height: blockH + Math.round(size * 0.2),
		fontFamily: FAMILY,
		fontSize: size,
		fontWeight: 900,
		lineHeight: 1.05,
		fill: p.color,
		align: "center",
		animators: [TextAnimator.wordPop({ scale: 0.6, y: 30, easing: "back.out(1.6)" })],
	}).animate(
		LayerAnimation.create()
			.kineticSweep(-1, 1, 0, Math.min(frames, Math.round(fps * 0.7)), "power3.out")
			.keys("scale", [[0, 1.08], [Math.min(frames, Math.round(fps * 0.6)), 1, "power2.out"]]),
	));
	if (behind) {
		comp.add(Layer.video(p.video, { position: "absolute", x: 0, y: shift, width: W, height: H, fit: "cover", muted: true })
			.withVision({ mode: "matte", enableSegmentation: true, variant: "s", confidence: 0.25, featherRadius: 0.04, keyBackground: true }));
	}
	const r = await comp.renderVideo({ outputPath: out, quality: "high" });
	await r.cleanup?.();
	return { output: out, frames, behind, shift };
}
