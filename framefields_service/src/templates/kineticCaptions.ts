// Kinetic captions: short phrases pop in on their first word, the spoken word lights up in the brand
// colour, and with music the phrase pulses on the beat (onset envelope from beats.ts).
import { Composition, Layer, LayerAnimation, Signal, TextAnimator } from "framefields";
import { FAMILY, fitText, probeDuration, registerFont, sec, textWidth } from "../common.js";
import { beatGrid } from "../beats.js";
import type { z } from "zod";
import type { KineticCaptions } from "../schema.js";

type Word = { text: string; start: number; end: number };

/** Groups words into phrases: ≤3 words, ≤ maxChars, broken at sentence ends and long pauses. */
function phrases(words: Word[], maxChars = 18): Word[][] {
	const out: Word[][] = [];
	let cur: Word[] = [];
	for (const w of words) {
		const len = cur.reduce((n, x) => n + x.text.length + 1, 0) + w.text.length;
		const gap = cur.length ? w.start - cur[cur.length - 1].end : 0;
		if (cur.length && (cur.length >= 3 || len > maxChars || gap > 0.45)) { out.push(cur); cur = []; }
		cur.push(w);
		if (/[.!?…]$/.test(w.text)) { out.push(cur); cur = []; }
	}
	if (cur.length) out.push(cur);
	return out;
}

export async function kineticCaptions(job: z.infer<typeof KineticCaptions>, out: string) {
	const { width: W, height: H, fps, params: p } = job;
	const font = await registerFont(p.font);
	const duration = p.durationSec ?? (await probeDuration(p.video));
	const frames = sec(duration, fps);
	const words = p.words.filter((w) => w.end > w.start && w.start < duration).map((w) => ({ ...w, text: w.text.toUpperCase() }));
	const boxW = Math.round(W * 0.96);
	const baseSize = Math.round(W * 0.085);
	const y = p.position === "bottom" ? Math.round(H * 0.7) : Math.round(H * 0.45);

	const comp = new Composition({ width: W, height: H, fps, durationMs: (frames / fps) * 1000, backgroundColor: "#000000", fonts: [font] });
	comp.add(Layer.video(p.video, { position: "absolute", x: 0, y: 0, width: W, height: H, fit: "cover", muted: false }));
	const pulse = p.music ? Signal.fromArray((await beatGrid(p.music, fps)).envelope, fps) : null;

	const groups = phrases(words);
	groups.forEach((g, gi) => {
		const start = Math.round(g[0].start * fps);
		const next = groups[gi + 1];
		const endSec = Math.min(duration, next ? Math.max(g[g.length - 1].end, next[0].start) : g[g.length - 1].end + 0.4);
		const dur = Math.max(4, Math.round(endSec * fps) - start);
		const phraseText = g.map((w) => w.text).join(" ");
		const { size, lines } = fitText(font, phraseText, Math.round(W * 0.72), 2, baseSize, Math.round(W * 0.05), H * 0.2, 1.05);
		const blockH = Math.round(size * 1.05 * lines.length + size * 0.25);
		// Pill sized to the text (not the whole box), centred.
		const padX = Math.round(size * 0.32);
		// The GPU text engine sets display faces ~10-15% wider than fontkit: pill gets that margin, and the
		// text box spans the full width so a line can never be forced to wrap.
		const textW = Math.max(...lines.map((l) => textWidth(font, l, size))) * 1.16;
		const pillW = Math.min(W - 16, Math.round(textW + padX * 2));
		const pillH = blockH + Math.round(size * 0.3);
		const anim = LayerAnimation.create().kineticSweep(-1, 1, 0, Math.min(dur, 6), "power2.out").keys("scale", [[0, 0.86], [5, 1.05, "back.out(2)"], [9, 1, "power2.out"]]);
		if (pulse) anim.signal("scale", pulse, { multiplier: 0.05, offset: 1, smoothing: 2 });
		const text = lines.join("\n");
		const base = { position: "absolute" as const, x: Math.round((W - boxW) / 2), y, width: boxW, height: pillH, fontFamily: FAMILY, fontSize: size, fontWeight: 900, lineHeight: 1.05, align: "center" as const, verticalAlign: "middle" as const, startFrame: start, durationFrames: dur };
		comp.add(Layer.box({ id: `cap-${gi}-pill`, position: "absolute", x: Math.round((W - pillW) / 2), y, width: pillW, height: pillH, background: "#000000A6", borderRadius: Math.round(size * 0.28), startFrame: start, durationFrames: dur })
			.animate(LayerAnimation.create().keys("opacity", [[0, 0], [3, 1]]).keys("scale", [[0, 0.9], [6, 1, "back.out(1.6)"]])));
		comp.add(Layer.text(text, { ...base, id: `cap-${gi}`, fill: p.textColor,
			animators: [TextAnimator.wordPop({ scale: 0.75, y: 14, easing: "back.out(1.7)" })] }).animate(anim));
		// Words light up as they are spoken: an identical layer whose colour sweep follows each word's start.
		// (Span colours don't render in framefields 2.0.8; the sweep offset maps linearly to the share of
		// characters reached: offset ≈ -0.3 + 1.1 × share.)
		const total = text.length;
		let cursor = 0;
		let prev = -1;
		const lit = LayerAnimation.create().keys("opacity", [[0, 0], [5, 1]]);
		g.forEach((w) => {
			const at = text.indexOf(w.text, cursor);
			cursor = at >= 0 ? at + w.text.length : cursor + w.text.length + 1;
			const target = Math.min(1, -0.3 + (1.1 * cursor) / total);
			const ws = Math.max(0, Math.round(w.start * fps) - start);
			lit.kineticSweep(prev, target, ws, Math.min(dur, ws + 3), "power1.out");
			prev = target;
		});
		if (pulse) lit.signal("scale", pulse, { multiplier: 0.05, offset: 1, smoothing: 2 });
		comp.add(Layer.text(text, { ...base, id: `cap-${gi}-lit`, fill: p.highlight,
			animators: [TextAnimator.colorSweep({ color: p.textColor, unit: "character" })] }).animate(lit));
	});
	const r = await comp.renderVideo({ outputPath: out, quality: "high" });
	await r.cleanup?.();
	return { output: out, frames, phrases: groups.length };
}
