// 3D product showcase from product photos: rounded cards on a slowly turning ring (turntable when
// there is one photo), tilted toward the camera, with the title above and a soft floor glow.
import { Composition, Layer, Layer3D, LayerAnimation, TextAnimator } from "framefields";
import { FAMILY, fitText, registerFont, sec } from "../common.js";
import type { z } from "zod";
import type { ProductCarousel } from "../schema.js";

export async function productCarousel(job: z.infer<typeof ProductCarousel>, out: string) {
	const { width: W, height: H, fps, params: p } = job;
	const font = await registerFont(p.font);
	const frames = sec(p.durationSec, fps);
	// A ring needs at least 4 cards: repeat the photos (one photo = a turntable of the same product).
	const photos = [...p.images];
	while (photos.length < 4) photos.push(...p.images);
	const n = Math.min(8, photos.length);
	const cardW = Math.round(W * 0.5);
	const cardH = Math.round(cardW * 1.25);
	const radius = Math.round((cardW * 0.62) / Math.tan(Math.PI / n));
	const items = photos.slice(0, n).map((src, i) =>
		Layer.box({
			id: `card-${i}`,
			width: cardW,
			height: cardH,
			background: "#FFFFFF",
			borderRadius: Math.round(cardW * 0.06),
			overflow: "hidden",
			padding: Math.round(cardW * 0.05),
			children: [Layer.image(src, { width: cardW - Math.round(cardW * 0.1), height: cardH - Math.round(cardW * 0.1), fit: "contain" })],
		}),
	);

	const comp = new Composition({ width: W, height: H, fps, durationMs: (frames / fps) * 1000, backgroundColor: p.background, fonts: [font] });
	// Floor glow in the brand colour.
	comp.add(Layer.shape("ellipse", { position: "absolute", x: Math.round(W * 0.1), y: Math.round(H * 0.72), width: Math.round(W * 0.8), height: Math.round(H * 0.08), fillColor: p.primary, opacity: 0.18 } as any));
	const ring = Layer3D.carousel({ id: "ring", radius, items, itemWidth: cardW, itemHeight: cardH, x: W / 2, y: Math.round(H * 0.52), rotateX: -8, twoSided: true });
	const turns = Math.max(0.35, Math.min(1, p.durationSec / 8));
	ring.animate(LayerAnimation.create().fromTo("rotateY", 0, -360 * turns, { start: 0, end: frames, ease: "sine.inOut" }).fadeIn(0, 12, "power2.out"));
	comp.add(ring);
	if (p.title) {
		const t = fitText(font, p.title.toUpperCase(), Math.round(W * 0.8), 2, Math.round(W * 0.1), Math.round(W * 0.055), H * 0.16, 1.05);
		comp.add(Layer.text(t.lines.join("\n"), {
			position: "absolute", x: Math.round(W * 0.06), y: Math.round(H * 0.09), width: Math.round(W * 0.88), height: Math.round(t.size * 1.05 * t.lines.length + t.size * 0.3),
			fontFamily: FAMILY, fontSize: t.size, fontWeight: 900, lineHeight: 1.05, fill: p.textColor, align: "center",
			animators: [TextAnimator.wordPop({ scale: 0.7, y: 24, easing: "back.out(1.5)" })],
		}).animate(LayerAnimation.create().kineticSweep(-1, 1, 0, 18, "power3.out")));
	}
	const r = await comp.renderVideo({ outputPath: out, quality: "high" });
	await r.cleanup?.();
	return { output: out, frames, cards: n };
}
