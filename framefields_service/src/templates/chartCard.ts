// Animated chart card for Fast Facts / Top-list: title, subtitle, then the chart draws itself on.
import { Composition, Layer, LayerAnimation, TextAnimator } from "framefields";
import { FAMILY, fitText, registerFont, sec } from "../common.js";
import type { z } from "zod";
import type { ChartCard } from "../schema.js";

export async function chartCard(job: z.infer<typeof ChartCard>, out: string) {
	const { width: W, height: H, fps, params: p } = job;
	const font = await registerFont(p.font);
	const frames = sec(p.durationSec, fps);
	const pad = Math.round(W * 0.07);
	const innerW = W - pad * 2;
	const title = fitText(font, p.title, Math.round(innerW * 0.9), 3, Math.round(W * 0.095), Math.round(W * 0.05), H * 0.2, 1.08);
	const titleH = Math.round(title.size * 1.08 * title.lines.length);
	const subSize = Math.round(W * 0.038);
	const sub = p.subtitle ? fitText(font, p.subtitle, Math.round(innerW * 0.9), 2, subSize, Math.round(W * 0.03), H * 0.08, 1.2) : null;
	const subH = sub ? Math.round(sub.size * 1.2 * sub.lines.length) : 0;
	const titleY = Math.round(H * 0.12);
	const chartY = titleY + titleH + subH + Math.round(H * 0.05);
	const chartH = Math.min(Math.round(H * 0.55), H - chartY - Math.round(H * 0.08));
	const colors = [p.primary, p.secondary || "#7A86A8", "#FFFFFF"];

	const comp = new Composition({ width: W, height: H, fps, durationMs: (frames / fps) * 1000, backgroundColor: p.background, fonts: [font] });
	// Accent bar.
	comp.add(Layer.box({ position: "absolute", x: pad, y: titleY - Math.round(H * 0.025), width: Math.round(W * 0.14), height: Math.round(H * 0.006), background: p.primary, borderRadius: 4 })
		.animate(LayerAnimation.create().keys("scaleX", [[0, 0], [12, 1, "expo.out"]])));
	comp.add(Layer.text(title.lines.join("\n"), {
		position: "absolute", x: pad, y: titleY, width: innerW, height: titleH + Math.round(title.size * 0.2),
		fontFamily: FAMILY, fontSize: title.size, fontWeight: 900, lineHeight: 1.08, fill: p.textColor, align: "start",
		animators: [TextAnimator.wordPop({ scale: 0.7, y: 24, easing: "back.out(1.5)" })],
	}).animate(LayerAnimation.create().kineticSweep(-1, 1, 0, 18, "power3.out")));
	if (sub) {
		comp.add(Layer.text(sub.lines.join("\n"), {
			position: "absolute", x: pad, y: titleY + titleH + Math.round(H * 0.01), width: innerW, height: subH + 10,
			fontFamily: FAMILY, fontSize: sub.size, lineHeight: 1.2, fill: p.textColor, align: "start", opacity: 0.75,
		}).animate(LayerAnimation.create().fadeIn(8, 22, "power2.out")));
	}
	const isPie = p.chart.type === "donut";
	comp.add(Layer.chart({
		type: p.chart.type,
		width: innerW,
		height: chartH,
		categories: p.chart.categories,
		series: p.chart.series.map((s, i) => ({ name: s.name, data: s.data, color: colors[i % colors.length] })),
		...(isPie ? {} : { yAxis: { grid: true, ...(p.chart.valueFormat ? { format: p.chart.valueFormat } : {}) } }),
		valueLabels: true,
		legend: p.chart.series.length > 1 || isPie,
		fontFamily: FAMILY,
		fontSize: Math.round(W * 0.03),
		colors,
		animate: { start: 14, duration: Math.max(18, Math.min(45, frames - 30)), stagger: 4, ease: "power3.out" },
	} as any, { position: "absolute", x: pad, y: chartY }));
	const r = await comp.renderVideo({ outputPath: out, quality: "high" });
	await r.cleanup?.();
	return { output: out, frames };
}
