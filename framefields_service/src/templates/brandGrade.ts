// One brand look on every AI clip: temporal de-flicker → brand LUT → subtle grain + vignette.
import fs from "node:fs";
import path from "node:path";
import { ApplyLUT, Composition, FilmGrain, Layer, TemporalDeflicker, Vignette } from "framefields";
import { probeDuration, sec } from "../common.js";
import { writeBrandLut } from "../lut.js";
import type { z } from "zod";
import type { BrandGrade } from "../schema.js";

export async function brandGrade(job: z.infer<typeof BrandGrade>, out: string, jobDir: string) {
	const { width: W, height: H, fps, params: p } = job;
	const duration = p.durationSec ?? (await probeDuration(p.video));
	const frames = sec(duration, fps);
	const lut = path.join(jobDir, "brand.cube");
	writeBrandLut(lut, p.primary, p.secondary, p.intensity);

	const comp = new Composition({ width: W, height: H, fps, durationMs: (frames / fps) * 1000, backgroundColor: "#000000" });
	let clip = Layer.video(p.video, { position: "absolute", x: 0, y: 0, width: W, height: H, fit: "cover", muted: false });
	if (p.deflicker) clip = clip.withEffect(new TemporalDeflicker({}));
	clip = clip
		// framefields loads LUTs with fetch(), which can't open a file path (the LUT was silently skipped).
		.withEffect(new ApplyLUT({ lutUrl: `data:text/plain;base64,${fs.readFileSync(lut).toString("base64")}`, intensity: 1 }))
		.withEffect(new Vignette({ strength: 0.12 + 0.18 * p.intensity, radius: 0.9 }));
	comp.add(clip);
	if (p.grain > 0) comp.apply(new FilmGrain({ strength: 0.02 + 0.05 * p.grain, size: 1.4, monochrome: true, animated: true }));
	const r = await comp.renderVideo({ outputPath: out, quality: "high" });
	await r.cleanup?.();
	return { output: out, frames };
}
