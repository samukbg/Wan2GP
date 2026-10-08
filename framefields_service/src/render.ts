// Entry point: node --import tsx src/render.ts <job.json>
// The job file is written by Wan2GP (workflow_endpoints.py) into a fresh job directory together with the
// downloaded inputs. Only the fixed templates below can run — no code, expressions or URLs come in.
import fs from "node:fs";
import path from "node:path";
import { Job } from "./schema.js";
import { insideJobDir } from "./paths.js";

const jobFile = process.argv[2];
if (!jobFile) { console.error("usage: render.ts <job.json>"); process.exit(2); }

const done = (result: unknown, code = 0) => {
	// Last stdout line is the machine-readable result for the Python caller.
	process.stdout.write(`\nFRAMEFIELDS_RESULT ${JSON.stringify(result)}\n`);
	process.exit(code);
};

try {
	const jobDir = path.dirname(fs.realpathSync(jobFile));
	const raw = JSON.parse(fs.readFileSync(jobFile, "utf8"));
	const output = path.resolve(jobDir, String(raw.output || ""));
	if (!output.startsWith(jobDir + path.sep)) throw new Error("output must be inside the job directory");
	const { output: _out, ...spec } = raw;
	const job = Job.parse(spec);
	// Resolve every media path inside the job directory (throws on URLs and escapes).
	const p: any = job.params;
	for (const k of ["video", "font", "audio", "music"]) if (typeof p[k] === "string") p[k] = insideJobDir(jobDir, p[k]);
	if (Array.isArray(p.images)) p.images = p.images.map((x: string) => insideJobDir(jobDir, x));
	if (p.model && typeof p.model === "string") p.model = insideJobDir(jobDir, p.model);
	if (!process.env.FRAMEFIELDS_MODELS_DIR) process.env.FRAMEFIELDS_MODELS_DIR = path.resolve(import.meta.dirname, "../models");

	const t0 = Date.now();
	let result: any;
	switch (job.template) {
		case "hook_title": result = await (await import("./templates/hookTitle.js")).hookTitle(job, output); break;
		case "brand_grade": result = await (await import("./templates/brandGrade.js")).brandGrade(job, output, jobDir); break;
		case "chart_card": result = await (await import("./templates/chartCard.js")).chartCard(job, output); break;
		case "product_carousel": result = await (await import("./templates/productCarousel.js")).productCarousel(job, output); break;
		case "kinetic_captions": result = await (await import("./templates/kineticCaptions.js")).kineticCaptions(job, output); break;
		case "beat_grid": {
			const { beatGrid } = await import("./beats.js");
			const grid = await beatGrid(p.audio, job.fps);
			fs.writeFileSync(output, JSON.stringify(grid));
			result = { output, bpm: grid.bpm, beats: grid.beats.length };
			break;
		}
	}
	done({ ok: true, template: job.template, ms: Date.now() - t0, ...result });
} catch (e: any) {
	done({ ok: false, error: String(e?.message || e).slice(0, 800) }, 1);
}
