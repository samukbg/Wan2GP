// Job contract between Wan2GP (Python) and the renderer. Every field is validated; anything unknown
// is rejected (.strict). Media are LOCAL paths inside the job directory — never URLs (see paths.ts).
import { z } from "zod";

const hex = z.string().regex(/^#[0-9a-fA-F]{6}$/, "hex colour like #RRGGBB");
const path = z.string().min(1).max(500);
const text = (max: number) => z.string().min(1).max(max).transform((s) => s.replace(/[\u0000-\u001f\u007f]/g, " ").trim());
const canvas = {
	width: z.number().int().min(240).max(2160).default(720),
	height: z.number().int().min(240).max(3840).default(1280),
	fps: z.number().int().min(12).max(60).default(30),
};

export const HookTitle = z.object({
	template: z.literal("hook_title"),
	...canvas,
	params: z.object({
		video: path,
		font: path,
		title: text(70),
		color: hex.default("#FFFFFF"),
		accent: hex.optional(),
		durationSec: z.number().min(0.5).max(20),
		position: z.enum(["top", "center"]).default("top"),
	}).strict(),
}).strict();

export const BrandGrade = z.object({
	template: z.literal("brand_grade"),
	...canvas,
	params: z.object({
		video: path,
		primary: hex,
		secondary: hex.optional(),
		intensity: z.number().min(0).max(1).default(0.35),
		grain: z.number().min(0).max(1).default(0.35),
		deflicker: z.boolean().default(true),
		durationSec: z.number().min(0.5).max(120).optional(),
	}).strict(),
}).strict();

export const ChartCard = z.object({
	template: z.literal("chart_card"),
	...canvas,
	params: z.object({
		font: path,
		title: text(80),
		subtitle: text(120).optional(),
		chart: z.object({
			type: z.enum(["bar", "line", "area", "donut"]),
			categories: z.array(text(24)).min(2).max(8),
			series: z.array(z.object({ name: text(30), data: z.array(z.number().finite()).min(2).max(8) }).strict()).min(1).max(3),
			// d3-format specifier, restricted to a safe character set (e.g. ",.0f", "$,.2f", ".0%").
			valueFormat: z.string().max(12).regex(/^[$,.%0-9a-zA-Z~+\- ]*$/).optional(),
		}).strict(),
		primary: hex,
		secondary: hex.optional(),
		background: hex.default("#0E0F12"),
		textColor: hex.default("#FFFFFF"),
		durationSec: z.number().min(2).max(15).default(5),
	}).strict(),
}).strict();

export const ProductCarousel = z.object({
	template: z.literal("product_carousel"),
	...canvas,
	params: z.object({
		images: z.array(path).min(1).max(6),
		font: path,
		title: text(60).optional(),
		primary: hex,
		background: hex.default("#0E0F12"),
		textColor: hex.default("#FFFFFF"),
		durationSec: z.number().min(2).max(15).default(5),
	}).strict(),
}).strict();

export const KineticCaptions = z.object({
	template: z.literal("kinetic_captions"),
	...canvas,
	params: z.object({
		video: path,
		font: path,
		words: z.array(z.object({ text: text(40), start: z.number().min(0), end: z.number().min(0) }).strict()).min(1).max(600),
		textColor: hex.default("#FFFFFF"),
		highlight: hex.default("#E6FF4F"),
		position: z.enum(["bottom", "center"]).default("bottom"),
		music: path.optional(),
		durationSec: z.number().min(0.5).max(600).optional(),
	}).strict(),
}).strict();

export const BeatGridJob = z.object({
	template: z.literal("beat_grid"),
	...canvas,
	params: z.object({ audio: path }).strict(),
}).strict();

export const Job = z.discriminatedUnion("template", [HookTitle, BrandGrade, ChartCard, ProductCarousel, KineticCaptions, BeatGridJob]);
export type Job = z.infer<typeof Job>;
export const TEMPLATES = ["hook_title", "brand_grade", "chart_card", "product_carousel", "kinetic_captions", "beat_grid"] as const;
