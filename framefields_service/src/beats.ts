// Beat grid for a music file: deterministic onset envelope + autocorrelation tempo + phase fit.
// ffmpeg decodes (argument array, no shell); the rest is plain DSP.
import { spawn } from "node:child_process";

export interface BeatGrid {
	bpm: number;
	/** Beat times in seconds. */
	beats: number[];
	/** Every 4th beat starting at the strongest phase (bar starts). */
	downbeats: number[];
	/** Onset strength per video frame (0..1), for beat-reactive motion. */
	envelope: number[];
	envelopeFps: number;
	durationSec: number;
}

const SR = 22050;
const HOP = 512;

export function decodeMono(file: string, sampleRate = SR): Promise<Float32Array> {
	return new Promise((resolve, reject) => {
		const ff = spawn("ffmpeg", ["-v", "error", "-i", file, "-ac", "1", "-ar", String(sampleRate), "-f", "f32le", "pipe:1"], { shell: false });
		const chunks: Buffer[] = [];
		let err = "";
		ff.stdout.on("data", (c: Buffer) => chunks.push(c));
		ff.stderr.on("data", (c: Buffer) => { err += c.toString(); });
		ff.on("error", reject);
		ff.on("close", (code) => {
			if (code !== 0) return reject(new Error(`ffmpeg decode failed (${code}): ${err.slice(0, 300)}`));
			const buf = Buffer.concat(chunks);
			resolve(new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4)));
		});
	});
}

/** Spectral-flux-like onset strength: rectified rise of low-band and full-band energy per hop. */
export function onsetEnvelope(samples: Float32Array): Float32Array {
	const n = Math.floor(samples.length / HOP);
	const env = new Float32Array(n);
	// One-pole low-pass to emphasise kicks/bass.
	let lp = 0;
	const a = Math.exp((-2 * Math.PI * 150) / SR);
	let prevLow = 0;
	let prevFull = 0;
	for (let i = 0; i < n; i++) {
		let low = 0;
		let full = 0;
		for (let j = 0; j < HOP; j++) {
			const x = samples[i * HOP + j];
			lp = a * lp + (1 - a) * x;
			low += lp * lp;
			full += x * x;
		}
		low = Math.log1p(1000 * Math.sqrt(low / HOP));
		full = Math.log1p(1000 * Math.sqrt(full / HOP));
		env[i] = Math.max(0, low - prevLow) * 0.65 + Math.max(0, full - prevFull) * 0.35;
		prevLow = low;
		prevFull = full;
	}
	// Normalise.
	let max = 0;
	for (const v of env) max = Math.max(max, v);
	if (max > 0) for (let i = 0; i < n; i++) env[i] /= max;
	return env;
}

export function estimateBeats(env: Float32Array, durationSec: number, minBpm = 70, maxBpm = 170): { bpm: number; beats: number[]; downbeats: number[] } {
	const hopSec = HOP / SR;
	const minLag = Math.max(2, Math.floor(60 / maxBpm / hopSec));
	const maxLag = Math.ceil(60 / minBpm / hopSec);
	const ac = (lag: number) => { let s = 0; for (let i = lag; i < env.length; i++) s += env[i] * env[i - lag]; return s / Math.max(1, env.length - lag); };
	const scores: number[] = [];
	let bestLag = minLag;
	let best = -1;
	for (let lag = minLag; lag <= maxLag; lag++) {
		// Mild preference for tempos near 118 BPM (typical for short-form music).
		const bpm = 60 / (lag * hopSec);
		const score = ac(lag) * (0.75 + 0.25 * Math.exp(-0.5 * ((bpm - 118) / 45) ** 2));
		scores[lag] = score;
		if (score > best) { best = score; bestLag = lag; }
	}
	// Fractional period: parabola through the peak and its neighbours.
	const a = scores[bestLag - 1] ?? best, b = best, c = scores[bestLag + 1] ?? best;
	const denom = a - 2 * b + c;
	const period = bestLag + (denom !== 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (a - c)) / denom)) : 0);

	// Track: start on the strongest onset in the first two periods, then predict + snap each beat to the
	// strongest onset within ±15% of a period (weighted toward the prediction).
	const win = Math.max(1, Math.round(period * 0.15));
	let first = 0;
	for (let i = 0; i < Math.min(env.length, Math.round(period * 2)); i++) if (env[i] > env[first]) first = i;
	const beatIdx: number[] = [first];
	let pos = first;
	while (pos + period < env.length) {
		const pred = pos + period;
		let bestI = Math.round(pred);
		let bestV = -1;
		for (let i = Math.max(0, Math.round(pred) - win); i <= Math.min(env.length - 1, Math.round(pred) + win); i++) {
			const v = env[i] * Math.exp(-0.5 * ((i - pred) / win) ** 2);
			if (v > bestV) { bestV = v; bestI = i; }
		}
		pos = env[bestI] > 0.08 ? bestI : pred; // silence: keep the steady grid
		beatIdx.push(pos);
	}
	// Back-fill beats before the first strong onset.
	for (let p0 = first - period; p0 >= 0; p0 -= period) beatIdx.unshift(p0);
	const beats = beatIdx.map((i) => Number((i * hopSec).toFixed(3))).filter((t) => t <= durationSec);
	// Downbeats: the phase (0..3) whose beats are strongest.
	let barPhase = 0;
	let barBest = -1;
	for (let k = 0; k < 4; k++) {
		let s = 0;
		for (let j = k; j < beats.length; j += 4) s += env[Math.round(beats[j] / hopSec)] ?? 0;
		if (s > barBest) { barBest = s; barPhase = k; }
	}
	const downbeats = beats.filter((_, i) => i % 4 === barPhase);
	return { bpm: Number((60 / (period * hopSec)).toFixed(1)), beats, downbeats };
}

export async function beatGrid(file: string, fps = 30): Promise<BeatGrid> {
	const samples = await decodeMono(file);
	const durationSec = samples.length / SR;
	const env = onsetEnvelope(samples);
	const { bpm, beats, downbeats } = estimateBeats(env, durationSec);
	// Resample the envelope to video frames (max over each frame window).
	const frames = Math.ceil(durationSec * fps);
	const perFrame = SR / HOP / fps;
	const envelope: number[] = [];
	for (let f = 0; f < frames; f++) {
		let m = 0;
		for (let i = Math.floor(f * perFrame); i < Math.min(env.length, Math.floor((f + 1) * perFrame) + 1); i++) m = Math.max(m, env[i]);
		envelope.push(Number(m.toFixed(3)));
	}
	return { bpm, beats, downbeats, envelope, envelopeFps: fps, durationSec: Number(durationSec.toFixed(3)) };
}
