// Brand look as a 3D .cube LUT: shadows lean toward the secondary colour, highlights toward the primary,
// a soft S-curve for contrast, and skin-ish hues kept close to the original so faces stay natural.
import fs from "node:fs";
import { rgb } from "./common.js";

const clamp = (v: number) => Math.max(0, Math.min(1, v));

export function writeBrandLut(file: string, primary: string, secondary: string | undefined, strength: number, size = 17): void {
	const hi = rgb(primary);
	const lo = rgb(secondary || primary);
	// Tints as offsets from neutral grey, kept subtle.
	const tint = (c: [number, number, number], k: number) => {
		const m = (c[0] + c[1] + c[2]) / 3;
		return c.map((v) => (v - m) * k) as [number, number, number];
	};
	const hiT = tint(hi, 0.22 * strength);
	const loT = tint(lo, 0.28 * strength);
	const lines = [`TITLE "SpreadOut brand"`, `LUT_3D_SIZE ${size}`, "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"];
	for (let b = 0; b < size; b++) for (let g = 0; g < size; g++) for (let r = 0; r < size; r++) {
		let R = r / (size - 1), G = g / (size - 1), B = b / (size - 1);
		const L = 0.2126 * R + 0.7152 * G + 0.0722 * B;
		// Soft S-curve on luma, applied as a ratio so hue stays.
		const sL = L + strength * 0.18 * (L - 0.5) * (1 - Math.abs(2 * L - 1));
		const ratio = L > 0.001 ? sL / L : 1;
		R *= ratio; G *= ratio; B *= ratio;
		// Skin protection: warm, mid-saturated colours get less tint.
		const skin = R > G && G > B && R - B > 0.08 && R - B < 0.45 ? 0.45 : 1;
		const wHi = L * L * skin;
		const wLo = (1 - L) * (1 - L) * skin;
		R += hiT[0] * wHi + loT[0] * wLo;
		G += hiT[1] * wHi + loT[1] * wLo;
		B += hiT[2] * wHi + loT[2] * wLo;
		lines.push(`${clamp(R).toFixed(6)} ${clamp(G).toFixed(6)} ${clamp(B).toFixed(6)}`);
	}
	fs.writeFileSync(file, lines.join("\n") + "\n");
}
