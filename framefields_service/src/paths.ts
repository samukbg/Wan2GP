// Every media path in a job must resolve inside the job directory (Python downloads inputs there).
// URLs are rejected outright, so no network fetch or shell command can be steered from a job.
import fs from "node:fs";
import path from "node:path";

export function insideJobDir(jobDir: string, p: string): string {
	if (/^[a-z][a-z0-9+.-]*:\/\//i.test(p) || p.startsWith("data:")) throw new Error(`URLs are not accepted: ${p.slice(0, 60)}`);
	const root = fs.realpathSync(jobDir);
	const full = fs.realpathSync(path.resolve(root, p));
	if (full !== root && !full.startsWith(root + path.sep)) throw new Error(`Path escapes the job directory: ${p}`);
	if (!fs.statSync(full).isFile()) throw new Error(`Not a file: ${p}`);
	return full;
}
