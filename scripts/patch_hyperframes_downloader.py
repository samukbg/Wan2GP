"""
Patch the npx-cached hyperframes CLI downloader to tolerate transient HTTP 5xx
(e.g. gradio worker tunnels returning 502 mid-render).

Default hyperframes downloader: maxTransientRetries = 1 (2 attempts, no delay),
so a brief origin blip on the file host fails the whole render. This patch:
  - raises local transient retries to 5 (6 attempts total)
  - adds exponential backoff: 2s, 4s, 8s, 16s, 30s (capped)

Idempotent: safe to run repeatedly. Run it on any machine that renders with
hyperframes, AFTER at least one `npx hyperframes ...` call has populated the
npx cache. Re-run after hyperframes updates (patch is cache-local).

Usage:  python scripts/patch_hyperframes_downloader.py
"""
import glob
import os
import re
import sys

OLD = (
    "async function downloadWithRetry(url, localPath, timeoutMs, signal, "
    "onTransientRetry, options = {}) {\n  const maxTransientRetries = 1;\n"
    "  for (let attempt = 0; ; attempt += 1) {\n"
    "    try {\n"
    "      return await runDownloadAttempt(url, localPath, timeoutMs, attempt + 1, options, signal);\n"
    "    } catch (error) {\n"
    "      const classified = classifyDownloadFailure(error);\n"
)
NEW = (
    "async function downloadWithRetry(url, localPath, timeoutMs, signal, "
    "onTransientRetry, options = {}) {\n  const maxTransientRetries = 5;\n"
    "  for (let attempt = 0; ; attempt += 1) {\n"
    "    try {\n"
    "      return await runDownloadAttempt(url, localPath, timeoutMs, attempt + 1, options, signal);\n"
    "    } catch (error) {\n"
    "      const classified = classifyDownloadFailure(error);\n"
    "      if (classified.locallyRetryable && attempt < maxTransientRetries && attempt > 0) {\n"
    "        await new Promise((r) => setTimeout(r, Math.min(30000, 2000 * Math.pow(2, attempt - 1))));\n"
    "      }\n"
)

CANDIDATE_GLOBS = [
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "npm-cache", "_npx", "*", "node_modules", "hyperframes", "dist", "cli.js"),
    os.path.join(os.path.expanduser("~"), ".npm", "_npx", "*", "node_modules", "hyperframes", "dist", "cli.js"),
]


def patch_file(path):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if "const maxTransientRetries = 5;" in src and "2000 * Math.pow(2, attempt - 1)" in src:
        print(f"already patched: {path}")
        return True
    if OLD not in src:
        print(f"pattern not found (hyperframes version changed?): {path}")
        return False
    patched = src.replace(OLD, NEW, 1)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(patched)
    print(f"patched: {path}")
    return True


def main():
    candidates = []
    for pattern in CANDIDATE_GLOBS:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        print("No hyperframes npx cache found. Run `npx hyperframes --version` once, then retry.")
        return 1
    ok = all(patch_file(p) for p in candidates)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
