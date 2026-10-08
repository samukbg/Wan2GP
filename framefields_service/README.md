# SpreadOut framefields render service

GPU-native motion design for SpreadOut videos, rendered with [framefields](https://github.com/gatewai-dev/framefields)
(WebGPU via Dawn: Vulkan / D3D12 / Metal — no browser). Called by Wan2GP's `framefields_render` Gradio API.

| Template | What it makes |
| --- | --- |
| `hook_title` | Hook title **behind the presenter** (person cutout); moves the plate down + blurred band when the framing is tight |
| `brand_grade` | Temporal **de-flicker** for AI clips + brand **LUT** (split-tone from the brand colours) + grain + vignette |
| `chart_card` | **Animated chart** card (bar / line / area / donut) for Fast Facts and Top-list |
| `product_carousel` | **3D product carousel** / turntable from product photos |
| `kinetic_captions` | **Kinetic captions** — phrases pop in, words light up as spoken, pulse on the music beat |
| `beat_grid` | Beat grid (BPM, beats, downbeats, onset envelope) of a music track, for **beat-locked cuts** |

## Safety model

- Only the fixed templates above run. No user/LLM code, no expressions (`new Function`) and no shell commands are reachable from a job.
- Jobs are validated with strict zod schemas (`src/schema.ts`); every media path must resolve inside the job directory and URLs are rejected (`src/paths.ts`).
- Wan2GP downloads inputs only from allow-listed hosts (`FRAMEFIELDS_ALLOWED_HOSTS`) and requires `SPREADOUT_RENDER_TOKEN` when it is set.
- The renderer runs with a minimal environment (no server secrets) and a time limit.
- Dependencies are pinned (`package-lock.json`, framefields `2.0.8`, integrity-checked by npm).
- Vision models download on first use from pinned Hugging Face revisions, SHA-256 verified, into `models/`.

## Setup (on each Wan2GP server)

Requirements: **Node.js ≥ 22**, **npm ≥ 11** (honours `allowScripts`: only `skia-canvas` may run an install script), ffmpeg on PATH, a GPU with Vulkan/D3D12.

```bash
cd framefields_service
npm ci --no-audit --no-fund
npm run selftest
```

Wan2GP runs `npm ci` itself the first time a render is requested if `node_modules` is missing.
