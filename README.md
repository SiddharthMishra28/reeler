# reeler

A fully automated story-reel factory that runs on free GitHub Actions CPU runners.

Every day it picks a random genre x category, writes a 5-minute "mini movie" script with a 10-scene 3-act structure, narrates it with the lightweight Kokoro-82M TTS model, illustrates it with the distilled SD-Turbo model (with automatic CLIP prompt-adherence QA), and assembles a 1080x1920 reel with Ken-Burns motion, crossfade transitions, a title card, and a synthesized ambient music bed. The finished reel is committed back to this repository under `output/`.

## Pipeline

| Stage | Tool |
|---|---|
| Script | OpenAI-compatible LLM (`muse-spark-1.2-contributor-free` via router.bynara.id) - 10 scenes, 3 acts, visual bible with fixed character descriptors |
| Narration | Kokoro-82M TTS (CPU, voice `af_heart`, auto speed fit to ~5 minutes) |
| Images | SD-Turbo (CPU, 2 steps, 512x896 portrait), CLIP ViT-B/32 best-of-N QA per scene with simplified-prompt retries |
| Video | ffmpeg: Ken-Burns (zoom in/out, pan left/right), 0.6s crossfades, title card overlay, synthesized ambient music bed, 1080x1920 H.264 |

## Runs

- Daily at 06:00 UTC (schedule)
- On every push to `main`
- Manually from the Actions tab (optional genre / category overrides)

## Secrets

| Secret | Purpose |
|---|---|
| `BYNARA_API_KEY` | API key for the scriptwriter endpoint |

## Output

`output/YYYY-MM-DD/<genre>-<category>/` containing `reel.mp4`, `story.json`, and the scene stills. `output/INDEX.md` lists every reel produced.

## Tunables (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `IMAGE_MODEL` | `stabilityai/sd-turbo` | Diffusion model |
| `IMAGE_STEPS` | `2` | Inference steps |
| `IMAGE_WIDTH` x `IMAGE_HEIGHT` | `512x896` | Portrait generation size |
| `CLIP_MIN_SCORE` / `CLIP_GOOD_SCORE` | `0.22` / `0.27` | QA thresholds (cosine similarity) |
| `CLIP_ATTEMPTS` | `3` | Max generations per scene |
| `IMAGE_SCENE_BUDGET` | `300` | Seconds allowed per scene |
| `TTS_VOICE` | `af_heart` | Kokoro voice |

## Runner constraints

- Peak RAM ~5-6 GB (limit 7 GB): stages run in separate processes so model memory never stacks.
- Runtime ~30-45 min (job timeout 50 min).
- Public repository, so Actions minutes are free and unlimited.
