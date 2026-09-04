# reeler

A fully automated short-story reel factory that runs on free GitHub Actions CPU runners.

Every day it picks a random genre x category, writes a ~3-minute narration script, narrates it with the lightweight Kokoro-82M TTS model, illustrates it with the ultra-compact OFA-Sys/small-stable-diffusion-v0 model, and assembles a 1080x1920 Ken-Burns-style reel with ffmpeg. The finished reel is committed back to this repository under `output/`.

## Pipeline

| Stage | Tool |
|---|---|
| Script | OpenAI-compatible LLM (`muse-spark-1.2-contributor-free` via router.bynara.id) |
| Narration | Kokoro-82M TTS (CPU, voice `af_heart`, auto speed-up to fit 3 minutes) |
| Images | OFA-Sys/small-stable-diffusion-v0 (CPU, 8 steps, attention slicing, 512x512) |
| Video | ffmpeg (1080x1920 vertical, Ken-Burns zoom, H.264 `-tune stillimage`) |

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

## Runner constraints

- Peak RAM ~3-4 GB (limit 7 GB): stages run in separate processes so model memory never stacks.
- Runtime ~20-30 min (job timeout 50 min).
- Public repository, so Actions minutes are free and unlimited.
