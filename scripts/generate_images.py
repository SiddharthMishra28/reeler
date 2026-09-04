import gc
import json
import os
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

BUILD = Path("build")
MODEL_ID = os.environ.get("IMAGE_MODEL", "OFA-Sys/small-stable-diffusion-v0")
STEPS = int(os.environ.get("IMAGE_STEPS", "8"))
GUIDANCE = float(os.environ.get("IMAGE_GUIDANCE", "3.5"))
SIZE = int(os.environ.get("IMAGE_SIZE", "512"))
STYLE_SUFFIX = "cinematic digital art, dramatic lighting, rich colors, highly detailed"
NEGATIVE_PROMPT = (
    "text, words, letters, watermark, signature, logo, blurry, low quality, "
    "deformed, disfigured, extra limbs, bad anatomy"
)


def main():
    story = json.loads((BUILD / "story.json").read_text(encoding="utf-8"))
    scenes = story["scenes"]
    print(f"Loading {MODEL_ID} on CPU...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    manifest = []
    for i, scene in enumerate(scenes, 1):
        prompt = f"{scene['image_prompt']}. {STYLE_SUFFIX}"
        out_path = BUILD / f"scene_{i}.png"
        for attempt in range(1, 4):
            try:
                started = time.time()
                image = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE,
                    height=SIZE,
                    width=SIZE,
                ).images[0]
                image.save(out_path)
                elapsed = time.time() - started
                print(
                    f"scene {i}: saved {out_path.name} ({elapsed:.0f}s, {STEPS} steps)",
                    flush=True,
                )
                manifest.append(
                    {"scene": i, "file": out_path.name, "prompt": prompt, "steps": STEPS}
                )
                break
            except Exception as exc:
                print(f"scene {i} attempt {attempt} failed: {exc}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(3)
    (BUILD / "images.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    del pipe
    gc.collect()
    print(f"Generated {len(manifest)} images", flush=True)


if __name__ == "__main__":
    main()
