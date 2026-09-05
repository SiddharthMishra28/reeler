import gc
import json
import os
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

BUILD = Path("build")
MODEL_ID = os.environ.get("IMAGE_MODEL", "stabilityai/sd-turbo")
STEPS = int(os.environ.get("IMAGE_STEPS", "2"))
GUIDANCE = float(os.environ.get("IMAGE_GUIDANCE", "0.0"))
WIDTH = int(os.environ.get("IMAGE_WIDTH", "512"))
HEIGHT = int(os.environ.get("IMAGE_HEIGHT", "896"))
SEED = int(os.environ.get("IMAGE_SEED", "0"))
QA_MODEL = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
MIN_SCORE = float(os.environ.get("CLIP_MIN_SCORE", "0.22"))
GOOD_SCORE = float(os.environ.get("CLIP_GOOD_SCORE", "0.27"))
QA_ATTEMPTS = int(os.environ.get("CLIP_ATTEMPTS", "3"))
SCENE_BUDGET = float(os.environ.get("IMAGE_SCENE_BUDGET", "300"))


def simplify_prompt(prompt):
    """Reduce a failed prompt to its most concrete core for the diffusion model."""
    words = prompt.replace(",", " ").split()
    stop = {"the", "a", "an", "with", "and", "in", "on", "of", "under", "over"}
    kept = [w for w in words if w.lower() not in stop]
    return " ".join(kept[:20])


def main():
    story = json.loads((BUILD / "story.json").read_text(encoding="utf-8"))
    scenes = story["scenes"]

    print(f"Loading {MODEL_ID} on CPU...", flush=True)
    pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    print(f"Loading CLIP judge {QA_MODEL}...", flush=True)
    from transformers import CLIPModel, CLIPProcessor

    clip_model = CLIPModel.from_pretrained(QA_MODEL, torch_dtype=torch.float32)
    clip_processor = CLIPProcessor.from_pretrained(QA_MODEL)
    clip_model.eval()

    def clip_score(image, prompt):
        inputs = clip_processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        with torch.no_grad():
            img_feat = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_feat = clip_model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ txt_feat.T).item()
        return sim

    manifest = []
    all_scores = []
    for i, scene in enumerate(scenes, 1):
        prompt = scene["image_prompt"]
        out_path = BUILD / f"scene_{i}.png"
        chosen = None
        chosen_score = -1.0
        chosen_prompt = prompt
        chosen_seed = SEED
        scene_started = time.time()
        for attempt in range(1, QA_ATTEMPTS + 1):
            if chosen is not None and (time.time() - scene_started) > SCENE_BUDGET:
                print(f"scene {i}: time budget hit, keeping best so far", flush=True)
                break
            seed_used = SEED + i * 1000 + attempt
            gen = torch.Generator(device="cpu").manual_seed(seed_used)
            eff_prompt = prompt if attempt <= 2 else simplify_prompt(prompt)
            try:
                started = time.time()
                image = pipe(
                    prompt=eff_prompt,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE,
                    height=HEIGHT,
                    width=WIDTH,
                    generator=gen,
                ).images[0]
                score = clip_score(image, prompt)
                elapsed = time.time() - started
                print(
                    f"scene {i} try {attempt} (seed {seed_used}): CLIP {score:.3f} ({elapsed:.0f}s)",
                    flush=True,
                )
                if score > chosen_score:
                    chosen = image
                    chosen_score = score
                    chosen_prompt = eff_prompt
                    chosen_seed = seed_used
                if score >= GOOD_SCORE:
                    break
            except Exception as exc:
                print(f"scene {i} try {attempt} errored: {exc}", flush=True)
                time.sleep(2)
        if chosen is None:
            raise SystemExit(f"scene {i}: all attempts failed")
        chosen.save(out_path)
        all_scores.append(chosen_score)
        manifest.append(
            {
                "scene": i,
                "file": out_path.name,
                "prompt": chosen_prompt,
                "clip_score": round(chosen_score, 4),
                "seed": chosen_seed,
                "passed": chosen_score >= MIN_SCORE,
            }
        )

    (BUILD / "images.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    del pipe, clip_model
    gc.collect()
    avg = sum(all_scores) / len(all_scores)
    weak = sum(1 for s in all_scores if s < MIN_SCORE)
    print(
        f"Images done: {len(manifest)} kept, avg CLIP {avg:.3f}, {weak} below {MIN_SCORE}",
        flush=True,
    )


if __name__ == "__main__":
    main()
