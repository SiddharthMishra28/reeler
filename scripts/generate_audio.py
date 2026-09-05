import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

BUILD = Path("build")
SR = 24000
VOICE = os.environ.get("TTS_VOICE", "af_heart")
TARGET_SECONDS = 300
MIN_SPEED = 0.85
MAX_SPEED = 1.15


def synth(pipeline, text, speed):
    chunks = []
    try:
        results = pipeline(text, voice=VOICE, speed=speed)
    except TypeError:
        results = pipeline(text, voice=VOICE)
    for result in results:
        if isinstance(result, tuple):
            audio = result[2]
        else:
            audio = getattr(result, "audio", None)
        if audio is None:
            continue
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        chunks.append(arr)
    if not chunks:
        return np.zeros(int(0.4 * SR), dtype=np.float32)
    return np.concatenate(chunks)


def main():
    story = json.loads((BUILD / "story.json").read_text(encoding="utf-8"))
    scenes = story["scenes"]
    print(f"Loading Kokoro TTS (voice {VOICE})...", flush=True)
    pipeline = KPipeline(lang_code="a")
    speed = 1.0
    audios = [synth(pipeline, s["text"], speed) for s in scenes]
    total = sum(len(a) for a in audios) / SR
    print(f"Pass 1 narration length: {total:.1f}s (target {TARGET_SECONDS}s)", flush=True)
    if abs(total - TARGET_SECONDS) / TARGET_SECONDS > 0.04:
        target_speed = round(total / TARGET_SECONDS, 2)
        target_speed = min(max(target_speed, MIN_SPEED), MAX_SPEED)
        if abs(target_speed - 1.0) >= 0.02:
            speed = target_speed
            print(f"Renarrating at speed {speed} to fit {TARGET_SECONDS}s", flush=True)
            audios = [synth(pipeline, s["text"], speed) for s in scenes]
            total = sum(len(a) for a in audios) / SR
            print(f"Pass 2 narration length: {total:.1f}s", flush=True)
    records = []
    for i, audio in enumerate(audios, 1):
        out_path = BUILD / f"scene_{i}.wav"
        sf.write(out_path, audio, SR, subtype="PCM_16")
        records.append({"scene": i, "file": out_path.name, "duration": len(audio) / SR})
    narration = {
        "sample_rate": SR,
        "voice": VOICE,
        "speed": speed,
        "scenes": records,
        "total_duration": total,
    }
    (BUILD / "narration.json").write_text(json.dumps(narration, indent=2), encoding="utf-8")
    print(f"Narration done: {len(records)} scenes, {total:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
