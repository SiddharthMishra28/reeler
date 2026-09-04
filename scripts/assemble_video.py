import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

BUILD = Path("build")
FPS = 30
W, H = 1080, 1920
GAP = 0.3
TAIL = 0.7
ZOOM_MAX = 1.12
SR = 24000
PRE_W = W * 12 // 10
PRE_H = H * 12 // 10


def sh(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def render_scene(image, frames, out_path):
    zoom_delta = (ZOOM_MAX - 1.0) / frames
    video_filter = (
        f"scale={PRE_W}:{PRE_H}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={PRE_W}:{PRE_H},"
        f"zoompan=z='min(zoom+{zoom_delta:.6f},{ZOOM_MAX})':d={frames}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={FPS}"
    )
    sh(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(image),
            "-filter_complex", video_filter,
            "-frames:v", str(frames), "-r", str(FPS),
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
            "-crf", "27", "-pix_fmt", "yuv420p", str(out_path),
        ]
    )


def main():
    narration = json.loads((BUILD / "narration.json").read_text(encoding="utf-8"))
    durations = [s["duration"] for s in narration["scenes"]]
    n = len(durations)
    for i in range(1, n + 1):
        image = BUILD / f"scene_{i}.png"
        if not image.exists():
            raise SystemExit(f"missing {image}")
    frames = [max(int(round((d + GAP) * FPS)), 45) for d in durations]
    frames[-1] += int(round(TAIL * FPS))
    for i in range(1, n + 1):
        render_scene(BUILD / f"scene_{i}.png", frames[i - 1], BUILD / f"scene_{i}.mp4")
    parts = []
    for i in range(1, n + 1):
        arr, _ = sf.read(BUILD / f"scene_{i}.wav", dtype="float32")
        need = frames[i - 1] * SR // FPS
        if len(arr) < need:
            arr = np.pad(arr, (0, need - len(arr)))
        else:
            arr = arr[:need]
        parts.append(arr)
    full = np.concatenate(parts)
    sf.write(BUILD / "narration_full.wav", full, SR, subtype="PCM_16")
    concat_list = BUILD / "concat.txt"
    concat_list.write_text(
        "".join(f"file 'scene_{i}.mp4'\n" for i in range(1, n + 1)), encoding="utf-8"
    )
    sh(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy", str(BUILD / "video_concat.mp4"),
        ]
    )
    sh(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(BUILD / "video_concat.mp4"),
            "-i", str(BUILD / "narration_full.wav"),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
            "-shortest", "-movflags", "+faststart", str(BUILD / "reel.mp4"),
        ]
    )
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(BUILD / "reel.mp4"),
        ],
        capture_output=True, text=True, check=True,
    )
    size_mb = (BUILD / "reel.mp4").stat().st_size / 1_048_576
    print(f"reel.mp4: {float(probe.stdout.strip()):.1f}s, {size_mb:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
