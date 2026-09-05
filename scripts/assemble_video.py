import json
import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

BUILD = Path("build")
FPS = 30
W, H = 1080, 1920
XFADE = 0.6
FADE_IN = 0.8
FADE_OUT = 1.2
TAIL = 1.0
GAP = 0.4
ZOOM_MAX = 1.10
SR = 24000
MUSIC_LEVEL = 0.16
TITLE_SECONDS = 2.8
TITLE_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
MOTION_STYLES = ["zoom_in", "zoom_out", "pan_left", "pan_right"]


def sh(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def esc(text):
    """Escape text for ffmpeg drawtext (used only for non-textfile options)."""
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("%", "\\%")
    )


def find_font():
    for candidate in TITLE_FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def motion_filter(style, frames):
    common = (
        f"scale=1536:2688:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop=1536:2688,unsharp=5:5:0.4:5:5:0.0,"
    )
    if style == "zoom_in":
        zd = (ZOOM_MAX - 1.0) / frames
        return (
            common
            + f"zoompan=z='min(zoom+{zd:.6f},{ZOOM_MAX})':d={frames}:"
            + f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={FPS}"
        )
    if style == "zoom_out":
        zd = (ZOOM_MAX - 1.0) / frames
        return (
            common
            + f"zoompan=z='if(eq(on,1),{ZOOM_MAX},max(zoom-{zd:.6f},1.0))':d={frames}:"
            + f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={FPS}"
        )
    if style == "pan_left":
        return (
            common
            + f"zoompan=z={ZOOM_MAX}:d={frames}:"
            + f"x='(iw-iw/zoom)*on/{frames}':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={FPS}"
        )
    return (
        common
        + f"zoompan=z={ZOOM_MAX}:d={frames}:"
        + f"x='(iw-iw/zoom)*(1-on/{frames})':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={FPS}"
    )


def render_scene(image, frames, style, out_path, title_text=None, font=None):
    filt = motion_filter(style, frames)
    if title_text and font:
        title_file = BUILD / "title.txt"
        title_file.write_text(str(title_text), encoding="utf-8")
        fade_in_end = TITLE_SECONDS * 0.25
        alpha = (
            f"alpha='if(lt(t,{fade_in_end:.2f}),t/{fade_in_end:.2f},"
            f"if(lt(t,{TITLE_SECONDS:.2f}),1,"
            f"if(lt(t,{TITLE_SECONDS + 0.4:.2f}),({TITLE_SECONDS + 0.4:.2f}-t)/0.4,0)))'"
        )
        filt += (
            f",drawtext=fontfile={font}:textfile={title_file}:"
            f"fontcolor=white:fontsize=68:borderw=3:bordercolor=black@0.65:"
            f"x=(w-text_w)/2:y=h*0.40:{alpha}"
        )
    sh(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(image),
            "-filter_complex", filt,
            "-frames:v", str(frames), "-r", str(FPS),
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
            "-crf", "27", "-pix_fmt", "yuv420p", str(out_path),
        ]
    )


def make_music(total_seconds, out_path):
    """Synthesize a soft ambient chord pad progression as the music bed."""
    story = json.loads((BUILD / "story.json").read_text(encoding="utf-8"))
    palette_len = max(len(story.get("palette", "").split(",")), 1)
    roots = [110.0, 130.81, 146.83, 98.0, 116.54]
    t = np.arange(int(total_seconds * SR)) / SR
    audio = np.zeros_like(t)
    chord_dur = 12.0
    n_chords = int(math.ceil(total_seconds / chord_dur))
    for c in range(n_chords):
        root = roots[(c + palette_len) % len(roots)]
        start = int(c * chord_dur * SR)
        end = min(int((c + 1) * chord_dur * SR), len(t))
        if start >= end:
            break
        seg_t = t[start:end] - t[start]
        chord = np.zeros_like(seg_t)
        for f in (root, root * 1.5, root * 2.0, root * 2.9966):
            vibrato = 1.0 + 0.0015 * np.sin(2 * np.pi * 0.2 * seg_t + f)
            tone = np.sin(2 * np.pi * f * vibrato * seg_t)
            tone += 0.35 * np.sin(2 * np.pi * f * 2 * seg_t)
            chord += tone / 4
        fade_len = min(int(2.0 * SR), len(seg_t) // 2)
        env = np.ones_like(seg_t)
        env[:fade_len] = np.linspace(0, 1, fade_len)
        env[-fade_len:] = np.linspace(1, 0, fade_len)
        audio[start:end] += chord * env
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    audio = audio * MUSIC_LEVEL
    lead = int(1.5 * SR)
    if len(audio) > lead:
        audio[:lead] *= np.linspace(0, 1, lead)
    tail = int(3.0 * SR)
    if len(audio) > tail:
        audio[-tail:] *= np.linspace(1, 0, tail)
    sf.write(out_path, audio.astype(np.float32), SR, subtype="PCM_16")


def main():
    story = json.loads((BUILD / "story.json").read_text(encoding="utf-8"))
    narration = json.loads((BUILD / "narration.json").read_text(encoding="utf-8"))
    durations = [s["duration"] for s in narration["scenes"]]
    n = len(durations)
    title = story.get("title", "Untitled")
    for i in range(1, n + 1):
        image = BUILD / f"scene_{i}.png"
        if not image.exists():
            raise SystemExit(f"missing {image}")

    # Per-scene clip durations: lead-in (transition/fade) + narration + gap.
    pre = [FADE_IN] + [XFADE] * (n - 1)
    post = [GAP] * (n - 1) + [TAIL + FADE_OUT]
    clip_seconds = [pre[i] + durations[i] + post[i] for i in range(n)]
    frames = [int(round(v * FPS)) for v in clip_seconds]
    clip_seconds = [f / FPS for f in frames]

    # Scene start offsets in the final xfade timeline.
    offsets = [0.0]
    for i in range(1, n):
        offsets.append(offsets[-1] + clip_seconds[i - 1] - XFADE)
    total = offsets[-1] + clip_seconds[-1]

    font = find_font()
    for i in range(1, n + 1):
        style = MOTION_STYLES[(i - 1) % len(MOTION_STYLES)]
        render_scene(
            BUILD / f"scene_{i}.png",
            frames[i - 1],
            style,
            BUILD / f"scene_{i}.mp4",
            title_text=title if i == 1 else None,
            font=font,
        )

    # Build the narration track by placing each scene at its exact offset.
    track = np.zeros(int(round(total * SR)) + SR, dtype=np.float32)
    for i in range(1, n + 1):
        arr, _ = sf.read(BUILD / f"scene_{i}.wav", dtype="float32")
        start = int(round((offsets[i - 1] + pre[i - 1]) * SR))
        end = min(start + len(arr), len(track))
        track[start:end] += arr[: end - start]
    track = track[: int(round(total * SR))]
    fi = int(FADE_IN * SR)
    track[:fi] *= np.linspace(0, 1, fi)
    fo = int(FADE_OUT * SR)
    track[-fo:] *= np.linspace(1, 0, fo)
    sf.write(BUILD / "narration_full.wav", track, SR, subtype="PCM_16")

    print(f"Synthesizing ambient music bed ({total:.1f}s)...", flush=True)
    make_music(total, BUILD / "music.wav")
    music, _ = sf.read(BUILD / "music.wav", dtype="float32")
    mix = track + music[: len(track)]
    peak = np.max(np.abs(mix))
    if peak > 0.98:
        mix = mix / peak * 0.98
    sf.write(BUILD / "final_audio.wav", mix, SR, subtype="PCM_16")

    # Video: xfade chain + global fade in/out, then mux with the mixed audio.
    inputs = []
    for i in range(1, n + 1):
        inputs += ["-i", str(BUILD / f"scene_{i}.mp4")]
    inputs += ["-i", str(BUILD / "final_audio.wav")]
    filter_parts = []
    prev_out = "[0:v]"
    offset = 0.0
    for i in range(1, n):
        offset = offsets[i]
        label = f"[v{i}]"
        filter_parts.append(
            f"{prev_out}[{i}:v]xfade=transition=fade:duration={XFADE}:offset={offset:.3f}{label}"
        )
        prev_out = label
    fade_out_start = max(total - FADE_OUT, 0)
    filter_parts.append(
        f"{prev_out}fade=t=in:st=0:d={FADE_IN},"
        f"fade=t=out:st={fade_out_start:.3f}:d={FADE_OUT}[vfinal]"
    )
    fc = ";".join(filter_parts)
    sh(
        [
            "ffmpeg", "-y", "-loglevel", "error", *inputs,
            "-filter_complex", fc,
            "-map", "[vfinal]", "-map", f"{n}:a",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "27",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart",
            str(BUILD / "reel.mp4"),
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
    print(
        f"timeline check: video total {total:.2f}s, narration placed at "
        + ", ".join(f"s{i+1}@{offsets[i]:.1f}s" for i in range(n)),
        flush=True,
    )


if __name__ == "__main__":
    main()
