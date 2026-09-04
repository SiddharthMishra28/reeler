import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BUILD = Path("build")
OUTPUT = Path("output")


def sh(*args):
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True)


def slugify(value):
    value = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return value or "story"


def main():
    story = json.loads((BUILD / "story.json").read_text(encoding="utf-8"))
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder = slugify(f"{story.get('genre', 'story')}-{story.get('category', 'reel')}")
    dest = OUTPUT / date / folder
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUILD / "reel.mp4", dest / "reel.mp4")
    shutil.copy2(BUILD / "story.json", dest / "story.json")
    for png in sorted(BUILD.glob("scene_*.png")):
        shutil.copy2(png, dest / png.name)
    index = OUTPUT / "INDEX.md"
    title = str(story.get("title", "Untitled")).replace("|", "/")
    if not index.exists():
        index.write_text(
            "| Date | Title | Genre / Category | Video |\n|---|---|---|---|\n", encoding="utf-8"
        )
    with index.open("a", encoding="utf-8") as f:
        f.write(
            f"| {date} | {title} | {story.get('genre', '')} / {story.get('category', '')} "
            f"| [reel.mp4]({date}/{folder}/reel.mp4) |\n"
        )
    sh("git", "config", "user.name", "github-actions[bot]")
    sh("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    sh("git", "checkout", "-B", "main")
    sh("git", "add", "output")
    status = subprocess.run(
        ["git", "status", "--porcelain", "output"],
        capture_output=True, text=True, check=True,
    )
    if not status.stdout.strip():
        print("Nothing new to commit.", flush=True)
        return
    message = f"reel: {story.get('title', 'Untitled')} ({date})"
    sh("git", "commit", "-m", message)
    sh("git", "push", "origin", "main")
    print(f"Committed reel to {dest}", flush=True)


if __name__ == "__main__":
    main()
