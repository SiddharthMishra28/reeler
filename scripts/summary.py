import json
import os
from pathlib import Path

BUILD = Path("build")


def main():
    lines = ["## Reel run summary"]
    story_path = BUILD / "story.json"
    if story_path.exists():
        story = json.loads(story_path.read_text(encoding="utf-8"))
        lines += [
            f"**Title:** {story.get('title', '')}",
            f"**Genre / Category:** {story.get('genre', '')} / {story.get('category', '')}",
            f"**Scenes:** {len(story.get('scenes', []))} · **Words:** {story.get('total_words', '?')}",
            f"**Characters:** "
            + ", ".join(c.get("name", "") for c in story.get("characters", [])),
        ]
    narration_path = BUILD / "narration.json"
    if narration_path.exists():
        narration = json.loads(narration_path.read_text(encoding="utf-8"))
        lines.append(
            f"**Narration:** {narration.get('total_duration', 0):.1f}s · "
            f"speed {narration.get('speed', 1)} · voice `{narration.get('voice', '')}`"
        )
    images_path = BUILD / "images.json"
    if images_path.exists():
        images = json.loads(images_path.read_text(encoding="utf-8"))
        scores = [img.get("clip_score", 0) for img in images]
        weak = [img["scene"] for img in images if not img.get("passed", True)]
        if scores:
            lines.append(
                f"**Images:** {len(images)} · CLIP avg {sum(scores) / len(scores):.3f} · "
                f"min {min(scores):.3f}"
                + (f" · weak scenes: {weak}" if weak else "")
            )
    reel = BUILD / "reel.mp4"
    if reel.exists():
        lines.append(f"**Video:** {reel.stat().st_size / 1_048_576:.1f} MB")
    text = "\n".join(lines) + "\n"
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
