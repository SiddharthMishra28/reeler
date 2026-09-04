import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

BUILD = Path("build")
API_KEY = os.environ.get("BYNARA_API_KEY", "").strip()
BASE_URL = os.environ.get("BYNARA_BASE_URL", "https://router.bynara.id/v1")
MODEL = os.environ.get("BYNARA_MODEL", "muse-spark-1.2-contributor-free")
GENRE_ENV = os.environ.get("GENRE", "").strip()
CATEGORY_ENV = os.environ.get("CATEGORY", "").strip()

GENRES = [
    "Sci-Fi", "Fantasy", "Mystery", "Horror", "Romance", "Adventure",
    "Comedy", "Thriller", "Fairy Tale", "Slice of Life", "Cyberpunk",
    "Western", "Post-Apocalyptic", "Magical Realism", "Mythology Retelling",
]
CATEGORIES = [
    "Bedtime Story", "Cautionary Tale", "Hero's Journey", "The Heist",
    "Ghost Story", "First Contact", "Lost Love", "Sweet Revenge",
    "Redemption Arc", "Underdog Victory", "Time Loop", "Forbidden Discovery",
    "Unlikely Friendship", "The Last of Its Kind", "A Deal Gone Wrong",
]

SYSTEM = (
    "You are a professional short-form video scriptwriter. You write original, "
    "gripping micro-stories that will be narrated aloud by a calm AI voice over "
    "still artwork. You always respond with valid JSON only."
)

TEMPLATE = """Write a narration script for a 3-minute vertical video reel.

Genre: {genre}
Category: {category}

Rules:
1. Respond with ONLY one valid JSON object. No markdown, no code fences, no commentary.
2. Schema: {{"title": string, "logline": string, "scenes": [{{"text": string, "image_prompt": string}}]}}
3. Exactly 5 scenes.
4. Each scene "text" is 70-85 words of flowing spoken narration, no headings, no scene numbers, no dialogue tags.
5. Scene 1 must open with a one-sentence hook that makes people stop scrolling.
6. Scene 5 must end with a satisfying payoff, twist, or haunting final line.
7. The full story must be narratable in under 3 minutes at a natural pace.
8. Each "image_prompt" describes one striking still image in under 25 words: subject, setting, mood, color palette, lighting. No close-up faces, no text or letters, no watermarks.
9. Family-friendly and completely original."""


def extract_json(raw):
    text = (raw or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in response")
    return json.loads(text[start : end + 1])


def validate(data, genre, category):
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    title = str(data.get("title") or "Untitled Reel").strip()
    scenes_raw = data.get("scenes")
    if not isinstance(scenes_raw, list) or len(scenes_raw) < 4:
        raise ValueError("need at least 4 scenes")
    scenes = []
    for i, sc in enumerate(scenes_raw[:5], 1):
        if not isinstance(sc, dict):
            raise ValueError(f"scene {i} is not an object")
        text = " ".join(str(sc.get("text") or "").split())
        image_prompt = " ".join(
            str(sc.get("image_prompt") or sc.get("imagePrompt") or "").split()
        )
        if len(text) < 100:
            raise ValueError(f"scene {i} narration too short")
        if not image_prompt:
            raise ValueError(f"scene {i} missing image_prompt")
        scenes.append(
            {
                "text": text,
                "image_prompt": image_prompt[:300],
                "word_count": len(text.split()),
            }
        )
    if len(scenes) < 4:
        raise ValueError("need at least 4 valid scenes")
    return {
        "title": title[:120],
        "logline": str(data.get("logline") or "").strip()[:300],
        "genre": genre,
        "category": category,
        "scenes": scenes,
    }


def call_llm(client, genre, category):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.85,
        max_tokens=1800,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": TEMPLATE.format(genre=genre, category=category)},
        ],
    )
    return response.choices[0].message.content or ""


def main():
    if not API_KEY:
        sys.exit("BYNARA_API_KEY is not set")
    BUILD.mkdir(parents=True, exist_ok=True)
    genre = GENRE_ENV or random.choice(GENRES)
    category = CATEGORY_ENV or random.choice(CATEGORIES)
    print(f"Selected combo: {genre} / {category}", flush=True)
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120)
    story = None
    last_error = None
    for attempt in range(1, 4):
        try:
            raw = call_llm(client, genre, category)
            story = validate(extract_json(raw), genre, category)
            break
        except Exception as exc:
            last_error = exc
            print(f"attempt {attempt} failed: {exc}", flush=True)
            if attempt < 3:
                time.sleep(5 * attempt)
    if story is None:
        sys.exit(f"story generation failed: {last_error}")
    story["created_at"] = datetime.now(timezone.utc).isoformat()
    story["model"] = MODEL
    story["total_words"] = sum(s["word_count"] for s in story["scenes"])
    (BUILD / "story.json").write_text(
        json.dumps(story, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"OK: '{story['title']}' - {len(story['scenes'])} scenes, "
        f"{story['total_words']} words",
        flush=True,
    )


if __name__ == "__main__":
    main()
