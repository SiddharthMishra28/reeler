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
    "gripping micro-stories narrated by a calm AI voice over still artwork. "
    "You always respond with valid JSON only. "
    "Do not think step by step. Do not reason or plan silently. "
    "Answer immediately with the JSON object."
)

SETUP_PROMPT = """Plan a mini movie told in 10 scenes for a 5-minute vertical reel.

Genre: {genre}
Category: {category}

Respond with ONLY one valid JSON object:
{{"title": string, "logline": string, "characters": [{{"name": string, "visual": string}}], "world": string, "palette": string}}

- title: punchy, under 8 words
- logline: one sentence under 30 words
- characters: 1-2 named characters, each "visual" a concrete 8-14 word physical description of visible features only
- world: concrete 8-14 word description of the recurring main setting
- palette: 3-6 literal color names"""

BEATS_PROMPT = """Write the 10 scene beats for this mini movie.

Title: {title}
Logline: {logline}
Genre: {genre} / {category}
World: {world}
Characters: {characters}

Respond with ONLY one valid JSON object:
{{"beats": [string]}}

"beats": exactly 10 strings, one per scene, each 10-15 words.
- Beats 1-3: cold-open hook, introduce hero, inciting incident.
- Beats 4-7: escalation, midpoint reversal at 5, all-is-lost at 7.
- Beats 8-10: climax, resolution, payoff line calling back to the opening hook."""

SCENE_PROMPT = """Write scene {i} of 10 of this mini movie.

Title: {title}
Genre: {genre} / {category}
World: {world}
Characters: {characters}
Beat for this scene: {beat}
{previous_context}
Respond with ONLY one valid JSON object:
{{"summary": string, "narration": string, "image_prompt": string}}

- summary: 12-20 words telling what happens and how the scene ends.
- narration: 72-80 words of flowing spoken narration. Short punchy sentences, concrete sensory details, present tense. No scene numbers, no headings, no dialogue tags.
- image_prompt: reuse the main character's visual description and the world description verbatim, then add this scene's single most important visual action. Under 35 words total. Concrete nouns and actions only. No mood words, no close-up faces, no text or letters.
{ending_rule}"""


def extract_json(raw):
    text = (raw or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in response")
    return json.loads(text[start : end + 1])


def clean(s):
    return " ".join(str(s or "").split())


def call(client, prompt, max_tokens):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.85,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def call_with_retry(client, prompt, budgets=(2500, 4000, 6000, 10000)):
    """Retry with escalating token budgets: the router model sometimes burns
    its whole budget on hidden reasoning; a larger budget lets the response
    finish and emit content."""
    last = None
    for attempt, budget in enumerate(budgets, 1):
        try:
            return extract_json(call(client, prompt, budget))
        except Exception as exc:
            last = exc
            print(f"  attempt {attempt} (budget {budget}) failed: {exc}", flush=True)
            if attempt < len(budgets):
                time.sleep(4 * attempt)
    raise last


def ending_rule(i):
    if i == 1:
        return "- The narration opens with a direct-address hook that makes people stop scrolling."
    if i == 10:
        return "- The narration ends with a satisfying payoff or haunting final line that calls back to the scene 1 hook."
    return "- The narration ends on a mini-cliffhanger or open question pulling the viewer to the next scene."


def validate_setup(data):
    if not isinstance(data, dict):
        raise ValueError("setup is not a JSON object")
    chars_raw = data.get("characters") or []
    characters = []
    if isinstance(chars_raw, list):
        for ch in chars_raw[:2]:
            if isinstance(ch, dict) and clean(ch.get("name")) and clean(ch.get("visual")):
                characters.append(
                    {"name": clean(ch["name"])[:40], "visual": clean(ch["visual"])[:160]}
                )
    if not characters:
        raise ValueError("need at least 1 character")
    if not clean(data.get("world")):
        raise ValueError("missing world")
    return {
        "title": clean(data.get("title"))[:120] or "Untitled Reel",
        "logline": clean(data.get("logline"))[:300],
        "characters": characters,
        "world": clean(data.get("world"))[:160],
        "palette": clean(data.get("palette"))[:80],
    }


def validate_beats(data):
    if not isinstance(data, dict):
        raise ValueError("beats response is not a JSON object")
    beats = data.get("beats")
    if not isinstance(beats, list) or len(beats) != 10:
        raise ValueError("need exactly 10 beats")
    beats = [clean(b)[:200] for b in beats]
    if any(len(b) < 20 for b in beats):
        raise ValueError("a beat is too short")
    return beats


def validate_scene(data, setup):
    if not isinstance(data, dict):
        raise ValueError("scene response is not a JSON object")
    summary = clean(data.get("summary"))
    narration = clean(data.get("narration"))
    image_prompt = clean(data.get("image_prompt"))
    if len(summary) < 30:
        raise ValueError("scene summary too short")
    words = narration.split()
    if not (55 <= len(words) <= 110):
        raise ValueError(f"narration length {len(words)} words outside 55-110")
    if not image_prompt:
        raise ValueError("missing image_prompt")
    # Visual bible enforcement: main character's look + world must be in prompt.
    hero_words = setup["characters"][0]["visual"].split()
    key = " ".join(hero_words[:3]).lower()
    if key not in image_prompt.lower():
        image_prompt = f"{setup['characters'][0]['visual']}, {image_prompt}"
    world_key = " ".join(setup["world"].split()[:3]).lower()
    if world_key not in image_prompt.lower():
        image_prompt = f"{image_prompt}, {setup['world']}"
    return {
        "summary": summary[:200],
        "text": narration,
        "image_prompt": image_prompt[:400],
        "word_count": len(words),
    }


def main():
    if not API_KEY:
        sys.exit("BYNARA_API_KEY is not set")
    BUILD.mkdir(parents=True, exist_ok=True)
    genre = GENRE_ENV or random.choice(GENRES)
    category = CATEGORY_ENV or random.choice(CATEGORIES)
    print(f"Selected combo: {genre} / {category}", flush=True)
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=300)

    print("Phase 1: title, world, characters...", flush=True)
    setup = validate_setup(
        call_with_retry(
            client, SETUP_PROMPT.format(genre=genre, category=category)
        )
    )
    print(f"  setup OK: '{setup['title']}'", flush=True)
    print(f"  world: {setup['world']}", flush=True)

    characters_str = "; ".join(
        f"{c['name']} ({c['visual']})" for c in setup["characters"]
    )
    print("Phase 1b: 10 beats...", flush=True)
    beats = validate_beats(
        call_with_retry(
            client,
            BEATS_PROMPT.format(
                title=setup["title"],
                logline=setup["logline"],
                genre=genre,
                category=category,
                world=setup["world"],
                characters=characters_str,
            ),
        )
    )
    print(f"  beats OK: {len(beats)}", flush=True)

    scenes = []
    for i, beat in enumerate(beats, 1):
        print(f"Phase 2: scene {i}...", flush=True)
        if i == 1:
            previous_context = "This is the opening scene."
        else:
            previous_context = f"Previous scene ended: {scenes[-1]['summary']}"
        prompt = SCENE_PROMPT.format(
            i=i,
            title=setup["title"],
            genre=genre,
            category=category,
            world=setup["world"],
            characters=characters_str,
            beat=beat,
            previous_context=previous_context,
            ending_rule=ending_rule(i),
        )
        scene = validate_scene(call_with_retry(client, prompt), setup)
        scenes.append(scene)
        print(f"  scene {i} OK: {scene['word_count']} words", flush=True)

    total_words = sum(s["word_count"] for s in scenes)
    story = {
        "title": setup["title"],
        "logline": setup["logline"],
        "genre": genre,
        "category": category,
        "characters": setup["characters"],
        "world": setup["world"],
        "palette": setup["palette"],
        "scenes": scenes,
        "total_words": total_words,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
    }
    (BUILD / "story.json").write_text(
        json.dumps(story, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"OK: '{story['title']}' - {len(scenes)} scenes, {total_words} words",
        flush=True,
    )


if __name__ == "__main__":
    main()
