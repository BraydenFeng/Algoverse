"""Story generation for emotion vector extraction.

v2 LaTeX §4 locks: Claude Opus 4.7, twenty stories per emotion, stratified across
narrative contexts. Each story should make the target emotion unambiguous by token 50.

Outputs to data/stories/{emotion}/{idx:03d}.txt
"""

import json
import os
import time
from pathlib import Path
from typing import Iterable

from .lib.config import load_config


_CONTEXTS = [
	"a hospital waiting room late at night",
	"a job interview that's gone badly off-script",
	"a long car ride alone",
	"a family dinner gone quiet",
	"a phone call with a friend who hasn't called in years",
	"a stage just before a performance",
	"a small apartment during a power outage",
	"a park bench in winter",
	"an empty office after everyone has left",
	"a train platform when the train is delayed indefinitely",
	"a kitchen at 3 AM",
	"a school hallway between classes",
	"a hotel room in a city the character doesn't know",
	"a hiking trail when fog rolls in",
	"a doctor's office after a routine appointment",
	"a wedding reception, between speeches",
	"a library after closing",
	"a bus station between buses",
	"a beach in the off-season",
	"a parent's house, returning for the first time in years",
]


def _build_prompt(emotion: str, context: str, target_words: int) -> str:
	return (
		f"Write a short story of about {target_words} words. The story should be set in: "
		f"{context}. The main character should clearly experience {emotion}. "
		f"The emotion should be unmistakable by the early part of the story (within the first "
		f"50 tokens), and should drive the character's thoughts and actions throughout. "
		f"Write in third-person past tense. Do not name the emotion explicitly more than once. "
		f"Show it through specifics: thoughts, gestures, what the character notices and doesn't. "
		f"Begin the story directly; no preamble, no title."
	)


def generate_one_story(emotion: str, context: str, target_words: int = 400) -> str:
	"""Call Claude Opus 4.7 for one story. Requires ANTHROPIC_API_KEY."""
	from anthropic import Anthropic

	key = os.environ.get("ANTHROPIC_API_KEY")
	if not key:
		raise RuntimeError("ANTHROPIC_API_KEY not set; cannot generate stories")

	client = Anthropic(api_key=key)
	prompt = _build_prompt(emotion, context, target_words)

	resp = client.messages.create(
		model="claude-opus-4-7",
		max_tokens=1024,
		messages=[{"role": "user", "content": prompt}],
	)
	return resp.content[0].text.strip()


def generate_emotion_corpus(
	emotion: str,
	n_stories: int,
	*,
	contexts: Iterable[str] = _CONTEXTS,
	target_words: int = 400,
	out_dir: Path,
	overwrite: bool = False,
) -> list[Path]:
	"""Generate n_stories for one emotion, stratified across contexts.

	If n_stories > len(contexts), contexts are reused (different stories result anyway
	due to model sampling).
	"""
	out_dir.mkdir(parents=True, exist_ok=True)
	contexts = list(contexts)
	manifest_path = out_dir / "manifest.json"
	manifest = []

	written: list[Path] = []
	for i in range(n_stories):
		story_path = out_dir / f"{i:03d}.txt"
		if story_path.exists() and not overwrite:
			print(f"[gen] {emotion}/{i:03d} exists, skipping")
			written.append(story_path)
			continue

		context = contexts[i % len(contexts)]
		try:
			story = generate_one_story(emotion, context, target_words=target_words)
		except Exception as e:
			print(f"[gen] failed {emotion}/{i:03d}: {e}; retrying once after 5s")
			time.sleep(5)
			story = generate_one_story(emotion, context, target_words=target_words)

		story_path.write_text(story, encoding="utf-8")
		manifest.append({"idx": i, "context": context, "path": str(story_path.name)})
		written.append(story_path)
		print(f"[gen] {emotion}/{i:03d} ({context[:40]}...) -> {len(story.split())} words")

	manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
	return written


def generate_all() -> dict[str, list[Path]]:
	"""Generate the full v2 story corpus per config.yaml."""
	cfg = load_config()
	n = cfg["extraction"]["stories_per_emotion"]
	target_words = cfg["extraction"]["story_target_word_count"]
	data_dir = Path(cfg["paths"]["data_dir"]) / "stories"

	result: dict[str, list[Path]] = {}
	for emotion in cfg["extraction"]["emotions"]:
		emotion_dir = data_dir / emotion
		result[emotion] = generate_emotion_corpus(
			emotion,
			n,
			out_dir=emotion_dir,
			target_words=target_words,
		)
	return result
