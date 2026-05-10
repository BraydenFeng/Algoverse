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


# neutral corpus uses descriptive, low-affect topics — these are what Anthropic-style
# protocols use for the PC project-out step (the goal is to capture generic narrative
# variance, not emotional content).
_NEUTRAL_TOPICS = [
	"how a paper-clip is manufactured",
	"the process of brewing tea step by step",
	"how to pack a suitcase efficiently",
	"the layout of a typical public library",
	"how a printing press operates",
	"the parts of a standard bicycle",
	"how to file a paper document into a cabinet",
	"the procedure for changing a light bulb",
	"how rain forms in clouds",
	"how to cross a busy intersection on foot",
	"the steps to make a peanut butter sandwich",
	"how a vending machine dispenses items",
	"the contents of a typical office supply drawer",
	"how to fold a paper airplane",
	"the procedure for boarding a commercial flight",
	"the layout of a grocery store",
	"how a digital calculator performs addition",
	"the steps to wash a load of laundry",
	"the parts of a wristwatch",
	"the process of mailing a letter",
]


def _build_emotion_prompt(emotion: str, context: str, target_words: int) -> str:
	return (
		f"Write a short story of about {target_words} words. The story should be set in: "
		f"{context}. The main character should clearly experience {emotion}. "
		f"The emotion should be unmistakable by the early part of the story (within the first "
		f"50 tokens), and should drive the character's thoughts and actions throughout. "
		f"Write in third-person past tense. Do not name the emotion explicitly more than once. "
		f"Show it through specifics: thoughts, gestures, what the character notices and doesn't. "
		f"Begin the story directly; no preamble, no title."
	)


def _build_neutral_prompt(topic: str, target_words: int) -> str:
	# distinct prompt: neutral corpus is descriptive prose about everyday processes,
	# used to derive PCs of generic narrative variance for project-out. No characters,
	# no affect, no implied emotion — just descriptive text of comparable length and style.
	return (
		f"Write a {target_words}-word descriptive passage about: {topic}. "
		f"Use a neutral, encyclopedic tone. No characters, no dialogue, no emotional content. "
		f"Describe the process or object in plain factual prose. "
		f"Begin directly; no preamble, no title."
	)


def _build_prompt(emotion: str, context: str, target_words: int) -> str:
	# kept for back-compat with anything importing the underscore name; routes to the
	# correct prompt builder.
	if emotion == "neutral":
		return _build_neutral_prompt(context, target_words)
	return _build_emotion_prompt(emotion, context, target_words)


def generate_one_story(emotion: str, context: str, target_words: int = 400) -> str:
	"""Call Claude Opus 4.7 for one story. Requires ANTHROPIC_API_KEY.

	emotion='neutral' switches to the descriptive-passage prompt (no character / no affect),
	and `context` is treated as the topic of description.
	"""
	from anthropic import Anthropic

	key = os.environ.get("ANTHROPIC_API_KEY")
	if not key:
		raise RuntimeError("ANTHROPIC_API_KEY not set; cannot generate stories")

	client = Anthropic(api_key=key)
	if emotion == "neutral":
		prompt = _build_neutral_prompt(context, target_words)
	else:
		prompt = _build_emotion_prompt(emotion, context, target_words)

	resp = client.messages.create(
		model="claude-opus-4-7",
		max_tokens=1024,
		messages=[{"role": "user", "content": prompt}],
	)
	text = resp.content[0].text.strip()
	if not text:
		raise RuntimeError(f"empty completion for emotion={emotion!r} context={context!r}")
	return text


def _generate_with_retry(
	emotion: str,
	context: str,
	target_words: int,
	*,
	max_attempts: int = 4,
	base_delay: float = 5.0,
) -> str:
	"""Exponential-backoff wrapper around generate_one_story. Last failure is re-raised."""
	last_err: Exception | None = None
	for attempt in range(1, max_attempts + 1):
		try:
			return generate_one_story(emotion, context, target_words=target_words)
		except Exception as e:
			last_err = e
			if attempt == max_attempts:
				break
			delay = base_delay * (2 ** (attempt - 1))
			print(f"[gen] attempt {attempt}/{max_attempts} failed ({e}); sleeping {delay:.0f}s")
			time.sleep(delay)
	raise RuntimeError(
		f"generate_one_story persistently failed for emotion={emotion!r} context={context!r}"
	) from last_err


def generate_emotion_corpus(
	emotion: str,
	n_stories: int,
	*,
	contexts: Iterable[str] | None = None,
	target_words: int = 400,
	out_dir: Path,
	overwrite: bool = False,
	skip_failures: bool = True,
) -> list[Path]:
	"""Generate n_stories for one emotion, stratified across contexts.

	If n_stories > len(contexts), contexts are reused (different stories result anyway
	due to model sampling).

	emotion='neutral' uses the descriptive-passage prompt with _NEUTRAL_TOPICS as
	default contexts (so cell 4 of m1_extract.ipynb produces a real neutral corpus
	rather than stories about "experiencing neutral").

	When skip_failures is True, a per-story persistent failure is logged and skipped
	so a flaky API call doesn't kill a multi-hour run; failures are recorded in the
	manifest with status='failed'.
	"""
	out_dir.mkdir(parents=True, exist_ok=True)
	if contexts is None:
		contexts = _NEUTRAL_TOPICS if emotion == "neutral" else _CONTEXTS
	contexts = list(contexts)

	# resume-aware manifest: load prior entries, merge by idx
	manifest_path = out_dir / "manifest.json"
	manifest_by_idx: dict[int, dict] = {}
	if manifest_path.exists():
		try:
			prior = json.loads(manifest_path.read_text(encoding="utf-8"))
			for entry in prior:
				manifest_by_idx[entry["idx"]] = entry
		except Exception as e:
			print(f"[gen] warn: could not parse existing manifest at {manifest_path}: {e}")

	written: list[Path] = []
	for i in range(n_stories):
		story_path = out_dir / f"{i:03d}.txt"
		context = contexts[i % len(contexts)]

		if story_path.exists() and not overwrite:
			print(f"[gen] {emotion}/{i:03d} exists, skipping")
			manifest_by_idx.setdefault(
				i, {"idx": i, "context": context, "path": story_path.name, "status": "ok"}
			)
			written.append(story_path)
			continue

		try:
			story = _generate_with_retry(emotion, context, target_words)
		except Exception as e:
			msg = f"persistent failure: {e}"
			print(f"[gen] {emotion}/{i:03d} {msg}")
			manifest_by_idx[i] = {
				"idx": i,
				"context": context,
				"path": story_path.name,
				"status": "failed",
				"error": str(e),
			}
			# checkpoint manifest after every failure so partial progress is durable
			manifest_path.write_text(
				json.dumps(sorted(manifest_by_idx.values(), key=lambda r: r["idx"]), indent=2),
				encoding="utf-8",
			)
			if skip_failures:
				continue
			raise

		story_path.write_text(story, encoding="utf-8")
		manifest_by_idx[i] = {
			"idx": i,
			"context": context,
			"path": story_path.name,
			"status": "ok",
		}
		written.append(story_path)
		print(f"[gen] {emotion}/{i:03d} ({context[:40]}...) -> {len(story.split())} words")
		# checkpoint every story so a kill -9 doesn't lose the manifest
		manifest_path.write_text(
			json.dumps(sorted(manifest_by_idx.values(), key=lambda r: r["idx"]), indent=2),
			encoding="utf-8",
		)

	manifest_path.write_text(
		json.dumps(sorted(manifest_by_idx.values(), key=lambda r: r["idx"]), indent=2),
		encoding="utf-8",
	)
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
