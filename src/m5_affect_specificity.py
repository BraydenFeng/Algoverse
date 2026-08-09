"""M5.A — is the refusal→fabrication effect specific to desperation?

Both COLM reviewers and the internal council flagged the same hole: the
direction-specificity evidence rests on a single calm control at α=0.3. Calm is
a low-arousal, positive-valence control, so it cannot separate "desperation"
from "negative affect in general". If sad and angry produce the same shift at
the headline α, the affective framing in the title is overclaiming and the
paper should be reframed around negative affect.

This runs the full FaithEval-unanswerable behavioral sweep at the headline α for
every extracted emotion, so all four arms are directly comparable to the
existing desperation number.

Design decisions:
	- Same α (headline, default 0.5) for every emotion. Vectors are unit-norm and
	  α scales by the same mean residual norm, so the perturbation magnitude is
	  matched across emotions — that is the whole point of the control.
	- norm_scale is estimated ONCE and reused for all emotions. Re-estimating per
	  emotion would introduce a nuisance difference between arms.
	- Every arm checkpoints to HF after each interval, so a Colab disconnect
	  costs at most `checkpoint_every` prompts.

Outputs land at outputs/m5_specificity/{lsuf}_a{alpha}/ and are mirrored to
the HF artifact repo under m5_specificity/{lsuf}_a{alpha}/.
"""

from pathlib import Path

import pandas as pd

from .faitheval_eval import run_eval, summary
from .lib.config import layer_suffix, load_config
from .steering import (
	estimate_residual_norm,
	load_emotion_vector,
	make_steering_hook_factory,
)

# calm is the published control; sad and angry are the reviewer-requested additions.
# desperation is re-run rather than reused so every arm shares one norm_scale and
# one code path — a reviewer comparing arms should not have to trust cross-run parity.
EMOTIONS = ("desperation", "sad", "angry", "calm")


def _alpha_tag(alpha: float) -> str:
	"""Filesystem-safe alpha tag matching the M3 convention (a0.5, a0.3)."""
	return f"a{alpha:g}"


def _neutral_corpus(cfg: dict) -> list[str]:
	"""Neutral stories used to calibrate the residual norm."""
	data_dir = Path(cfg["paths"]["data_dir"]) / "stories" / "neutral"
	paths = sorted(data_dir.glob("*.txt"))
	if not paths:
		raise FileNotFoundError(
			f"no neutral stories at {data_dir}; norm_scale cannot be estimated. "
			"Check that data/stories/neutral/ is present in the repo checkout."
		)
	texts = []
	for p in paths:
		try:
			texts.append(p.read_text(encoding="utf-8"))
		except OSError as e:
			print(f"[m5a] skipping unreadable {p.name}: {e}")
	return texts


def run_specificity_sweep(
	model,
	tokenizer,
	*,
	alpha: float = 0.5,
	model_key: str = "primary",
	emotions: tuple[str, ...] = EMOTIONS,
	limit: int | None = None,
	batch_size: int = 8,
	checkpoint_every: int = 200,
	baseline: bool = True,
) -> pd.DataFrame:
	"""Run FaithEval at `alpha` for each emotion; return a per-arm summary frame.

	Args:
		alpha: steering coefficient, shared across every emotion arm. Default 0.5
			is the paper's headline (capability-preserving peak in Gemma).
		model_key: "primary" (Gemma) or "llama"; selects layer + vector directory.
		emotions: arms to run. Defaults to all four extracted directions.
		limit: cap prompts (None = full 2,492 set). Use a small value to smoke-test.
		baseline: also run an unsteered arm, so the sweep is self-contained rather
			than depending on the M3 baseline having used identical settings.

	Each arm's per-prompt CSV is checkpointed locally and mirrored to HF.
	"""
	cfg = load_config()
	lsuf = layer_suffix(cfg, model_key)
	layer = cfg["models"][model_key]["extraction_layer"]
	use_chat_template = bool(cfg["models"][model_key].get("uses_chat_template", False))
	repo = cfg["paths"]["hf_artifact_repo"]

	run_tag = f"{lsuf}_{_alpha_tag(alpha)}"
	out_dir = Path(cfg["paths"]["outputs_dir"]) / "m5_specificity" / run_tag
	out_dir.mkdir(parents=True, exist_ok=True)
	repo_dir = f"m5_specificity/{run_tag}"

	norm_scale = estimate_residual_norm(
		model,
		tokenizer,
		layer=layer,
		calibration_texts=_neutral_corpus(cfg),
		token_skip=cfg["extraction"]["token_skip"],
	)
	print(f"[m5a] {model_key} L{layer}  norm_scale={norm_scale:.2f}  alpha={alpha}")

	arms: list[tuple[str, object]] = []
	if baseline:
		arms.append(("baseline", None))
	for emotion in emotions:
		vec = load_emotion_vector(emotion, model_key=model_key)
		hook = make_steering_hook_factory(
			vector=vec, layer=layer, alpha=alpha, norm_scale=norm_scale
		)
		arms.append((emotion, hook))

	rows = []
	for name, hook in arms:
		print(f"\n[m5a] === arm: {name} ===")
		df = run_eval(
			model,
			tokenizer,
			limit=limit,
			pre_forward_hook=hook,
			checkpoint_path=out_dir / f"{name}.csv",
			checkpoint_every=checkpoint_every,
			hf_sync_repo=repo,
			hf_sync_path=f"{repo_dir}/{name}.csv",
			batch_size=batch_size,
			use_chat_template=use_chat_template,
		)
		s = summary(df)
		s["arm"] = name
		s["alpha"] = 0.0 if name == "baseline" else alpha
		rows.append(s)
		print(f"[m5a] {name}: {s}")

	comparison = pd.DataFrame(rows)[
		["arm", "alpha", "n", "refusal_rate", "hallucination_rate", "off_topic_rate"]
	]

	# effect vs the unsteered arm, which is the number the paper actually compares
	if baseline:
		base_fab = float(
			comparison.loc[comparison["arm"] == "baseline", "hallucination_rate"].iloc[0]
		)
		comparison["fab_delta_pts"] = (comparison["hallucination_rate"] - base_fab) * 100

	comp_path = out_dir / "comparison.csv"
	comparison.to_csv(comp_path, index=False)
	_upload(comp_path, repo, f"{repo_dir}/comparison.csv")

	verdict = _verdict(comparison)
	verdict_path = out_dir / "verdict.txt"
	verdict_path.write_text(verdict, encoding="utf-8")
	_upload(verdict_path, repo, f"{repo_dir}/verdict.txt")

	print("\n" + verdict)
	return comparison


def _verdict(comparison: pd.DataFrame) -> str:
	"""Report the specificity gaps and the suggested framing tag.

	Interpretation is human-owned per CLAUDE.md — this prints the numbers and a
	suggested reading, it does not decide the paper's framing.
	"""
	lines = ["M5.A — affective specificity", "=" * 44, ""]
	lines.append(comparison.to_string(index=False))
	lines.append("")

	if "fab_delta_pts" not in comparison.columns:
		lines.append("no baseline arm; deltas unavailable")
		return "\n".join(lines)

	by_arm = dict(zip(comparison["arm"], comparison["fab_delta_pts"]))
	desp = by_arm.get("desperation")
	negatives = [by_arm[e] for e in ("sad", "angry") if e in by_arm]
	if desp is None or not negatives:
		lines.append("desperation and/or negative-affect arms missing; no verdict")
		return "\n".join(lines)

	worst_negative = max(negatives)
	margin = desp - worst_negative
	lines += [
		f"desperation effect:            {desp:+.1f} pts",
		f"largest other negative affect: {worst_negative:+.1f} pts",
		f"margin:                        {margin:+.1f} pts",
		"",
	]
	if margin >= 5.0:
		lines.append("SUGGESTED: desperation-specific — the affective framing holds.")
	elif margin >= 2.0:
		lines.append("SUGGESTED: partially specific — desperation leads, but report all arms.")
	else:
		lines.append(
			"SUGGESTED: NOT desperation-specific — negative affect broadly drives the "
			"shift. Reframe the title/abstract away from 'desperation'."
		)
	lines.append("Framing call is Brayden's; this is a suggestion, not a decision.")
	return "\n".join(lines)


def _upload(local_path: Path, repo_id: str, path_in_repo: str) -> None:
	try:
		from huggingface_hub import upload_file

		upload_file(
			path_or_fileobj=str(local_path),
			path_in_repo=path_in_repo,
			repo_id=repo_id,
			repo_type="dataset",
		)
		print(f"[m5a] synced {path_in_repo}")
	except Exception as e:
		print(f"[m5a] HF sync of {path_in_repo} failed (local copy kept): {e}")
