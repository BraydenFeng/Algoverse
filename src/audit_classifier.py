"""Hand-label audit harness for the FaithEval classifier.

HUMAN-OWNED per project CLAUDE.md: this module builds the audit sample and
scores agreement. It does NOT decide whether to retune the judge prompt or
switch Haiku 4.5 -> Sonnet 4.6 — that call is Brayden's, made after reading
the disagreement breakdown this produces.

Workflow:
	1. build_audit_sample()  -> writes audit_sample.csv with a blank
	   `human_label` column, stratified across labels and arms.
	2. (human) fill `human_label` with refuses|fabricates|off_topic, blind to
	   the classifier_label column (hide/ignore it while labelling).
	3. score_audit()         -> agreement %, per-method (rule vs judge)
	   breakdown, confusion matrix. Flags if agreement < 0.85.
"""

from pathlib import Path

import pandas as pd


_VALID = {"refuses", "fabricates", "off_topic"}
_AGREEMENT_FLOOR = 0.85  # CLAUDE.md threshold; below this -> surface to Brayden


def _load_all_arms(m3_dir: Path) -> pd.DataFrame:
	"""Concatenate every faitheval_*.csv (one row per prompt) with an arm tag."""
	frames = []
	for path in sorted(m3_dir.glob("faitheval_*.csv")):
		stem = path.stem
		if stem.startswith("faitheval_steered_a"):
			arm = f"steered@{stem.removeprefix('faitheval_steered_a')}"
		elif stem == "faitheval_baseline":
			arm = "baseline"
		elif stem == "faitheval_ablated":
			arm = "ablated"
		else:
			continue
		df = pd.read_csv(path)
		df["arm"] = arm
		frames.append(df)
	if not frames:
		raise FileNotFoundError(
			f"no faitheval_*.csv in {m3_dir}; pull m3_results/ from the HF artifact repo first"
		)
	return pd.concat(frames, ignore_index=True)


def build_audit_sample(
	m3_dir: str | Path,
	*,
	n: int = 100,
	seed: int = 0,
	judge_min: int = 0,
	out_path: str | Path = "audit_sample.csv",
) -> Path:
	"""Write a stratified ~n-row sample for blind hand-labelling.

	Stratifies across (classifier label x arm) so rare labels and the steered
	arms are not swamped by the dominant baseline-refuses cell. The output CSV
	deliberately puts `human_label` first and `classifier_label`/`method` last
	so the labeller can hide the latter columns while working.

	judge_min: guarantee at least this many method=="judge" rows in the sample.
		The rule stage is near-tautological to hand-validate (a human applies
		the same heuristic), so the load-bearing subset is the judge rows. The
		default uniform sample only caught 3 judge rows in 100; set e.g.
		judge_min=40 to force a judge-stratified audit that can actually bound
		the arm-correlation of the judge's failure mode.
	"""
	m3_dir = Path(m3_dir)
	full = _load_all_arms(m3_dir)

	groups = list(full.groupby(["label", "arm"]))
	per_cell = max(1, n // max(1, len(groups)))
	# iterate-and-concat preserves ALL columns including the groupby keys;
	# groupby.apply(include_groups=False) silently drops label/arm — do not use it here
	parts = [g.sample(min(per_cell, len(g)), random_state=seed) for _, g in groups]
	sample = pd.concat(parts, ignore_index=True)

	# force judge-method coverage: the rule subset is tautological to audit;
	# the judge subset is where independent human judgement actually tests the
	# classifier, and a uniform sample barely hits it
	if judge_min > 0:
		have = (sample["method"] == "judge").sum() if "method" in sample.columns else 0
		if have < judge_min:
			seen = set(zip(sample["qid"], sample["arm"]))
			judge_pool = full[
				(full["method"] == "judge")
				& ~full[["qid", "arm"]].apply(tuple, axis=1).isin(seen)
			]
			take = min(judge_min - have, len(judge_pool))
			if take > 0:
				sample = pd.concat(
					[sample, judge_pool.sample(take, random_state=seed)],
					ignore_index=True,
				)

	# top up to n if integer division left us short (dedupe by qid+arm,
	# not row index — concat above reset the index)
	if len(sample) < n:
		seen = set(zip(sample["qid"], sample["arm"]))
		remaining = full[~full[["qid", "arm"]].apply(tuple, axis=1).isin(seen)]
		if len(remaining):
			sample = pd.concat(
				[sample, remaining.sample(min(n - len(sample), len(remaining)), random_state=seed)],
				ignore_index=True,
			)

	sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle order
	out = pd.DataFrame({
		"qid": sample["qid"],
		"arm": sample["arm"],
		"question": sample["question"],
		"context": sample["context"].astype(str).str.slice(0, 600),
		"output": sample["output"],
		"human_label": "",  # <-- fill this in, blind to the columns below
		"classifier_label": sample["label"],
		"classifier_method": sample["method"],
	})
	out_path = Path(out_path)
	out.to_csv(out_path, index=False)
	print(f"[audit] wrote {len(out)} rows to {out_path}")
	print("[audit] fill the `human_label` column with refuses|fabricates|off_topic,")
	print("[audit] ignoring classifier_label/classifier_method while you label.")
	return out_path


def score_audit(filled_path: str | Path) -> dict:
	"""Compare human_label vs classifier_label; report agreement + breakdown.

	Surfaces, but does not act on, sub-threshold agreement. Per CLAUDE.md the
	remediation choice (retune prompt vs swap judge model) is Brayden's.
	"""
	df = pd.read_csv(filled_path)
	df["human_label"] = df["human_label"].astype(str).str.strip().str.lower()

	labelled = df[df["human_label"].isin(_VALID)].copy()
	n_missing = len(df) - len(labelled)
	if len(labelled) == 0:
		raise ValueError("no rows with a valid human_label; fill the column first")

	labelled["agree"] = labelled["human_label"] == labelled["classifier_label"]
	overall = float(labelled["agree"].mean())

	by_method = labelled.groupby("classifier_method")["agree"].agg(["mean", "count"]).to_dict("index")
	confusion = (
		labelled.groupby(["classifier_label", "human_label"]).size().unstack(fill_value=0)
	)

	# judge-subset agreement broken down by arm — this is the load-bearing
	# diagnostic: if the judge's failure rate is arm-correlated, the M3
	# fabricate/refuse DELTAS (not just absolute levels) are biased
	judge = labelled[labelled["classifier_method"] == "judge"]
	judge_by_arm = (
		judge.groupby("arm")["agree"].agg(["mean", "count"]).to_dict("index")
		if len(judge) else {}
	)

	print(f"[audit] labelled {len(labelled)} rows ({n_missing} unlabelled/skipped)")
	print(f"[audit] overall agreement: {overall:.3f}  (floor {_AGREEMENT_FLOOR})")
	print("[audit] NOTE: overall is inflated by the tautological rule subset; the")
	print("[audit]       judge subset below is the number that actually validates.")
	print("[audit] agreement by classifier method:")
	for method, stats in by_method.items():
		print(f"          {method:6s}  {stats['mean']:.3f}  (n={int(stats['count'])})")
	print(f"[audit] judge-subset agreement by arm (n_judge={len(judge)}):")
	if judge_by_arm:
		for arm, stats in sorted(judge_by_arm.items()):
			print(f"          {arm:14s}  {stats['mean']:.3f}  (n={int(stats['count'])})")
		spread = max(s["mean"] for s in judge_by_arm.values()) - min(
			s["mean"] for s in judge_by_arm.values()
		)
		print(f"          arm spread = {spread:.3f}  (large spread => judge error is "
		      f"arm-correlated => M3 deltas biased, not just offset)")
	else:
		print("          (no judge rows in sample — re-run build_audit_sample with judge_min)")
	print("[audit] confusion (rows=classifier, cols=human):")
	print(confusion.to_string())

	if overall < _AGREEMENT_FLOOR:
		print(
			f"\n[audit] *** agreement {overall:.3f} < {_AGREEMENT_FLOOR} — "
			f"SURFACE TO BRAYDEN. Do NOT auto-swap the judge model. "
			f"Decision (retune classifier.py rule/prompt vs Haiku->Sonnet) is human-owned."
		)
	else:
		print(f"\n[audit] agreement >= {_AGREEMENT_FLOOR} — M3 classifier rates are trustworthy for the writeup.")

	return {
		"overall_agreement": overall,
		"n_labelled": len(labelled),
		"n_missing": n_missing,
		"by_method": by_method,
		"judge_by_arm": judge_by_arm,
		"n_judge": len(judge),
		"confusion": confusion.to_dict(),
		"below_floor": overall < _AGREEMENT_FLOOR,
	}
