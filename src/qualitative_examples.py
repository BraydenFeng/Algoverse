"""Pull qualitative FaithEval examples from M3 per-arm result CSVs.

Mentor asked for concrete question/context/output examples alongside the
quantitative rates, especially prompts where the label *flips* between the
baseline arm and a steered arm (refuses->off_topic, refuses->fabricates) —
those are the most illustrative of the steering mechanism.

Reads the m3_results/faitheval_*.csv files (columns: qid, question, context,
output, label, method, reason). Notebook is a thin caller; logic lives here.
"""

from pathlib import Path

import pandas as pd


_ARM_FROM_STEM = {
	"faitheval_baseline": "baseline",
	"faitheval_ablated": "ablated",
}


def _arm_label(stem: str) -> str:
	"""Map a CSV filename stem to an arm label (e.g. faitheval_steered_a0.1 -> steered@0.1)."""
	if stem in _ARM_FROM_STEM:
		return _ARM_FROM_STEM[stem]
	if stem.startswith("faitheval_steered_a"):
		return f"steered@{stem.removeprefix('faitheval_steered_a')}"
	return stem


def _load_arm(m3_dir: Path, stem: str) -> pd.DataFrame:
	path = m3_dir / f"{stem}.csv"
	try:
		df = pd.read_csv(path)
	except FileNotFoundError as e:
		raise FileNotFoundError(
			f"{path} not found; pull m3_results/ from the HF artifact repo first"
		) from e
	df["arm"] = _arm_label(stem)
	return df


def _truncate(s: str, n: int) -> str:
	s = "" if s is None else str(s)
	return s if len(s) <= n else s[: n - 1] + "…"


def label_flips(
	m3_dir: str | Path,
	*,
	baseline_stem: str = "faitheval_baseline",
	compare_stem: str = "faitheval_steered_a0.1",
	per_transition: int = 4,
	seed: int = 0,
	context_chars: int = 300,
) -> pd.DataFrame:
	"""Return prompts whose classifier label changed baseline -> compare arm.

	Grouped by (baseline_label -> compare_label) transition, `per_transition`
	sampled rows each. These are the strongest qualitative evidence for what
	the steering actually does to the model's behavior.
	"""
	m3_dir = Path(m3_dir)
	base = _load_arm(m3_dir, baseline_stem)[["qid", "question", "context", "output", "label"]]
	comp = _load_arm(m3_dir, compare_stem)[["qid", "output", "label"]]

	merged = base.merge(comp, on="qid", suffixes=("_base", "_steered"))
	flipped = merged[merged["label_base"] != merged["label_steered"]].copy()
	flipped["transition"] = flipped["label_base"] + " -> " + flipped["label_steered"]

	# iterate-and-concat: preserves the `transition` column (groupby.apply
	# with include_groups=False would strip it, breaking to_markdown)
	parts = [
		g.sample(min(per_transition, len(g)), random_state=seed)
		for _, g in flipped.groupby("transition")
	]
	out = pd.concat(parts, ignore_index=True) if parts else flipped.iloc[:0].copy()
	out["context"] = out["context"].map(lambda s: _truncate(s, context_chars))
	out["output_base"] = out["output_base"].map(lambda s: _truncate(s, 400))
	out["output_steered"] = out["output_steered"].map(lambda s: _truncate(s, 400))
	return out


def to_markdown(flips: pd.DataFrame) -> str:
	"""Render label_flips() output as a readable markdown doc for the writeup appendix."""
	lines: list[str] = ["# FaithEval qualitative examples — label flips (baseline -> steered α=0.1)\n"]
	for transition, group in flips.groupby("transition"):
		lines.append(f"\n## {transition}  ({len(group)} shown)\n")
		for _, r in group.iterrows():
			lines.append(f"**qid `{r['qid']}`**")
			lines.append(f"- **Q:** {r['question']}")
			lines.append(f"- **Context:** {r['context']}")
			lines.append(f"- **Baseline output** ({r['label_base']}): {r['output_base']!r}")
			lines.append(f"- **Steered output** ({r['label_steered']}): {r['output_steered']!r}")
			lines.append("")
	return "\n".join(lines)
