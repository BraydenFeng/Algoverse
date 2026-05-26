"""M3 retention-conditioned rate analysis (PI ask 2026-05-25).

Vasu asked for both conditional rates on retained outputs and unconditional
rates over all 2,492 FaithEval prompts, so reviewers can tell whether
fabrication is genuinely increasing or whether the surviving outputs are a
biased subset at high α. This module reads the per-α FaithEval CSVs from
`outputs/m3/` and produces a single tidy table.

Two retention definitions are computed in parallel:
	- nonempty: model produced any text at all (output.strip() != "")
	- strict:   nonempty AND label != "off_topic" — i.e. coherent enough to be
	            classified as a refuse-or-fabricate

α-regime labels follow Vasu's framing: α ≤ 0.3 = interpretable, α ≥ 0.5 =
overshoot / collapse, baseline / ablated stand outside the sweep.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ARMS_DEFAULT = {
	"baseline": ("faitheval_baseline.csv", None),
	"ablated":  ("faitheval_ablated.csv", None),
}

# steered arms — α values from the full sweep (M2-validated + extended + very-high)
STEERED_ALPHAS = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]


def alpha_regime(arm: str, alpha: float | None) -> str:
	"""Vasu's framing: α ≤ 0.3 interpretable, α ≥ 0.5 overshoot/collapse."""
	if arm == "baseline":
		return "baseline"
	if arm == "ablated":
		return "ablated"
	if alpha is None:
		return "unknown"
	if alpha <= 0.3:
		return "interpretable"
	if alpha >= 0.5:
		return "overshoot"
	return "transition"


@dataclass
class ArmRates:
	arm: str
	alpha: float | None
	regime: str
	n_total: int
	n_nonempty: int
	n_strict: int  # nonempty AND not off_topic
	# unconditional (denom = n_total)
	refuse_uncond: float
	fab_uncond: float
	off_topic_uncond: float
	empty_uncond: float
	# conditional on nonempty (denom = n_nonempty)
	refuse_cond_ne: float
	fab_cond_ne: float
	off_topic_cond_ne: float
	# conditional on strict retention (denom = n_strict; off_topic excluded)
	refuse_cond_strict: float
	fab_cond_strict: float
	# retention indicators
	retention_nonempty: float
	retention_strict: float


def _safe_div(num: int, denom: int) -> float:
	return float(num) / denom if denom > 0 else 0.0


def compute_arm_rates(df: pd.DataFrame, arm: str, alpha: float | None) -> ArmRates:
	"""Compute conditional + unconditional rates for a single arm's CSV."""
	df = df.copy()
	df["output"] = df["output"].fillna("").astype(str)

	n_total = len(df)
	nonempty_mask = df["output"].str.strip() != ""
	n_nonempty = int(nonempty_mask.sum())

	# strict retention = nonempty AND label in {refuses, fabricates}
	strict_mask = nonempty_mask & df["label"].isin(["refuses", "fabricates"])
	n_strict = int(strict_mask.sum())

	n_refuse = int((df["label"] == "refuses").sum())
	n_fab = int((df["label"] == "fabricates").sum())
	n_off = int((df["label"] == "off_topic").sum())
	n_empty = n_total - n_nonempty

	# nonempty-restricted counts
	ne_refuse = int(((df["label"] == "refuses") & nonempty_mask).sum())
	ne_fab = int(((df["label"] == "fabricates") & nonempty_mask).sum())
	ne_off = int(((df["label"] == "off_topic") & nonempty_mask).sum())

	# strict-restricted counts (off_topic = 0 by construction)
	st_refuse = int(((df["label"] == "refuses") & strict_mask).sum())
	st_fab = int(((df["label"] == "fabricates") & strict_mask).sum())

	return ArmRates(
		arm=arm,
		alpha=alpha,
		regime=alpha_regime(arm, alpha),
		n_total=n_total,
		n_nonempty=n_nonempty,
		n_strict=n_strict,
		refuse_uncond=_safe_div(n_refuse, n_total),
		fab_uncond=_safe_div(n_fab, n_total),
		off_topic_uncond=_safe_div(n_off, n_total),
		empty_uncond=_safe_div(n_empty, n_total),
		refuse_cond_ne=_safe_div(ne_refuse, n_nonempty),
		fab_cond_ne=_safe_div(ne_fab, n_nonempty),
		off_topic_cond_ne=_safe_div(ne_off, n_nonempty),
		refuse_cond_strict=_safe_div(st_refuse, n_strict),
		fab_cond_strict=_safe_div(st_fab, n_strict),
		retention_nonempty=_safe_div(n_nonempty, n_total),
		retention_strict=_safe_div(n_strict, n_total),
	)


def build_retention_table(
	csv_dir: Path,
	steered_alphas: list[float] = STEERED_ALPHAS,
	min_n_total: int = 2000,
) -> pd.DataFrame:
	"""Scan csv_dir for baseline / steered / ablated CSVs and build the table.

	Missing arms are silently skipped — the table reports whatever ran.
	Arms with n_total < min_n_total are skipped with a stderr note so partial
	runs (Colab session death mid-arm) don't get mixed in with completed ones
	at non-comparable denominators. Default 2000 leaves headroom under the
	full 2,492 FaithEval prompts.
	"""
	import sys

	rows: list[ArmRates] = []

	def _maybe_append(df: pd.DataFrame, arm: str, alpha: float | None) -> None:
		r = compute_arm_rates(df, arm, alpha)
		if r.n_total < min_n_total:
			label = f"{arm} α={alpha}" if alpha is not None else arm
			print(
				f"[m3_retention] skipping {label}: n_total={r.n_total} < {min_n_total} "
				"(partial run; complete the arm before including it)",
				file=sys.stderr,
			)
			return
		rows.append(r)

	baseline_path = csv_dir / "faitheval_baseline.csv"
	if baseline_path.exists():
		_maybe_append(pd.read_csv(baseline_path), "baseline", None)

	for alpha in steered_alphas:
		path = csv_dir / f"faitheval_steered_a{alpha}.csv"
		if path.exists():
			_maybe_append(pd.read_csv(path), "steered", alpha)

	ablated_path = csv_dir / "faitheval_ablated.csv"
	if ablated_path.exists():
		_maybe_append(pd.read_csv(ablated_path), "ablated", None)

	return pd.DataFrame([r.__dict__ for r in rows])


def format_table(df: pd.DataFrame) -> str:
	"""Pretty-print the retention table for the decision file."""
	display_cols = [
		"arm", "alpha", "regime",
		"n_total", "n_nonempty", "retention_nonempty", "retention_strict",
		"refuse_uncond", "fab_uncond",
		"refuse_cond_ne", "fab_cond_ne",
		"refuse_cond_strict", "fab_cond_strict",
	]
	view = df[display_cols].copy()
	for c in display_cols:
		if view[c].dtype == float:
			view[c] = view[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
	return view.to_string(index=False)


if __name__ == "__main__":
	# CLI usage: python -m src.m3_retention [csv_dir]
	import sys

	csv_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/m3")
	if not csv_dir.exists():
		raise SystemExit(f"csv_dir not found: {csv_dir}")

	table = build_retention_table(csv_dir)
	if len(table) == 0:
		raise SystemExit(f"no FaithEval CSVs found in {csv_dir}")

	out_csv = csv_dir / "retention_table.csv"
	table.to_csv(out_csv, index=False)
	print(format_table(table))
	print(f"\nwrote {out_csv}")
