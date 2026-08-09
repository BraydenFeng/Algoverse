"""M5.A specificity: the verdict thresholds decide whether the paper's title
survives, so pin them. The sweep itself needs a GPU; these cover the pure logic."""

import pandas as pd

from src.m5_affect_specificity import _alpha_tag, _verdict


def _frame(desperation, sad, angry, calm=0.5):
	rows = [
		("baseline", 0.0),
		("desperation", desperation),
		("sad", sad),
		("angry", angry),
		("calm", calm),
	]
	return pd.DataFrame(
		[
			{
				"arm": arm,
				"alpha": 0.0 if arm == "baseline" else 0.5,
				"n": 2492,
				"refusal_rate": 0.0,
				"hallucination_rate": 0.0,
				"off_topic_rate": 0.0,
				"fab_delta_pts": delta,
			}
			for arm, delta in rows
		]
	)


def test_alpha_tag_matches_m3_convention():
	assert _alpha_tag(0.5) == "a0.5"
	assert _alpha_tag(0.15) == "a0.15"
	assert _alpha_tag(1.0) == "a1"  # %g drops the trailing zero, as M3 filenames do


def test_verdict_specific_when_desperation_clears_others():
	out = _verdict(_frame(desperation=15.3, sad=4.0, angry=3.0))
	assert "desperation-specific" in out


def test_verdict_not_specific_when_negative_affect_matches():
	out = _verdict(_frame(desperation=15.3, sad=14.8, angry=13.9))
	assert "NOT desperation-specific" in out


def test_verdict_partial_when_margin_is_middling():
	out = _verdict(_frame(desperation=15.3, sad=12.0, angry=11.0))
	assert "partially specific" in out


def test_verdict_uses_worst_case_negative_arm():
	# angry nearly matches desperation -> not specific, even though sad is far behind
	out = _verdict(_frame(desperation=15.0, sad=1.0, angry=14.5))
	assert "NOT desperation-specific" in out


def test_verdict_handles_missing_baseline():
	df = _frame(desperation=15.0, sad=1.0, angry=1.0).drop(columns=["fab_delta_pts"])
	assert "deltas unavailable" in _verdict(df)
