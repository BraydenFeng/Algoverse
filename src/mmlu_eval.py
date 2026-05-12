"""MMLU capability check via single-token log-probability scoring.

v2 §3 capability gate: after applying steering at α, MMLU accuracy must not
drop more than `mmlu_drop_tolerance` (default 1pt) from the unsteered baseline.
The α-sweep result is the largest α that meets this gate.

We score by comparing the log-probabilities of the four answer tokens (" A",
" B", " C", " D") at the first generation position, rather than generating and
parsing. One forward pass per question instead of multi-token decode; ~30x
faster and not subject to format-following noise.

Used by Module 2.
"""

from dataclasses import dataclass
from typing import Callable

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from .lib.config import load_config


_LETTERS = ["A", "B", "C", "D"]


@dataclass
class MMLUResult:
	accuracy: float
	n_total: int
	n_correct: int
	by_subject: dict[str, float]
	per_row: pd.DataFrame


def _build_prompt(subject: str, question: str, choices: list[str]) -> str:
	"""Standard zero-shot MMLU formatting. We append 'Answer:' so the next token's
	logits cleanly target a letter.
	"""
	subject_clean = subject.replace("_", " ")
	return (
		f"The following is a multiple choice question about {subject_clean}. "
		f"Answer with only the letter (A, B, C, or D).\n\n"
		f"Question: {question}\n"
		f"A) {choices[0]}\n"
		f"B) {choices[1]}\n"
		f"C) {choices[2]}\n"
		f"D) {choices[3]}\n"
		f"Answer:"
	)


def _letter_token_ids(tokenizer) -> list[int]:
	"""Token id for ' A', ' B', ' C', ' D' (with leading space — matches the
	'Answer:' prompt where the model's next token will naturally have one).

	Falls back to bare 'A','B','C','D' on tokenizers that split the space.
	"""
	ids = []
	for L in _LETTERS:
		tok = tokenizer(f" {L}", add_special_tokens=False).input_ids
		if len(tok) == 1:
			ids.append(tok[0])
		else:
			tok2 = tokenizer(L, add_special_tokens=False).input_ids
			if len(tok2) != 1:
				raise RuntimeError(
					f"could not get single-token id for letter {L!r}; got ' {L}' -> {tok} and "
					f"{L!r} -> {tok2}. The tokenizer is splitting letters across tokens; switch "
					f"to generation-and-parse scoring."
				)
			ids.append(tok2[0])
	return ids


def _stratified_sample(ds, n_per_subject: int, *, seed: int = 0):
	"""Take n_per_subject rows from each MMLU subject (deterministic via shuffle+seed)."""
	df = ds.to_pandas()
	out = (
		df.groupby("subject", group_keys=False)
		.apply(lambda g: g.sample(min(n_per_subject, len(g)), random_state=seed))
		.reset_index(drop=True)
	)
	return out


def run_mmlu(
	model,
	tokenizer,
	*,
	pre_forward_hook: Callable | None = None,
	n_per_subject: int | None = None,
	seed: int = 0,
) -> MMLUResult:
	"""Score MMLU on `model`, optionally with a steering hook applied.

	Args:
		pre_forward_hook: same shape as faitheval_eval — callable(model) -> handle.
			Used by Module 2 to inject steering at each α.
		n_per_subject: stratified sample size per subject. Defaults to
			config.mmlu.samples_per_subject (20 in v2).
		seed: stratified-sample seed.

	Returns an MMLUResult with overall accuracy, per-subject accuracy, and the
	full per-row DataFrame (predicted letter, gold letter, log-prob per option).
	"""
	cfg = load_config()
	if n_per_subject is None:
		n_per_subject = cfg["mmlu"]["samples_per_subject"]
	ds = load_dataset(cfg["mmlu"]["hf_dataset"], "all", split="test")
	df = _stratified_sample(ds, n_per_subject, seed=seed)

	letter_ids = _letter_token_ids(tokenizer)
	letter_tensor = torch.tensor(letter_ids, device=model.device)

	hook_handle = None
	if pre_forward_hook is not None:
		hook_handle = pre_forward_hook(model)

	records = []
	try:
		for _, row in tqdm(df.iterrows(), total=len(df), desc="MMLU"):
			prompt = _build_prompt(row["subject"], row["question"], list(row["choices"]))
			inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
			with torch.no_grad():
				out = model(**inputs)
			# logits at the last position predict the next token
			logits = out.logits[0, -1, :].float()
			letter_logits = logits[letter_tensor]  # (4,)
			pred_idx = int(letter_logits.argmax().cpu())
			records.append({
				"subject": row["subject"],
				"gold": _LETTERS[int(row["answer"])],
				"pred": _LETTERS[pred_idx],
				"correct": pred_idx == int(row["answer"]),
				**{f"logit_{L}": float(letter_logits[i].cpu()) for i, L in enumerate(_LETTERS)},
			})
	finally:
		if hook_handle is not None:
			hook_handle.remove()

	out_df = pd.DataFrame(records)
	by_subject = out_df.groupby("subject")["correct"].mean().to_dict()
	accuracy = float(out_df["correct"].mean()) if len(out_df) else 0.0
	return MMLUResult(
		accuracy=accuracy,
		n_total=len(out_df),
		n_correct=int(out_df["correct"].sum()),
		by_subject=by_subject,
		per_row=out_df,
	)
