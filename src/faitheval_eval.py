"""FaithEval-unanswerable evaluation driver.

Loads Salesforce/FaithEval-unanswerable-v1.0, runs greedy decoding through a
loaded model, classifies outputs, returns metrics and per-prompt records.

Used by:
	- Module 0.0 (baseline refusal rate; viability gate)
	- Module 2.B (steered+ablated hallucination rate)
	- Module 2.C (steered+enhanced hallucination rate)
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from .lib.classifier import classify
from .lib.config import load_config


@dataclass
class EvalRecord:
	qid: str
	question: str
	context: str
	output: str
	label: str
	method: str
	reason: str


def _build_prompt(template: str, context: str, question: str) -> str:
	return template.format(context=context, question=question)


def _greedy_generate(
	model,
	tokenizer,
	prompt: str,
	max_new_tokens: int = 128,
	pre_forward_hook: Callable | None = None,
) -> str:
	"""Greedy decode. pre_forward_hook lets M2.B/M2.C inject steering or ablation."""
	inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

	hook_handle = None
	if pre_forward_hook is not None:
		hook_handle = pre_forward_hook(model)

	try:
		with torch.no_grad():
			out = model.generate(
				**inputs,
				max_new_tokens=max_new_tokens,
				do_sample=False,
				temperature=1.0,
				pad_token_id=tokenizer.eos_token_id,
			)
	finally:
		if hook_handle is not None:
			hook_handle.remove()

	# strip the prompt prefix; return only the model's continuation
	generated = out[0][inputs["input_ids"].shape[1] :]
	return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_eval(
	model,
	tokenizer,
	*,
	limit: int | None = None,
	pre_forward_hook: Callable | None = None,
	checkpoint_path: Path | None = None,
	checkpoint_every: int = 100,
	force_judge: bool = False,
) -> pd.DataFrame:
	"""Run FaithEval-unanswerable through `model`.

	Args:
		model, tokenizer: loaded Gemma model + tokenizer.
		limit: cap on prompts processed (use for smoke tests; None = all 2,492).
		pre_forward_hook: callable(model) -> hook_handle; used to inject steering
			or ablation in M2.B/M2.C. None for baseline M0.0.
		checkpoint_path: write incremental CSV here every checkpoint_every prompts.
		checkpoint_every: checkpoint interval.
		force_judge: skip the regex rule pass and send every output to the Claude
			judge. Use for diagnosis when refusal_rate=1.0 or 0.0 looks suspicious
			(see lib/classifier.classify docstring). Costs N judge calls.

	Returns DataFrame indexed by qid with classification + raw output.
	"""
	cfg = load_config()
	ds = load_dataset(cfg["faitheval"]["hf_dataset"], split="test")
	if limit is not None:
		ds = ds.select(range(min(limit, len(ds))))

	template = cfg["faitheval"]["prompt_template"]

	records: list[EvalRecord] = []
	resume_qids: set[str] = set()

	# resume from checkpoint if present
	if checkpoint_path is not None and checkpoint_path.exists():
		prev = pd.read_csv(checkpoint_path)
		records = [EvalRecord(**row) for row in prev.to_dict(orient="records")]
		resume_qids = set(prev["qid"].tolist())
		print(f"[run_eval] resumed from {checkpoint_path}: {len(resume_qids)} prompts already done")

	for i, row in enumerate(tqdm(ds, desc="FaithEval")):
		qid = row["qid"]
		if qid in resume_qids:
			continue

		prompt = _build_prompt(template, row["context"], row["question"])
		try:
			output = _greedy_generate(
				model, tokenizer, prompt, pre_forward_hook=pre_forward_hook
			)
		except Exception as e:
			output = ""
			print(f"[run_eval] generation failed for qid={qid}: {e}")

		result = classify(output, row["question"], row["context"], force_judge=force_judge)
		records.append(
			EvalRecord(
				qid=qid,
				question=row["question"],
				context=row["context"],
				output=output,
				label=result.label,
				method=result.method,
				reason=result.reason,
			)
		)

		if checkpoint_path is not None and (i + 1) % checkpoint_every == 0:
			pd.DataFrame([asdict(r) for r in records]).to_csv(checkpoint_path, index=False)

	df = pd.DataFrame([asdict(r) for r in records])
	if checkpoint_path is not None:
		df.to_csv(checkpoint_path, index=False)
	return df


def refusal_rate(df: pd.DataFrame) -> float:
	if len(df) == 0:
		return 0.0
	return float((df["label"] == "refuses").sum() / len(df))


def hallucination_rate(df: pd.DataFrame) -> float:
	if len(df) == 0:
		return 0.0
	return float((df["label"] == "fabricates").sum() / len(df))


def summary(df: pd.DataFrame) -> dict:
	return {
		"n": len(df),
		"refusal_rate": refusal_rate(df),
		"hallucination_rate": hallucination_rate(df),
		"off_topic_rate": float((df["label"] == "off_topic").sum() / max(len(df), 1)),
		"by_method": df["method"].value_counts().to_dict() if len(df) else {},
	}
