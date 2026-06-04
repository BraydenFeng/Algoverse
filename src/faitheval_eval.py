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


def _build_inputs(tokenizer, prompt: str, use_chat_template: bool):
	"""Tokenize one prompt, optionally wrapping it as a chat turn.

	Instruct models (Llama-3.1, Gemma-2-IT) only behave as assistants — emitting a
	stop token after a concise answer — when the prompt goes through their chat
	template. Fed raw, Llama in particular never stops and rambles to max_new_tokens
	(bug surfaced in M3 Llama: ~99% of outputs hit the 128-token cap). config.yaml's
	`uses_chat_template` is the source of truth; the notebook threads it in.
	"""
	if use_chat_template:
		messages = [{"role": "user", "content": prompt}]
		return tokenizer.apply_chat_template(
			messages,
			add_generation_prompt=True,
			return_tensors="pt",
			return_dict=True,
		)
	return tokenizer(prompt, return_tensors="pt")


def _greedy_generate(
	model,
	tokenizer,
	prompt: str,
	max_new_tokens: int = 128,
	pre_forward_hook: Callable | None = None,
	use_chat_template: bool = False,
) -> str:
	"""Greedy decode. pre_forward_hook lets M2.B/M2.C inject steering or ablation."""
	inputs = _build_inputs(tokenizer, prompt, use_chat_template).to(model.device)

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


def _greedy_generate_batch(
	model,
	tokenizer,
	prompts: list[str],
	max_new_tokens: int = 128,
	pre_forward_hook: Callable | None = None,
	use_chat_template: bool = False,
) -> list[str]:
	"""Greedy decode a batch of prompts. Equivalent to calling _greedy_generate once
	per prompt, up to floating-point reduction order, but amortizes the weight reads
	across the batch (the dominant cost in batch-1 decode).

	Left-padding is required: decoder-only generation continues from the rightmost
	token, so every row's true last token must sit flush right. The steering/ablation
	hooks add a (d_model,) vector that broadcasts over (B, T, d), and padded positions
	are masked out of attention, so batched output matches the single-prompt path.
	"""
	prev_side = tokenizer.padding_side
	tokenizer.padding_side = "left"
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token
	try:
		if use_chat_template:
			conversations = [[{"role": "user", "content": p}] for p in prompts]
			inputs = tokenizer.apply_chat_template(
				conversations,
				add_generation_prompt=True,
				return_tensors="pt",
				return_dict=True,
				padding=True,
			).to(model.device)
		else:
			inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

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
					pad_token_id=tokenizer.pad_token_id,
				)
		finally:
			if hook_handle is not None:
				hook_handle.remove()

		# every row shares the padded prompt length, so slice it off uniformly
		gen = out[:, inputs["input_ids"].shape[1] :]
		return [tokenizer.decode(row, skip_special_tokens=True).strip() for row in gen]
	finally:
		tokenizer.padding_side = prev_side


def _hf_sync_checkpoint(local_path: Path, repo_id: str, path_in_repo: str) -> None:
	"""Upload a checkpoint CSV to HF Hub so it survives Colab session death."""
	try:
		from huggingface_hub import upload_file

		upload_file(
			path_or_fileobj=str(local_path),
			path_in_repo=path_in_repo,
			repo_id=repo_id,
			repo_type="dataset",
		)
	except Exception as e:
		# don't crash the eval loop over a transient HF upload failure;
		# the local checkpoint is still on disk and will be retried next interval
		print(f"[run_eval] HF sync to {repo_id}/{path_in_repo} failed: {e}")


def run_eval(
	model,
	tokenizer,
	*,
	limit: int | None = None,
	pre_forward_hook: Callable | None = None,
	checkpoint_path: Path | None = None,
	checkpoint_every: int = 500,
	hf_sync_repo: str | None = None,
	hf_sync_path: str | None = None,
	force_judge: bool = False,
	batch_size: int = 1,
	use_chat_template: bool = False,
) -> pd.DataFrame:
	"""Run FaithEval-unanswerable through `model`.

	Args:
		model, tokenizer: loaded Gemma model + tokenizer.
		limit: cap on prompts processed (use for smoke tests; None = all 2,492).
		pre_forward_hook: callable(model) -> hook_handle; used to inject steering
			or ablation in M2.B/M2.C. None for baseline M0.0.
		checkpoint_path: write incremental CSV here every checkpoint_every prompts.
		checkpoint_every: checkpoint interval.
		hf_sync_repo: if set, also upload the checkpoint CSV to this HF dataset
			repo after each local write. Lets M3 survive Colab session death.
		hf_sync_path: path-in-repo for the HF copy (e.g. "m3_checkpoints/faitheval_baseline.csv").
			Required if hf_sync_repo is set.
		force_judge: skip the regex rule pass and send every output to the Claude
			judge. Use for diagnosis when refusal_rate=1.0 or 0.0 looks suspicious
			(see lib/classifier.classify docstring). Costs N judge calls.
		batch_size: prompts decoded together per forward pass. 1 = the original
			single-prompt path; >1 uses left-padded batched generation.
		use_chat_template: wrap each FaithEval prompt as a chat turn via
			tokenizer.apply_chat_template before tokenizing. Required for instruct
			models to answer concisely and stop; pass cfg["models"][key]["uses_chat_template"].

	Returns DataFrame indexed by qid with classification + raw output.
	"""
	if hf_sync_repo is not None and hf_sync_path is None:
		raise ValueError("hf_sync_path is required when hf_sync_repo is set")

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

	pending = [row for row in ds if row["qid"] not in resume_qids]
	last_ckpt = len(records)

	def _write_checkpoint():
		pd.DataFrame([asdict(r) for r in records]).to_csv(checkpoint_path, index=False)
		if hf_sync_repo is not None:
			_hf_sync_checkpoint(checkpoint_path, hf_sync_repo, hf_sync_path)

	pbar = tqdm(total=len(pending), desc="FaithEval")
	for start in range(0, len(pending), batch_size):
		chunk = pending[start : start + batch_size]
		prompts = [_build_prompt(template, r["context"], r["question"]) for r in chunk]
		try:
			if batch_size == 1:
				outputs = [
					_greedy_generate(
						model, tokenizer, prompts[0],
						pre_forward_hook=pre_forward_hook,
						use_chat_template=use_chat_template,
					)
				]
			else:
				outputs = _greedy_generate_batch(
					model, tokenizer, prompts,
					pre_forward_hook=pre_forward_hook,
					use_chat_template=use_chat_template,
				)
		except Exception as e:
			# a single bad row (or transient OOM) shouldn't drop the whole batch —
			# fall back to per-prompt so at most one prompt is lost
			print(f"[run_eval] batch generation failed at offset {start} ({e}); retrying per-prompt")
			outputs = []
			for p in prompts:
				try:
					outputs.append(
						_greedy_generate(
							model, tokenizer, p,
							pre_forward_hook=pre_forward_hook,
							use_chat_template=use_chat_template,
						)
					)
				except Exception as e2:
					outputs.append("")
					print(f"[run_eval]   per-prompt generation failed: {e2}")

		for r, output in zip(chunk, outputs):
			result = classify(output, r["question"], r["context"], force_judge=force_judge)
			records.append(
				EvalRecord(
					qid=r["qid"],
					question=r["question"],
					context=r["context"],
					output=output,
					label=result.label,
					method=result.method,
					reason=result.reason,
				)
			)

		pbar.update(len(chunk))
		if checkpoint_path is not None and len(records) - last_ckpt >= checkpoint_every:
			_write_checkpoint()
			last_ckpt = len(records)
	pbar.close()

	df = pd.DataFrame([asdict(r) for r in records])
	if checkpoint_path is not None:
		df.to_csv(checkpoint_path, index=False)
		if hf_sync_repo is not None:
			_hf_sync_checkpoint(checkpoint_path, hf_sync_repo, hf_sync_path)
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
