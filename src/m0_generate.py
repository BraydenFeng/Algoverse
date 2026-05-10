"""
Module 0.0 — Base Model Viability Check (Generation)
v5.2 plan, step 1 of M0.

Generates Gemma-3-12B-PT outputs on FaithEval-unanswerable.
Saves raw outputs to JSONL for downstream classification.

VERIFY BEFORE RUNNING:
  python src/m0_generate.py --inspect_only
This prints the dataset schema + 3 examples. If the dataset path or
column names don't match what format_prompt() assumes, fix that first.

Usage:
  # Inspect dataset
  python src/m0_generate.py --inspect_only

  # Smoke test on 10 prompts
  python src/m0_generate.py --smoke_test 10

  # Full run
  python src/m0_generate.py

  # Resume after a crash
  python src/m0_generate.py --resume

  # If PT fails M0, run IT variant
  python src/m0_generate.py --model google/gemma-3-12b-it --mode it \\
      --output outputs/m0/raw_outputs_it.jsonl
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument("--model", default="google/gemma-3-12b-pt")
	p.add_argument("--mode", choices=["pt", "it"], default="pt",
				   help="pt = raw prompt, it = apply chat template")
	p.add_argument("--dataset", default="Salesforce/FaithEval-unanswerable-v1.0",
				   help="VERIFY this HF path. Use --inspect_only first.")
	p.add_argument("--split", default="test")
	p.add_argument("--output", default="outputs/m0/raw_outputs.jsonl")
	p.add_argument("--smoke_test", type=int, default=0)
	p.add_argument("--batch_size", type=int, default=4)
	p.add_argument("--max_new_tokens", type=int, default=150)
	p.add_argument("--resume", action="store_true")
	p.add_argument("--inspect_only", action="store_true")
	return p.parse_args()


def inspect_dataset(ds):
	print("\n=== Features ===")
	print(ds.features)
	print(f"\n=== Size: {len(ds)} ===")
	print("\n=== First 3 examples ===")
	for i in range(min(3, len(ds))):
		print(f"\n--- Example {i} ---")
		for k, v in ds[i].items():
			s = str(v)
			print(f"{k}: {s[:500]}{'...[truncated]' if len(s) > 500 else ''}")
	print("\nIf column names differ from what format_prompt() handles, edit that function.")


def format_prompt(example, mode, tokenizer):
	"""
	Build the input string for one example. ADJUST after --inspect_only if
	your dataset uses different column names.
	"""
	if "context" in example and "question" in example:
		raw = f"Context: {example['context']}\n\nQuestion: {example['question']}\n\nAnswer:"
	elif "prompt" in example:
		raw = example["prompt"]
	elif "input" in example:
		raw = example["input"]
	elif "text" in example:
		raw = example["text"]
	else:
		raise KeyError(f"No known prompt field in example. Keys: {list(example.keys())}. "
					   f"Edit format_prompt() to match your dataset.")

	if mode == "it":
		messages = [{"role": "user", "content": raw}]
		return tokenizer.apply_chat_template(
			messages, tokenize=False, add_generation_prompt=True
		)
	return raw


def load_done_ids(output_path):
	if not output_path.exists():
		return set()
	done = set()
	with open(output_path) as f:
		for line in f:
			try:
				obj = json.loads(line)
				if "_metadata" not in obj:
					done.add(obj["prompt_id"])
			except (json.JSONDecodeError, KeyError):
				continue
	return done


def generate_batch(model, tokenizer, prompts, max_new_tokens):
	inputs = tokenizer(
		prompts,
		return_tensors="pt",
		padding=True,
		truncation=True,
		max_length=2048,
	).to(model.device)

	with torch.no_grad():
		outputs = model.generate(
			**inputs,
			max_new_tokens=max_new_tokens,
			do_sample=False,
			pad_token_id=tokenizer.pad_token_id,
		)

	new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
	return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def main():
	args = parse_args()
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	print(f"Loading dataset: {args.dataset} [{args.split}]")
	ds = load_dataset(args.dataset, split=args.split)
	print(f"Loaded {len(ds)} examples. Columns: {ds.column_names}")

	if args.inspect_only:
		inspect_dataset(ds)
		return

	if args.smoke_test > 0:
		ds = ds.select(range(min(args.smoke_test, len(ds))))
		print(f"SMOKE TEST mode: {len(ds)} examples")

	done_ids = load_done_ids(output_path) if args.resume else set()
	if done_ids:
		print(f"Resume: {len(done_ids)} already done, skipping those.")

	print(f"\nLoading {args.model} in bf16...")
	tokenizer = AutoTokenizer.from_pretrained(args.model)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	# left-padding is required for batched causal-LM generation; right-padding
	# would put the prompt's last real token in the wrong position vs the kv cache.
	tokenizer.padding_side = "left"

	model = AutoModelForCausalLM.from_pretrained(
		args.model,
		torch_dtype=torch.bfloat16,
		device_map="auto",
	)
	model.eval()
	print(f"Model loaded.")

	pending = [(i, ex) for i, ex in enumerate(ds) if str(i) not in done_ids]
	print(f"\nGenerating {len(pending)} completions (batch_size={args.batch_size})...")

	metadata = {
		"_metadata": True,
		"model": args.model,
		"mode": args.mode,
		"dataset": args.dataset,
		"split": args.split,
		"max_new_tokens": args.max_new_tokens,
		"batch_size": args.batch_size,
		"timestamp_start_utc": datetime.utcnow().isoformat(),
	}

	write_metadata = not done_ids and not output_path.exists()
	with open(output_path, "a") as fout:
		if write_metadata:
			fout.write(json.dumps(metadata) + "\n")
			fout.flush()

		for start in tqdm(range(0, len(pending), args.batch_size)):
			batch = pending[start:start + args.batch_size]
			prompts = [format_prompt(ex, args.mode, tokenizer) for _, ex in batch]

			try:
				completions = generate_batch(model, tokenizer, prompts, args.max_new_tokens)
			except torch.cuda.OutOfMemoryError:
				print(f"\nOOM at batch {start}. Reduce --batch_size and resume with --resume.")
				raise

			for (idx, ex), prompt, completion in zip(batch, prompts, completions):
				record = {
					"prompt_id": str(idx),
					"prompt_formatted": prompt,
					"completion": completion,
					"original_example": dict(ex),
				}
				fout.write(json.dumps(record, default=str) + "\n")
			fout.flush()

	print(f"\nDone. Outputs: {output_path}")
	print(f"Next: python src/m0_classify.py --input {output_path}")


if __name__ == "__main__":
	main()
