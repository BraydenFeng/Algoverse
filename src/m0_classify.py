"""
Module 0.0 — Classification & Decision
v5.2 plan, step 2 of M0.

Pipeline:
  1. Load JSONL produced by src/m0_generate.py.
  2. Classify each completion via src.lib.classifier.classify (rule + Claude judge).
     This is the single canonical classifier — also used by src/faitheval_eval.py
     for M3, so M0 and M3 share classification logic.
  3. Compute H_refuse_base = refuses / total.
  4. Apply v5.2 decision rule:
       H_refuse_base >= 0.15  -> PROCEED with PT
       0.05 <= H < 0.15       -> MARGINAL: run IT in parallel
       H < 0.05               -> SWITCH to IT pipeline-wide
  5. Sample 100 records into a hand-label CSV for judge-agreement audit.
     If hand-label agreement < 85%, refine the judge prompt in
     src/lib/classifier.py and re-run.

Usage:
  export ANTHROPIC_API_KEY=...
  python src/m0_classify.py
"""

import argparse
import csv
import json
import random
from pathlib import Path

from tqdm import tqdm

from .lib.classifier import classify


def _extract_question_context(record):
	"""Pull question/context out of the original example so the judge sees the grounding.

	Returns ("", "") if the source dataset uses a different schema; classifier still
	works on the completion alone (the judge prompt just gets empty fields).
	"""
	ex = record.get("original_example", {})
	question = ex.get("question") or ex.get("query") or ""
	context = ex.get("context") or ex.get("passage") or ex.get("input") or ""
	return question, context


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--input", default="outputs/m0/raw_outputs.jsonl")
	p.add_argument("--output", default="outputs/m0/classified.csv")
	p.add_argument("--decision", default="outputs/m0/decision.txt")
	p.add_argument("--handlabel_sample", default="outputs/m0/handlabel_sample.csv",
				   help="100 random records to hand-label for judge agreement check.")
	p.add_argument("--seed", type=int, default=42)
	args = p.parse_args()

	Path(args.output).parent.mkdir(parents=True, exist_ok=True)

	records = []
	with open(args.input) as f:
		for line in f:
			obj = json.loads(line)
			if obj.get("_metadata"):
				continue
			records.append(obj)
	print(f"Loaded {len(records)} records.")

	for r in tqdm(records, desc="Classifying"):
		question, context = _extract_question_context(r)
		result = classify(r["completion"], question, context)
		r["final_class"] = result.label
		r["classify_method"] = result.method
		r["classify_reason"] = result.reason

	n_total = len(records)
	n_refuses = sum(1 for r in records if r["final_class"] == "refuses")
	n_fabricates = sum(1 for r in records if r["final_class"] == "fabricates")
	n_off_topic = sum(1 for r in records if r["final_class"] == "off_topic")
	n_other = n_total - n_refuses - n_fabricates - n_off_topic
	# H_refuse_base denominator = full n_total, matching v5.2 plan threshold calibration
	# (0.15 / 0.05 thresholds were chosen against this denominator). lib/classifier.py
	# returns a 3-way label (refuses / fabricates / off_topic, plus rare unparseable
	# "ambiguous" leakage from the judge); off_topic + ambiguous count toward the
	# denominator but neither numerator-side bucket. If you want to revisit and exclude
	# off_topic instead, also revisit the 0.15 / 0.05 thresholds.
	H_refuse_base = (n_refuses / n_total) if n_total else 0.0

	if H_refuse_base >= 0.15:
		decision = "PROCEED with PT. H_refuse_base >= 0.15 threshold met."
	elif H_refuse_base >= 0.05:
		decision = ("MARGINAL. Run IT variant in parallel: "
					"python src/m0_generate.py --model google/gemma-3-12b-it --mode it "
					"--output outputs/m0/raw_outputs_it.jsonl")
	else:
		decision = "SWITCH to IT pipeline-wide. PT has no usable refusal mode."

	with open(args.output, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=[
			"prompt_id", "final_class", "classify_method", "classify_reason", "completion_snippet"
		])
		writer.writeheader()
		for r in records:
			writer.writerow({
				"prompt_id": r["prompt_id"],
				"final_class": r["final_class"],
				"classify_method": r.get("classify_method", ""),
				"classify_reason": r.get("classify_reason", ""),
				"completion_snippet": r["completion"][:300].replace("\n", " "),
			})

	random.seed(args.seed)
	sample = random.sample(records, min(100, len(records)))
	with open(args.handlabel_sample, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=[
			"prompt_id", "completion", "model_class", "your_label"
		])
		writer.writeheader()
		for r in sample:
			writer.writerow({
				"prompt_id": r["prompt_id"],
				"completion": r["completion"][:1000].replace("\n", " "),
				"model_class": r["final_class"],
				"your_label": "",
			})

	with open(args.decision, "w") as f:
		f.write("Module 0.0 - Base Model Viability Decision\n")
		f.write("=" * 50 + "\n\n")
		f.write(f"Total prompts:   {n_total}\n")
		f.write(f"Refuses:         {n_refuses}\n")
		f.write(f"Fabricates:      {n_fabricates}\n")
		f.write(f"Off-topic:       {n_off_topic}\n")
		f.write(f"Ambiguous/other: {n_other}\n\n")
		f.write(f"H_refuse_base = refuses / (refuses + fabricates) = {H_refuse_base:.4f}\n\n")
		f.write(f"Decision: {decision}\n\n")
		f.write(f"Hand-label step: fill in 'your_label' column of {args.handlabel_sample}\n")
		f.write(f"Then compute agreement with 'model_class'. If <85%, refine the\n")
		f.write(f"judge prompt in src/lib/classifier.py and re-run.\n")

	print("\n" + "=" * 50)
	print(f"H_refuse_base = {H_refuse_base:.4f}")
	print(decision)
	print("=" * 50)
	print(f"Classified outputs:  {args.output}")
	print(f"Decision record:     {args.decision}")
	print(f"Hand-label this:     {args.handlabel_sample}")


if __name__ == "__main__":
	main()
