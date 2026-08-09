"""M5.B/M5.C — corrected mediation driver, same-layer rerun, and latent screen.

Why this exists rather than reusing notebooks/m4_mediation.ipynb: that notebook
captures the mediator at the LAST PROMPT TOKEN and steers at alpha=0.3. Both are
stale. Last-token capture reads identically zero for the unknown-entity latent
(it fires on the entity mention, not the decision token), which made the mediator
degenerate; and the paper headlines alpha=0.5. The numbers in the paper came from
a corrected session that was never committed. This module is that session, in
src/, so it is reproducible.

Three entry points:

	run_mediation()      M5.B — arms A/B/C/D at a chosen layer with entity-token
	                     capture. Run it at the EXTRACTION layer to answer the
	                     reviewers' "behaviour at L28, mediation at L21, the
	                     15.2/15.3 agreement could be coincidence" objection.

	rank_candidates()    M5.C phase 1 — forward-only. For each prompt, SAE-encode
	                     every prompt position under baseline and under steering,
	                     reduce to each latent's peak activation, and rank latents
	                     by the steered-minus-baseline shift. Cheap; narrows ~16k
	                     latents to a screenable handful.

	screen_latents()     M5.C phase 2 — the go/no-go. For each candidate latent,
	                     run the rescue arm (steer + clamp that latent back to its
	                     own per-prompt baseline) and report how much fabrication
	                     drops versus the steered arm. That drop is the latent's
	                     ACME. A latent that carries the effect shows a large drop;
	                     the published unknown-entity latent showed ~0.

Phase 1 ranks by correlation, phase 2 filters by causation. Attribution-only work
stops at phase 1; the mediation arms are what separate a mediator from a correlate.

Everything checkpoints locally and mirrors to the HF artifact repo.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from .lib.classifier import classify
from .lib.config import layer_suffix, load_config
from .m4_mediation import (
	_entity_position,
	compute_mediation,
	format_result,
	make_mediator_capture_hook,
	make_rescue_hook,
)
from .steering import estimate_residual_norm, load_emotion_vector


@dataclass
class M5Record:
	qid: str
	question: str
	output: str
	label: str
	method: str
	reason: str
	mediator: float
	entity_pos: int


def _upload(local_path: Path, repo_id: str, path_in_repo: str) -> None:
	try:
		from huggingface_hub import upload_file

		upload_file(
			path_or_fileobj=str(local_path),
			path_in_repo=path_in_repo,
			repo_id=repo_id,
			repo_type="dataset",
		)
	except Exception as e:
		print(f"[m5] HF sync of {path_in_repo} failed (local copy kept): {e}")


def _inputs(tokenizer, prompt: str, use_chat_template: bool, device):
	"""Tokenize one prompt. Instruct models need the chat template or they ramble
	past the answer (the M3 bug); M4 used it too, so mediation matches M3."""
	if use_chat_template:
		enc = tokenizer.apply_chat_template(
			[{"role": "user", "content": prompt}],
			add_generation_prompt=True,
			return_tensors="pt",
			return_dict=True,
		)
	else:
		enc = tokenizer(prompt, return_tensors="pt")
	return {k: v.to(device) for k, v in enc.items()}


def _steering_tensor(vector: np.ndarray, alpha: float, norm_scale: float, model) -> torch.Tensor:
	v = np.asarray(vector, dtype=np.float32)
	v = v / np.linalg.norm(v)
	t = torch.tensor(v, dtype=model.dtype, device=model.device)
	return t * float(alpha * norm_scale)


def _steer_and_capture_hook(sae, latent_idx: int, layer: int, tok_idx: int, steer_vec, capture: dict):
	"""Arm B: steer every position, then read the latent at the entity token.
	Must be one hook — two hooks on the same layer race and only the last wins."""

	def factory(model):
		def hook(module, args, output):
			h = output[0] if isinstance(output, tuple) else output
			h_steered = h + steer_vec
			if h_steered.shape[1] > tok_idx:
				vec = h_steered[0, tok_idx, :].float()
				with torch.no_grad():
					a = sae.encode(vec.unsqueeze(0).to(next(sae.parameters()).dtype)).squeeze(0)
				capture["value"] = float(a[latent_idx].item())
			if isinstance(output, tuple):
				return (h_steered, *output[1:])
			return h_steered

		return model.model.layers[layer].register_forward_hook(hook)

	return factory


def _clamp_only_hook(sae, latent_idx: int, layer: int, tok_idx: int, clamp_value: float, capture: dict):
	"""Arm D: swap latent j's contribution with no steering. Inertness probe."""
	cache: dict = {}

	def factory(model):
		if "w" not in cache:
			W_dec = sae.W_dec if hasattr(sae, "W_dec") else sae.decoder.weight.T
			cache["w"] = W_dec[latent_idx].detach().to(model.device).to(model.dtype)
		w_j = cache["w"]

		def hook(module, args, output):
			h = output[0] if isinstance(output, tuple) else output
			if h.shape[1] > tok_idx:
				vec = h[0, tok_idx, :].float()
				with torch.no_grad():
					a = sae.encode(vec.unsqueeze(0).to(next(sae.parameters()).dtype)).squeeze(0)
				delta = (clamp_value - float(a[latent_idx].item())) * w_j
				h_new = h.clone()
				h_new[0, tok_idx, :] = h_new[0, tok_idx, :] + delta
				capture["value"] = clamp_value
				if isinstance(output, tuple):
					return (h_new, *output[1:])
				return h_new
			return output

		return model.model.layers[layer].register_forward_hook(hook)

	return factory


def _run_arm(
	model, tokenizer, sae, *,
	arm_name: str,
	layer: int,
	latent_idx: int,
	hook_for: Callable,          # (qid, tok_idx, capture) -> hook factory or None
	checkpoint_path: Path,
	repo: str | None,
	repo_path: str | None,
	limit: int | None,
	use_chat_template: bool,
	max_new_tokens: int = 128,
	checkpoint_every: int = 200,
) -> pd.DataFrame:
	"""Generate + classify one arm, capturing the mediator at the entity token."""
	cfg = load_config()
	ds = load_dataset(cfg["faitheval"]["hf_dataset"], split="test")
	if limit is not None:
		ds = ds.select(range(min(limit, len(ds))))
	template = cfg["faitheval"]["prompt_template"]

	records: list[M5Record] = []
	done: set[str] = set()
	if checkpoint_path.exists():
		try:
			prev = pd.read_csv(checkpoint_path)
			records = [M5Record(**r) for r in prev.to_dict(orient="records")]
			done = set(prev["qid"].tolist())
			print(f"[m5:{arm_name}] resumed {len(done)} prompts")
		except Exception as e:
			print(f"[m5:{arm_name}] checkpoint unreadable ({e}); starting fresh")

	pending = [r for r in ds if r["qid"] not in done]
	for i, row in enumerate(tqdm(pending, desc=f"m5:{arm_name}")):
		prompt = template.format(context=row["context"], question=row["question"])
		capture = {"value": float("nan")}
		try:
			enc = _inputs(tokenizer, prompt, use_chat_template, model.device)
			tok_idx = _entity_position(model, sae, enc, layer, latent_idx)
			factory = hook_for(row["qid"], tok_idx, capture)
			handle = factory(model) if factory is not None else None
			try:
				with torch.no_grad():
					out = model.generate(
						**enc, max_new_tokens=max_new_tokens, do_sample=False,
						pad_token_id=tokenizer.eos_token_id,
					)
				text = tokenizer.decode(
					out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
				).strip()
			finally:
				if handle is not None:
					handle.remove()
		except Exception as e:
			print(f"[m5:{arm_name}] qid={row['qid']} failed: {e}")
			text, tok_idx = "", -1

		res = classify(text, row["question"], row["context"])
		records.append(M5Record(
			qid=row["qid"], question=row["question"], output=text,
			label=res.label, method=res.method, reason=res.reason,
			mediator=capture["value"], entity_pos=int(tok_idx),
		))

		if (i + 1) % checkpoint_every == 0:
			pd.DataFrame([asdict(r) for r in records]).to_csv(checkpoint_path, index=False)
			if repo and repo_path:
				_upload(checkpoint_path, repo, repo_path)

	df = pd.DataFrame([asdict(r) for r in records])
	df.to_csv(checkpoint_path, index=False)
	if repo and repo_path:
		_upload(checkpoint_path, repo, repo_path)
	return df


def _rates(df: pd.DataFrame) -> dict:
	"""Rates over non-empty generations, matching the M3/M4 convention."""
	d = df.copy()
	d["output"] = d["output"].fillna("").astype(str)
	ne = d[d["output"].str.strip() != ""]
	n = max(len(ne), 1)
	return {
		"refuse": float((ne["label"] == "refuses").sum() / n),
		"fabricate": float((ne["label"] == "fabricates").sum() / n),
		"off_topic": float((ne["label"] == "off_topic").sum() / n),
		"n_nonempty": len(ne),
		"n_total": len(d),
	}


def run_mediation(
	model, tokenizer, sae, *,
	latent_idx: int,
	layer: int,
	alpha: float = 0.5,
	model_key: str = "primary",
	limit: int | None = None,
	tag: str | None = None,
	emotion: str = "desperation",
) -> dict:
	"""M5.B — the four arms with entity-token capture at `layer` and `alpha`.

	Run with layer = the EXTRACTION layer to co-locate behaviour and mediation and
	close the reviewers' cross-layer objection.
	"""
	cfg = load_config()
	repo = cfg["paths"]["hf_artifact_repo"]
	use_ct = bool(cfg["models"][model_key].get("uses_chat_template", False))
	run_tag = tag or f"{layer_suffix(cfg, model_key)}_L{layer}_a{alpha:g}"
	out_dir = Path(cfg["paths"]["outputs_dir"]) / "m5_mediation" / run_tag
	out_dir.mkdir(parents=True, exist_ok=True)
	repo_dir = f"m5_mediation/{run_tag}"

	vec = load_emotion_vector(emotion, model_key=model_key)
	norm_scale = estimate_residual_norm(
		model, tokenizer, layer=layer,
		calibration_texts=_neutral_texts(cfg),
		token_skip=cfg["extraction"]["token_skip"],
	)
	steer = _steering_tensor(vec, alpha, norm_scale, model)
	print(f"[m5] L{layer} alpha={alpha} latent={latent_idx} norm_scale={norm_scale:.2f}")

	common = dict(
		model=model, tokenizer=tokenizer, sae=sae, layer=layer, latent_idx=latent_idx,
		repo=repo, limit=limit, use_chat_template=use_ct,
	)

	df_a = _run_arm(
		arm_name="A_baseline",
		hook_for=lambda qid, t, cap: make_mediator_capture_hook(sae, latent_idx, layer, t, cap),
		checkpoint_path=out_dir / "arm_A.csv", repo_path=f"{repo_dir}/arm_A.csv", **common,
	)
	df_b = _run_arm(
		arm_name="B_steered",
		hook_for=lambda qid, t, cap: _steer_and_capture_hook(sae, latent_idx, layer, t, steer, cap),
		checkpoint_path=out_dir / "arm_B.csv", repo_path=f"{repo_dir}/arm_B.csv", **common,
	)

	clamp = dict(zip(df_a["qid"], df_a["mediator"]))

	def rescue_hook(qid, t, cap):
		v = float(clamp.get(qid, float("nan")))
		if not np.isfinite(v):
			return None
		cap["value"] = v
		return make_rescue_hook(
			sae=sae, latent_idx=latent_idx, layer=layer, target_token_idx=t,
			clamp_value=v, steering_vector_on_device=steer,
		)

	df_c = _run_arm(
		arm_name="C_rescue", hook_for=rescue_hook,
		checkpoint_path=out_dir / "arm_C.csv", repo_path=f"{repo_dir}/arm_C.csv", **common,
	)

	low = float(np.nanpercentile(df_a["mediator"], 10))
	df_d = _run_arm(
		arm_name="D_suppress",
		hook_for=lambda qid, t, cap: _clamp_only_hook(sae, latent_idx, layer, t, low, cap),
		checkpoint_path=out_dir / "arm_D.csv", repo_path=f"{repo_dir}/arm_D.csv", **common,
	)

	rates = {"A": _rates(df_a), "B": _rates(df_b), "C": _rates(df_c), "D": _rates(df_d)}
	res = compute_mediation(
		rates,
		{k: rates[k]["n_nonempty"] for k in rates},
		{
			"A_mean": float(df_a["mediator"].mean()),
			"B_mean": float(df_b["mediator"].mean()),
			"A_to_B_shift": float(df_b["mediator"].mean() - df_a["mediator"].mean()),
		},
	)
	text = format_result(res)
	text += (
		f"\n\nlayer={layer} alpha={alpha} latent={latent_idx} (entity-token capture)\n"
		f"fabrication: A={rates['A']['fabricate']:.3f} B={rates['B']['fabricate']:.3f} "
		f"C={rates['C']['fabricate']:.3f} D={rates['D']['fabricate']:.3f}\n"
		f"suppression clamp (10th pct of A) = {low:.4f}\n"
	)
	p = out_dir / "decision.txt"
	p.write_text(text, encoding="utf-8")
	_upload(p, repo, f"{repo_dir}/decision.txt")
	print("\n" + text)
	return {"result": res, "rates": rates, "arms": {"A": df_a, "B": df_b, "C": df_c, "D": df_d}}


def _neutral_texts(cfg: dict) -> list[str]:
	d = Path(cfg["paths"]["data_dir"]) / "stories" / "neutral"
	paths = sorted(d.glob("*.txt"))
	if not paths:
		raise FileNotFoundError(f"no neutral stories at {d}; cannot calibrate norm_scale")
	return [p.read_text(encoding="utf-8") for p in paths]


def rank_candidates(
	model, tokenizer, sae, *,
	layer: int,
	alpha: float = 0.5,
	model_key: str = "primary",
	n_prompts: int = 200,
	top_k: int = 15,
	emotion: str = "desperation",
) -> pd.DataFrame:
	"""M5.C phase 1 — rank latents by steered-minus-baseline peak activation.

	Peak-over-positions (not a fixed token) because each latent fires at its own
	position; using one token would bias the ranking toward latents that happen to
	fire where the published unknown-entity latent does.
	"""
	cfg = load_config()
	repo = cfg["paths"]["hf_artifact_repo"]
	use_ct = bool(cfg["models"][model_key].get("uses_chat_template", False))
	ds = load_dataset(cfg["faitheval"]["hf_dataset"], split="test").select(range(n_prompts))
	template = cfg["faitheval"]["prompt_template"]

	vec = load_emotion_vector(emotion, model_key=model_key)
	norm_scale = estimate_residual_norm(
		model, tokenizer, layer=layer, calibration_texts=_neutral_texts(cfg),
		token_skip=cfg["extraction"]["token_skip"],
	)
	steer = _steering_tensor(vec, alpha, norm_scale, model)

	cap: dict = {}

	def grab(_m, _a, output):
		cap["h"] = (output[0] if isinstance(output, tuple) else output).detach()
		return output

	def peaks(enc, steered: bool):
		handle = model.model.layers[layer].register_forward_hook(
			(lambda m, a, o: grab(m, a, (o[0] + steer,) + tuple(o[1:]) if isinstance(o, tuple) else o + steer))
			if steered else grab
		)
		try:
			with torch.no_grad():
				model(**enc)
			acts = sae.encode(cap["h"][0].to(next(sae.parameters()).dtype))
			return acts.max(dim=0).values.float().cpu().numpy()
		finally:
			handle.remove()

	base_sum = steer_sum = None
	n_ok = 0
	for row in tqdm(ds, desc="m5:rank"):
		prompt = template.format(context=row["context"], question=row["question"])
		try:
			enc = _inputs(tokenizer, prompt, use_ct, model.device)
			b, s = peaks(enc, False), peaks(enc, True)
		except Exception as e:
			print(f"[m5:rank] skip {row['qid']}: {e}")
			continue
		base_sum = b if base_sum is None else base_sum + b
		steer_sum = s if steer_sum is None else steer_sum + s
		n_ok += 1

	if not n_ok:
		raise RuntimeError("no prompts encoded; cannot rank candidates")

	base_mean, steer_mean = base_sum / n_ok, steer_sum / n_ok
	shift = steer_mean - base_mean
	order = np.argsort(-np.abs(shift))[:top_k]
	df = pd.DataFrame({
		"latent_idx": order.astype(int),
		"baseline_peak": base_mean[order],
		"steered_peak": steer_mean[order],
		"shift": shift[order],
	})

	out_dir = Path(cfg["paths"]["outputs_dir"]) / "m5_screen" / f"L{layer}_a{alpha:g}"
	out_dir.mkdir(parents=True, exist_ok=True)
	p = out_dir / "candidates.csv"
	df.to_csv(p, index=False)
	_upload(p, repo, f"m5_screen/L{layer}_a{alpha:g}/candidates.csv")
	print(df.to_string(index=False))
	return df


def screen_latents(
	model, tokenizer, sae, *,
	candidates: list[int],
	layer: int,
	alpha: float = 0.5,
	model_key: str = "primary",
	n_prompts: int = 150,
	threshold_pts: float = 3.0,
) -> pd.DataFrame:
	"""M5.C phase 2 — per-candidate rescue screen. THE GO/NO-GO.

	For each candidate: fabrication(steered) - fabrication(rescue-on-that-latent),
	in points. That is the latent's ACME. Pre-registered decision rule:
		any candidate >= threshold_pts -> CLEAR    (localizable; run the full screen)
		structure but none clear       -> DISTRIBUTED (pivot to group/subspace)
		nothing moves                  -> DEAD     (stop; do not commit further)
	"""
	cfg = load_config()
	repo = cfg["paths"]["hf_artifact_repo"]
	rows = []
	steered_fab = None

	for j, latent_idx in enumerate(candidates):
		print(f"\n[m5:screen] candidate {j+1}/{len(candidates)}: latent {latent_idx}")
		out = run_mediation(
			model, tokenizer, sae, latent_idx=latent_idx, layer=layer, alpha=alpha,
			model_key=model_key, limit=n_prompts,
			tag=f"screen_L{layer}_a{alpha:g}/latent_{latent_idx}",
		)
		r = out["rates"]
		acme_pts = (r["B"]["fabricate"] - r["C"]["fabricate"]) * 100
		steered_fab = r["B"]["fabricate"]
		rows.append({
			"latent_idx": latent_idx,
			"fab_baseline": r["A"]["fabricate"],
			"fab_steered": r["B"]["fabricate"],
			"fab_rescue": r["C"]["fabricate"],
			"fab_suppress": r["D"]["fabricate"],
			"acme_pts": acme_pts,
			"te_pts": (r["B"]["fabricate"] - r["A"]["fabricate"]) * 100,
		})
		print(f"[m5:screen] latent {latent_idx}: ACME = {acme_pts:+.1f} pts")

	df = pd.DataFrame(rows).sort_values("acme_pts", key=abs, ascending=False)
	best = float(df["acme_pts"].abs().max()) if len(df) else 0.0
	if best >= threshold_pts:
		verdict = f"CLEAR — latent {int(df.iloc[0]['latent_idx'])} carries {best:.1f} pts. Run the full screen."
	elif best >= 1.0:
		verdict = f"DISTRIBUTED — best single latent only {best:.1f} pts. Pivot to group/subspace mediation."
	else:
		verdict = f"DEAD — no latent moves fabrication ({best:.1f} pts). Do not commit further."

	out_dir = Path(cfg["paths"]["outputs_dir"]) / "m5_screen" / f"L{layer}_a{alpha:g}"
	out_dir.mkdir(parents=True, exist_ok=True)
	df.to_csv(out_dir / "screen.csv", index=False)
	text = df.to_string(index=False) + f"\n\nsteered fabrication = {steered_fab}\n\n{verdict}\n"
	(out_dir / "verdict.txt").write_text(text, encoding="utf-8")
	_upload(out_dir / "screen.csv", repo, f"m5_screen/L{layer}_a{alpha:g}/screen.csv")
	_upload(out_dir / "verdict.txt", repo, f"m5_screen/L{layer}_a{alpha:g}/verdict.txt")
	print("\n" + text)
	return df
