"""M4 — derive the unknown-entity SAE latent index at Gemma-2-9B PT, layer 21.

Implements the Ferrando v2 §3-§4 separation-score recipe on our hand-curated
entity prompts (src/m4_entity_dataset.py). Produces a JSON with the top-N
unknown-entity latents ranked by min-across-entity-types separation score.

Pipeline:
	1. For each entity prompt, run the model with a hook capturing the residual
	   stream at the target layer.
	2. Use the tokenizer's offset_mapping to find the LAST TOKEN of the entity
	   (not the last token of the prompt — the entity is mid-prompt).
	3. SAE-encode that residual to get sparse latent activations a (dim ~16k).
	4. For each latent j, accumulate active-counts split by (entity_type x
	   known/unknown). A latent is "active" if a[j] > 0.
	5. Per type, compute f_known and f_unknown (activation frequencies).
	6. Separation score s_unknown[j] = min over entity types of
	   (f_unknown - f_known). The argmax over j is the most generalizable
	   unknown-entity latent.
	7. Write JSON: ranked top-N latents with per-type per-class frequencies.

Outputs land at outputs/m4/L{layer}/unknown_entity_latents.json. M4 mediation
(src/m4_mediation.py) reads this file to pick the latent to intervene on.

NOTE: Ferrando also filters out latents that activate >2% on random Pile tokens
(to enforce entity-specificity). We skip this filter for v2 — surface it in the
JSON as a "TODO: pile-frequency filter not applied" field. If the top latent
fails downstream validation (mediation arms show no causal effect), revisit.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .lib.config import layer_suffix, load_config
from .m4_entity_dataset import ENTITY_TYPES, EntityPrompt, build_prompts


@dataclass
class LatentScore:
	idx: int                                            # latent index in the SAE dictionary
	min_unknown_sep: float                              # min over entity types — the ranking metric
	per_type: dict[str, dict[str, float]]               # {entity_type: {f_known, f_unknown, sep_unknown}}


def _find_entity_last_token_idx(tokenizer, prompt: str, entity_name: str) -> int:
	"""Locate the index of the last token of `entity_name` inside `prompt`.

	Uses the tokenizer's offset_mapping. Returns the 0-indexed token position
	of the entity's final token in the encoded prompt. Raises if the entity
	can't be located.
	"""
	enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
	offsets = enc["offset_mapping"]

	entity_start = prompt.find(entity_name)
	if entity_start < 0:
		raise ValueError(f"entity {entity_name!r} not found in prompt {prompt!r}")
	entity_end = entity_start + len(entity_name)  # exclusive char index past last entity char

	# the entity's last token is the rightmost token whose char span overlaps
	# the entity span (i.e., whose start_char < entity_end and end_char > entity_start)
	last_idx: int | None = None
	for i, (s, e) in enumerate(offsets):
		if e <= entity_start or s >= entity_end:
			continue  # no overlap
		last_idx = i
	if last_idx is None:
		raise RuntimeError(
			f"could not locate entity {entity_name!r} token span inside {prompt!r}; "
			f"offsets={offsets}"
		)
	return last_idx


def _capture_residual_at(
	model, tokenizer, prompt: str, entity_name: str, layer: int,
) -> torch.Tensor:
	"""Forward pass on prompt; return the residual-stream vector at the last
	entity token at the requested layer. Shape: (d_model,)."""
	enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
	captured: dict[str, torch.Tensor] = {}

	def hook(module, args, output):
		h = output[0] if isinstance(output, tuple) else output
		captured["h"] = h.detach()  # (1, T, d_model)

	target = model.model.layers[layer]
	handle = target.register_forward_hook(hook)
	try:
		with torch.no_grad():
			model(**enc)
	finally:
		handle.remove()

	tok_idx = _find_entity_last_token_idx(tokenizer, prompt, entity_name)
	return captured["h"][0, tok_idx, :].float()  # (d_model,)


def _sae_encode(sae, residual: torch.Tensor) -> torch.Tensor:
	"""Apply the SAE encoder to one residual vector. Returns sparse activations.

	Works with both the sae_lens SAE wrapper (.encode method) and a plain
	(W_enc, b_enc, threshold) tuple for testability.
	"""
	if hasattr(sae, "encode"):
		# sae_lens SAE: handles JumpReLU + device casting internally
		with torch.no_grad():
			x = residual.to(next(sae.parameters()).device).to(next(sae.parameters()).dtype)
			a = sae.encode(x.unsqueeze(0)).squeeze(0)  # (d_sae,)
		return a.float()
	# fallback for tests: tuple of tensors
	W_enc, b_enc, threshold = sae
	pre = residual @ W_enc + b_enc
	return torch.where(pre > threshold, pre, torch.zeros_like(pre))


def derive_unknown_entity_latent(
	model, tokenizer, sae,
	*,
	layer: int,
	top_n: int = 10,
	prompts: list[EntityPrompt] | None = None,
) -> tuple[LatentScore, list[LatentScore]]:
	"""Run the §4 recipe; return (top latent, full top-N).

	The top latent (by min-across-entity-types separation score) is what M4
	mediation will use. The next N-1 are reported for diagnostic comparison —
	if mediation on the top latent shows no effect, the runner-up is the
	natural next thing to try.
	"""
	if prompts is None:
		prompts = build_prompts()

	# active_counts[entity_type][is_known][latent_idx] = count of prompts where
	# latent_idx was active (> 0). Lazy-shaped on first encoding.
	active_counts: dict[str, dict[bool, np.ndarray | None]] = {
		et: {True: None, False: None} for et in ENTITY_TYPES
	}
	prompt_counts: dict[str, dict[bool, int]] = {
		et: {True: 0, False: 0} for et in ENTITY_TYPES
	}

	for p in tqdm(prompts, desc="encode/entity"):
		try:
			resid = _capture_residual_at(model, tokenizer, p.prompt, p.entity_name, layer)
		except Exception as e:
			print(f"[derive] skip {p.entity_type}/{p.entity_name!r}: {e}")
			continue

		a = _sae_encode(sae, resid).cpu().numpy()  # (d_sae,)
		active = (a > 0).astype(np.int64)  # binary per-latent

		bucket = active_counts[p.entity_type][p.is_known]
		if bucket is None:
			active_counts[p.entity_type][p.is_known] = active.copy()
		else:
			bucket += active
		prompt_counts[p.entity_type][p.is_known] += 1

	# build per-type frequencies + separation per latent
	d_sae = next(arr for et in active_counts.values() for arr in et.values() if arr is not None).shape[0]
	# matrix of separation scores: rows = entity types, cols = latents
	sep_matrix = np.zeros((len(ENTITY_TYPES), d_sae), dtype=np.float64)
	freq_known = np.zeros_like(sep_matrix)
	freq_unknown = np.zeros_like(sep_matrix)
	for ti, et in enumerate(ENTITY_TYPES):
		n_known = max(prompt_counts[et][True], 1)
		n_unknown = max(prompt_counts[et][False], 1)
		k_counts = active_counts[et][True]
		u_counts = active_counts[et][False]
		if k_counts is None or u_counts is None:
			print(f"[derive] WARNING: no prompts encoded for entity_type {et}")
			continue
		freq_known[ti, :] = k_counts / n_known
		freq_unknown[ti, :] = u_counts / n_unknown
		sep_matrix[ti, :] = freq_unknown[ti, :] - freq_known[ti, :]

	# min-across-types separation for each latent — this is what Ferrando ranks by
	min_sep_unknown = sep_matrix.min(axis=0)  # (d_sae,)
	top_idxs = np.argsort(-min_sep_unknown)[:top_n]

	def _build(idx: int) -> LatentScore:
		per_type: dict[str, dict[str, float]] = {}
		for ti, et in enumerate(ENTITY_TYPES):
			per_type[et] = {
				"f_known": float(freq_known[ti, idx]),
				"f_unknown": float(freq_unknown[ti, idx]),
				"sep_unknown": float(sep_matrix[ti, idx]),
			}
		return LatentScore(
			idx=int(idx),
			min_unknown_sep=float(min_sep_unknown[idx]),
			per_type=per_type,
		)

	top_n_scores = [_build(int(i)) for i in top_idxs]
	return top_n_scores[0], top_n_scores


def write_json(top: LatentScore, top_n: list[LatentScore], out_path: Path) -> None:
	"""Persist the result for downstream M4 mediation to consume."""
	out = {
		"top_unknown_entity_latent_idx": top.idx,
		"top_min_unknown_sep": top.min_unknown_sep,
		"pile_frequency_filter_applied": False,  # TODO if v2 review needs it
		"top_n": [
			{
				"idx": s.idx,
				"min_unknown_sep": s.min_unknown_sep,
				"per_type": s.per_type,
			}
			for s in top_n
		],
	}
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def run_and_save(model, tokenizer, sae) -> Path:
	"""End-to-end: derive top latent, write JSON to outputs/m4/L{layer}/."""
	cfg = load_config()
	layer = cfg["sae"]["layer"]  # M4 lives at the SAE layer specifically
	out_dir = Path(cfg["paths"]["outputs_dir"]) / "m4" / layer_suffix(cfg)
	out_path = out_dir / "unknown_entity_latents.json"

	top, top_n = derive_unknown_entity_latent(model, tokenizer, sae, layer=layer)
	write_json(top, top_n, out_path)

	print(f"\n[derive] top unknown-entity latent at L{layer}: idx={top.idx}")
	print(f"[derive]   min-across-types separation: {top.min_unknown_sep:.3f}")
	print(f"[derive]   per-type breakdown:")
	for et, scores in top.per_type.items():
		print(f"             {et:8s}  f_known={scores['f_known']:.2f}  "
		      f"f_unknown={scores['f_unknown']:.2f}  sep={scores['sep_unknown']:+.2f}")
	print(f"[derive] wrote {out_path}")
	return out_path
