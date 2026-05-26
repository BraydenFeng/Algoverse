"""M4 — Imai 2010 causal mediation through the unknown-entity SAE latent.

Four arms (Vasu's PI ask, 2026-05-25):
	A. baseline + capture        — no intervention, record per-prompt mediator M(0)
	B. steered + capture         — desperation @ α=α_head, record M(1)
	C. rescue (steered + clamp)  — desperation steering applied AND mediator
	                                clamped to M(0)[qid]. Y(T=1, M=M(0)).
	D. reverse (suppress only)   — no desperation; clamp mediator to a low value
	                                across all prompts. Tests whether the latent
	                                alone is sufficient to mimic desperation's
	                                effect on refusal. (Vasu's specific request;
	                                differs slightly from Imai-strict D which
	                                clamps per-prompt to M(1)[qid].)

Imai decomposition (computed from arms A/B/C only):
	TE   = E[Y_B] - E[Y_A]                       total effect of steering
	ACME = E[Y_B] - E[Y_C]                       indirect (through mediator)
	ADE  = E[Y_C] - E[Y_A]                       direct (bypassing mediator)
	check: TE = ACME + ADE

Y is the refusal rate (refuses / n_nonempty) per the M3 conditional convention.

Mediator design choice (load-bearing):
	The mediator is the unknown-entity latent's activation value at the LAST
	PROMPT TOKEN at layer 21. During generation we re-clamp at every forward
	pass — the clamp value stays at the prompt-final value, not per-generated-
	token. Rationale: FaithEval prompts have no explicit entity tokens, so the
	last-prompt-token position is the natural "where the model is about to
	decide" location. Re-clamping per step keeps the intervention from washing
	out after one token.

Intervention math (per-latent swap, not full SAE reconstruction):
	original residual: h
	latent activation: a_j = SAE_encode(h)[j]
	current contribution from latent j: c_j = a_j * W_dec[j, :]
	target contribution: c_j' = a_j_target * W_dec[j, :]
	corrected residual: h - c_j + c_j'

	This swaps ONLY latent j's contribution; everything else passes through
	unchanged. Avoids the SAE's reconstruction loss (~5-10% MSE) contaminating
	the rest of the residual stream.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch


@dataclass
class MediationResult:
	te: float                                    # total effect of steering on refusal
	acme: float                                  # average causal mediation effect (indirect)
	ade: float                                   # average direct effect
	te_check: float                              # acme + ade (sanity check vs te)
	refusal_rate: dict[str, float]               # per-arm refusal rate
	fabrication_rate: dict[str, float]           # per-arm fab rate (informational)
	off_topic_rate: dict[str, float]             # per-arm off_topic rate
	n_per_arm: dict[str, int]                    # sample size per arm
	mediator_baseline: dict[str, float]          # per-arm mediator stats (mean, median)


def _last_prompt_token_idx(tokenizer, prompt: str) -> int:
	"""Return the 0-indexed position of the final token of the prompt."""
	ids = tokenizer(prompt, add_special_tokens=True).input_ids
	return len(ids) - 1


def make_mediator_capture_hook(
	sae,
	latent_idx: int,
	layer: int,
	target_token_idx: int,
	capture: dict,
) -> Callable:
	"""Return a hook factory `f(model) -> handle` that records the value of
	`sae`'s latent `latent_idx` at the specified token position on every
	forward pass. Writes into `capture['value']` (last-seen value wins).

	Used by arms A and B (capture only, no intervention).
	"""

	def factory(model):
		def hook(module, args, output):
			h = output[0] if isinstance(output, tuple) else output  # (B, T, d)
			# only capture if the requested token exists in this pass
			if h.shape[1] <= target_token_idx:
				return output
			vec = h[0, target_token_idx, :].float()
			with torch.no_grad():
				a = sae.encode(vec.unsqueeze(0).to(next(sae.parameters()).dtype)).squeeze(0)
			capture["value"] = float(a[latent_idx].item())
			return output

		target = model.model.layers[layer]
		return target.register_forward_hook(hook)

	return factory


def make_mediator_clamp_hook(
	sae,
	latent_idx: int,
	layer: int,
	target_token_idx: int,
	clamp_value: float,
	desperation_steering_factory: Callable | None = None,
) -> Callable:
	"""Return a hook factory that:
		1. Optionally applies desperation steering (if `desperation_steering_factory`
		   is provided) — used by arm C (rescue: steering + clamp).
		2. Replaces the contribution of `sae`'s latent `latent_idx` at
		   `target_token_idx` with what it would be at value `clamp_value`.

	The steering and clamp must be applied in the SAME hook so the order is:
	steering-add THEN clamp-correction. Two separate hooks on the same layer
	would race and only the last-registered would actually modify the output.
	"""

	W_dec_j = None  # lazy-init on first call (need device/dtype from model)

	def factory(model):
		nonlocal W_dec_j
		# extract the j-th decoder row once, cast to model's device/dtype
		if W_dec_j is None:
			W_dec = sae.W_dec if hasattr(sae, "W_dec") else sae.decoder.weight.T
			W_dec_j = W_dec[latent_idx].detach().to(model.device).to(model.dtype)

		# steering hook (if rescue arm) — register temporarily so we can re-apply manually
		# rather than chaining hooks. We do steering inline in the same hook below.
		steering_vec = None
		if desperation_steering_factory is not None:
			# extract the steering vector from the factory's closure — this is fragile
			# but avoids hook chaining. The clean alternative is to pass the steering
			# tensor directly; see steering.py for what factories return.
			# Easier: compute it ourselves by running the factory once on a tiny model
			# stub. Since we can't, we ask the caller to pre-extract the steering tensor
			# and pass it via the factory closure (see __init__ comment).
			raise NotImplementedError(
				"rescue-arm hook expects steering to be pre-composed. See "
				"make_rescue_hook() instead, which takes a precomputed steering tensor."
			)

		def hook(module, args, output):
			h = output[0] if isinstance(output, tuple) else output  # (B, T, d)
			if h.shape[1] <= target_token_idx:
				if isinstance(output, tuple):
					return (h, *output[1:])
				return h
			# read current latent value at target position
			vec = h[0, target_token_idx, :].float()
			with torch.no_grad():
				a = sae.encode(vec.unsqueeze(0).to(next(sae.parameters()).dtype)).squeeze(0)
			a_j = float(a[latent_idx].item())
			# swap contribution: subtract current, add target
			delta = (clamp_value - a_j) * W_dec_j  # (d,) in model dtype/device
			h_new = h.clone()
			h_new[0, target_token_idx, :] = h_new[0, target_token_idx, :] + delta
			if isinstance(output, tuple):
				return (h_new, *output[1:])
			return h_new

		target = model.model.layers[layer]
		return target.register_forward_hook(hook)

	return factory


def make_rescue_hook(
	sae,
	latent_idx: int,
	layer: int,
	target_token_idx: int,
	clamp_value: float,
	steering_vector_on_device: torch.Tensor,
) -> Callable:
	"""Rescue arm: applies BOTH desperation steering (add) AND mediator clamp
	(swap latent j's contribution) in a single hook.

	`steering_vector_on_device` must be `alpha * norm_scale * v_unit_norm` —
	the same magnitude the desperation arm uses, pre-cast to the model's
	device/dtype. Added to EVERY token position (matching steering.py's
	convention), then the clamp is applied only at `target_token_idx`.
	"""

	W_dec_j_cache: dict[str, torch.Tensor] = {}

	def factory(model):
		if "W_dec_j" not in W_dec_j_cache:
			W_dec = sae.W_dec if hasattr(sae, "W_dec") else sae.decoder.weight.T
			W_dec_j_cache["W_dec_j"] = W_dec[latent_idx].detach().to(model.device).to(model.dtype)
		W_dec_j = W_dec_j_cache["W_dec_j"]

		def hook(module, args, output):
			h = output[0] if isinstance(output, tuple) else output
			# step 1: apply steering everywhere
			h_steered = h + steering_vector_on_device
			# step 2: clamp the mediator at the target token (if present in this pass)
			if h_steered.shape[1] > target_token_idx:
				vec = h_steered[0, target_token_idx, :].float()
				with torch.no_grad():
					a = sae.encode(vec.unsqueeze(0).to(next(sae.parameters()).dtype)).squeeze(0)
				a_j = float(a[latent_idx].item())
				delta = (clamp_value - a_j) * W_dec_j
				h_steered = h_steered.clone()
				h_steered[0, target_token_idx, :] = h_steered[0, target_token_idx, :] + delta
			if isinstance(output, tuple):
				return (h_steered, *output[1:])
			return h_steered

		target = model.model.layers[layer]
		return target.register_forward_hook(hook)

	return factory


def compute_mediation(
	rates_per_arm: dict[str, dict[str, float]],
	n_per_arm: dict[str, int],
	mediator_stats: dict[str, float],
) -> MediationResult:
	"""Aggregate per-arm rates into Imai TE / ACME / ADE.

	Expects keys {"A", "B", "C"} at minimum; "D" is optional and informational.
	`rates_per_arm[arm]` must contain `refuse`, `fabricate`, `off_topic`.
	"""
	r = lambda arm: rates_per_arm[arm]["refuse"]
	te = r("B") - r("A")
	acme = r("B") - r("C")
	ade = r("C") - r("A")

	return MediationResult(
		te=te,
		acme=acme,
		ade=ade,
		te_check=acme + ade,
		refusal_rate={arm: rates_per_arm[arm]["refuse"] for arm in rates_per_arm},
		fabrication_rate={arm: rates_per_arm[arm]["fabricate"] for arm in rates_per_arm},
		off_topic_rate={arm: rates_per_arm[arm]["off_topic"] for arm in rates_per_arm},
		n_per_arm=n_per_arm,
		mediator_baseline=mediator_stats,
	)


def format_result(res: MediationResult) -> str:
	"""Pretty-print the mediation result for decision.txt."""
	lines = [
		"M4 — Imai 2010 mediation through unknown-entity SAE latent",
		"=" * 60,
		"",
		"Per-arm refusal / fabricate / off_topic rates (non-empty):",
	]
	for arm in res.refusal_rate:
		lines.append(
			f"  {arm}  refuse={res.refusal_rate[arm]:.3f}  "
			f"fab={res.fabrication_rate[arm]:.3f}  "
			f"off_topic={res.off_topic_rate[arm]:.3f}  "
			f"(n={res.n_per_arm.get(arm, '?')})"
		)
	lines.extend([
		"",
		f"Total effect of steering (TE):                 {res.te:+.4f}",
		f"Indirect / mediated effect (ACME = Y_B - Y_C): {res.acme:+.4f}",
		f"Direct effect (ADE = Y_C - Y_A):               {res.ade:+.4f}",
		f"Decomposition check (ACME + ADE vs TE):        {res.te_check:+.4f} vs {res.te:+.4f}",
		"",
		f"Mediator value stats (latent activation at last prompt token):",
	])
	for arm, val in res.mediator_baseline.items():
		lines.append(f"  {arm}: mean={val:.4f}")
	return "\n".join(lines)
