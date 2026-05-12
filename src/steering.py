"""Residual-stream steering via forward hooks.

v2 §3 Methodology: add α × v_emotion to the residual stream at the extraction
layer during every forward pass. α is expressed as a fraction of the mean
residual-stream norm at that layer (Anthropic convention) so that α values
generalize across models.

Used by:
	- Module 2 (steering + MMLU capability check)
	- Module 3 (FaithEval dose-response, steered runs)
"""

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from tqdm import tqdm

from .lib.config import load_config


def load_emotion_vector(emotion: str) -> np.ndarray:
	"""Load the M1-extracted emotion vector from outputs/m1_vectors/{emotion}.npy.

	Raises FileNotFoundError with a clear M1-not-run message if the vector
	hasn't been produced yet.
	"""
	cfg = load_config()
	vec_path = Path(cfg["paths"]["outputs_dir"]) / "m1_vectors" / f"{emotion}.npy"
	if not vec_path.exists():
		raise FileNotFoundError(
			f"emotion vector not found at {vec_path}; run M1 (notebooks/m1_extract.ipynb) first"
		)
	v = np.load(vec_path)
	# defensive: M1 already ℓ₂-normalizes, but re-normalize so steering is invariant
	# to upstream changes
	n = np.linalg.norm(v)
	if n == 0:
		raise ValueError(f"emotion vector {vec_path} has zero norm; M1 output is broken")
	return v / n


def estimate_residual_norm(
	model,
	tokenizer,
	layer: int,
	calibration_texts: Iterable[str],
	*,
	token_skip: int = 50,
	max_length: int = 2048,
) -> float:
	"""Mean ℓ₂ norm of the residual stream at `layer`, averaged across tokens past
	`token_skip` for each calibration text.

	The scale returned here is what α multiplies against. Anthropic's protocol uses
	α as a fraction of this quantity so that the same α reads similarly across
	models with different residual magnitudes.
	"""
	norms: list[float] = []
	captured = {}

	def hook(module, args, output):
		h = output[0] if isinstance(output, tuple) else output
		captured["h"] = h.detach()

	target = model.model.layers[layer]
	handle = target.register_forward_hook(hook)
	try:
		for text in tqdm(list(calibration_texts), desc="calibrate/||h||"):
			inputs = tokenizer(
				text, return_tensors="pt", truncation=True, max_length=max_length
			).to(model.device)
			seq_len = inputs["input_ids"].shape[1]
			if seq_len <= token_skip:
				continue
			with torch.no_grad():
				model(**inputs)
			h = captured["h"][0, token_skip:, :].float()  # (T, d)
			# per-token norm, then mean — this is the convention; mean-of-norms not norm-of-mean
			norms.append(float(h.norm(dim=-1).mean().cpu()))
	finally:
		handle.remove()

	if not norms:
		raise RuntimeError("no calibration text was long enough; lower token_skip or pass longer texts")
	return float(np.mean(norms))


def make_steering_hook_factory(
	vector: np.ndarray,
	layer: int,
	alpha: float,
	norm_scale: float,
) -> Callable:
	"""Return a factory `f(model) -> handle` compatible with faitheval_eval.run_eval's
	`pre_forward_hook` parameter.

	Args:
		vector: (d_model,) emotion direction. Assumed unit-norm (load_emotion_vector
			guarantees this); a defensive re-normalization is applied anyway.
		layer: index into model.model.layers for the hook target.
		alpha: fraction of residual norm. v2 sweeps {0.025, 0.05, 0.075, 0.1}.
		norm_scale: mean residual norm at `layer`, from estimate_residual_norm.

	The injected vector has magnitude `alpha * norm_scale`, added to every token
	position of the residual stream at the chosen layer on every forward pass
	until the handle is removed.
	"""
	v_np = np.asarray(vector, dtype=np.float32)
	n = np.linalg.norm(v_np)
	if n == 0:
		raise ValueError("steering vector has zero norm")
	v_np = v_np / n
	magnitude = float(alpha * norm_scale)

	def factory(model):
		v = torch.tensor(v_np, dtype=model.dtype, device=model.device)
		steering = v * magnitude  # (d_model,) on device, broadcastable to (B, T, d)

		def hook(module, args, output):
			if isinstance(output, tuple):
				h = output[0] + steering
				return (h, *output[1:])
			return output + steering

		target = model.model.layers[layer]
		return target.register_forward_hook(hook)

	return factory


def make_ablation_hook_factory(
	vector: np.ndarray,
	layer: int,
) -> Callable:
	"""Return a factory that projects `vector`'s direction OUT of the residual stream
	at `layer`. Used by Module 3 ablation arm: h' = h − (h·v̂)v̂.

	Unlike steering, ablation has no α — it's a hard projection.
	"""
	v_np = np.asarray(vector, dtype=np.float32)
	n = np.linalg.norm(v_np)
	if n == 0:
		raise ValueError("ablation vector has zero norm")
	v_np = v_np / n

	def factory(model):
		v = torch.tensor(v_np, dtype=model.dtype, device=model.device)

		def hook(module, args, output):
			h = output[0] if isinstance(output, tuple) else output
			# (h · v̂) v̂ — broadcasts over (B, T, d)
			coef = (h * v).sum(dim=-1, keepdim=True)  # (B, T, 1)
			h_new = h - coef * v
			if isinstance(output, tuple):
				return (h_new, *output[1:])
			return h_new

		target = model.model.layers[layer]
		return target.register_forward_hook(hook)

	return factory
