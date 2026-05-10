"""Module 1 — emotion vector extraction on Gemma-2-9B-IT.

v2 §3 Methodology:
	1. For each story, capture residual-stream activations.
	2. Average from token 50 onward -> per-story vector.
	3. Per-emotion mean across stories.
	4. Subtract cross-emotion mean (cancel shared narrative content).
	5. Project out top PCs of neutral corpus explaining 50% variance.
	6. l2-normalize.
	7. Extract at ~2/3 model depth.

Outputs to outputs/m1_vectors/{emotion}.npy + extraction_log.json.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

from .lib.config import load_config


@dataclass
class ExtractionResult:
	emotion: str
	vector: np.ndarray  # shape (d_model,)
	per_story_means: np.ndarray  # shape (n_stories, d_model)
	n_stories_used: int


def _capture_residual_stream(
	model, tokenizer, text: str, layer: int, token_skip: int = 50
) -> np.ndarray | None:
	"""Run text through model, return mean residual-stream activation from token `token_skip` onward.

	Returns None if the story has fewer than token_skip+1 tokens.
	"""
	inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(
		model.device
	)
	seq_len = inputs["input_ids"].shape[1]
	if seq_len <= token_skip:
		return None

	captured = {}

	def hook(module, args, output):
		# output is typically (hidden_states, ...) or just hidden_states depending on layer module
		hidden = output[0] if isinstance(output, tuple) else output
		captured["h"] = hidden.detach()

	# Gemma-2 layer module: model.model.layers[layer]
	target = model.model.layers[layer]
	handle = target.register_forward_hook(hook)
	try:
		with torch.no_grad():
			model(**inputs)
	finally:
		handle.remove()

	# captured["h"]: (1, seq_len, d_model). Average from token_skip onward.
	h = captured["h"][0, token_skip:, :].float().cpu().numpy()
	return h.mean(axis=0)


def _load_stories(emotion: str, data_dir: Path) -> list[str]:
	emotion_dir = data_dir / "stories" / emotion
	if not emotion_dir.exists():
		raise FileNotFoundError(
			f"no stories at {emotion_dir}; run generate_stories.generate_all() first"
		)
	paths = sorted(emotion_dir.glob("*.txt"))
	return [p.read_text(encoding="utf-8") for p in paths]


def _load_neutral_corpus(data_dir: Path) -> list[str]:
	"""Stories that are emotionally neutral. Used to derive PCs to project out.

	v2 protocol doesn't specify the corpus; following Anthropic's approach we use
	short pretraining-like passages without strong affect. Fallback: generate a small
	neutral corpus on the fly with the story generator using emotion='neutral'.
	"""
	neutral_dir = data_dir / "stories" / "neutral"
	if not neutral_dir.exists():
		raise FileNotFoundError(
			f"no neutral corpus at {neutral_dir}; "
			f"generate one with generate_stories.generate_emotion_corpus('neutral', ...)"
		)
	return [p.read_text(encoding="utf-8") for p in sorted(neutral_dir.glob("*.txt"))]


def _compute_projection_matrix(neutral_vecs: np.ndarray, target_var_explained: float) -> np.ndarray:
	"""Return (d_model, d_model) projection that removes top PCs explaining target_var fraction."""
	# center
	mu = neutral_vecs.mean(axis=0)
	centered = neutral_vecs - mu
	# SVD
	_, s, vh = np.linalg.svd(centered, full_matrices=False)
	var = s**2
	cum_var = np.cumsum(var) / var.sum()
	k = int(np.searchsorted(cum_var, target_var_explained)) + 1
	# top-k PCs as rows of vh
	top_pcs = vh[:k]  # (k, d_model)
	d = top_pcs.shape[1]
	# projector onto top PCs
	P_top = top_pcs.T @ top_pcs  # (d, d)
	# project-out operator
	return np.eye(d) - P_top


def extract_all(model, tokenizer) -> dict[str, ExtractionResult]:
	"""Run full v2 extraction protocol; return per-emotion vectors."""
	cfg = load_config()
	emotions = cfg["extraction"]["emotions"]
	layer = cfg["models"]["primary"]["extraction_layer"]
	token_skip = cfg["extraction"]["token_skip"]
	var_explained = cfg["extraction"]["pc_project_out_variance"]
	do_l2 = cfg["extraction"]["l2_normalize"]
	data_dir = Path(cfg["paths"]["data_dir"])
	out_dir = Path(cfg["paths"]["outputs_dir"]) / "m1_vectors"
	out_dir.mkdir(parents=True, exist_ok=True)

	# 1. capture per-story means for each emotion
	per_emotion_story_means: dict[str, np.ndarray] = {}
	for emotion in emotions:
		stories = _load_stories(emotion, data_dir)
		vecs = []
		for story in tqdm(stories, desc=f"capture/{emotion}"):
			v = _capture_residual_stream(model, tokenizer, story, layer, token_skip)
			if v is not None:
				vecs.append(v)
		if not vecs:
			raise RuntimeError(f"no usable stories for {emotion} (all shorter than token_skip)")
		per_emotion_story_means[emotion] = np.stack(vecs, axis=0)

	# 2. per-emotion mean
	per_emotion_mean = {
		e: per_emotion_story_means[e].mean(axis=0) for e in emotions
	}

	# 3. cross-emotion mean
	cross_mean = np.stack(list(per_emotion_mean.values()), axis=0).mean(axis=0)

	# 4. subtract cross-emotion mean
	centered = {e: per_emotion_mean[e] - cross_mean for e in emotions}

	# 5. project out top PCs of neutral corpus
	neutral_stories = _load_neutral_corpus(data_dir)
	neutral_vecs = []
	for story in tqdm(neutral_stories, desc="capture/neutral"):
		v = _capture_residual_stream(model, tokenizer, story, layer, token_skip)
		if v is not None:
			neutral_vecs.append(v)
	neutral_vecs = np.stack(neutral_vecs, axis=0)
	projector = _compute_projection_matrix(neutral_vecs, var_explained)
	projected = {e: projector @ centered[e] for e in emotions}

	# 6. l2-normalize
	def _norm(v: np.ndarray) -> np.ndarray:
		n = np.linalg.norm(v)
		return v / n if n > 0 else v

	final = {e: _norm(projected[e]) if do_l2 else projected[e] for e in emotions}

	# 7. save
	results: dict[str, ExtractionResult] = {}
	for e in emotions:
		vec_path = out_dir / f"{e}.npy"
		np.save(vec_path, final[e])
		results[e] = ExtractionResult(
			emotion=e,
			vector=final[e],
			per_story_means=per_emotion_story_means[e],
			n_stories_used=per_emotion_story_means[e].shape[0],
		)

	# how many neutral PCs were projected out — useful for sanity-checking the project-out step
	n_pcs_projected = int(np.linalg.matrix_rank(np.eye(projector.shape[0]) - projector))

	# log + cosine sanity check
	log = {
		"layer": layer,
		"token_skip": token_skip,
		"pc_project_out_variance": var_explained,
		"l2_normalize": do_l2,
		"n_stories": {e: results[e].n_stories_used for e in emotions},
		"n_neutral_used": int(neutral_vecs.shape[0]),
		"n_pcs_projected_out": n_pcs_projected,
		"d_model": int(final[emotions[0]].shape[0]),
	}
	# pairwise cosines — if desperation correlates > 0.5 with any control the control isn't clean
	cosines = {}
	for i, e1 in enumerate(emotions):
		for e2 in emotions[i + 1 :]:
			cos = float(final[e1] @ final[e2])
			cosines[f"{e1}|{e2}"] = cos
	log["pairwise_cosines"] = cosines

	(out_dir / "extraction_log.json").write_text(json.dumps(log, indent=2))
	return results
