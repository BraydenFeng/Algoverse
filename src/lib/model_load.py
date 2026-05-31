"""HF model loading for the Gemma and Llama tracks. Both are gated; expects HF_TOKEN in env."""

from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_config


_DTYPE_MAP = {
	"bfloat16": torch.bfloat16,
	"float16": torch.float16,
	"float32": torch.float32,
}


def _load_model_by_key(model_key: str):
	cfg = load_config()
	model_cfg = cfg["models"][model_key]
	hf_id = model_cfg["hf_id"]
	dtype = _DTYPE_MAP[model_cfg["dtype"]]

	try:
		tokenizer = AutoTokenizer.from_pretrained(hf_id)
		model = AutoModelForCausalLM.from_pretrained(
			hf_id,
			torch_dtype=dtype,
			device_map="auto",
		)
	except Exception as e:
		raise RuntimeError(
			f"failed to load {hf_id}; check HF_TOKEN and that you've accepted the model license at https://huggingface.co/{hf_id}"
		) from e

	model.eval()
	return model, tokenizer


def load_gemma(variant: Literal["primary", "sanity"] = "primary"):
	"""Return (model, tokenizer) for the requested Gemma config variant.

	variant:
		"primary" -> Gemma-2-9B-IT (v2 target)
		"sanity"  -> Gemma-2-2B-IT (smoke test on T4)
	"""
	return _load_model_by_key(variant)


def load_llama():
	"""Return (model, tokenizer) for the Llama parallel track (Llama-3.1-8B-Instruct).

	Reads the "llama" key from config.yaml. Brayden-owned as of the 2026-05-29
	module reassignment (M4 → teammate, Llama M1-M3 → Brayden).
	"""
	return _load_model_by_key("llama")


def apply_chat_template(tokenizer, user_text: str) -> str:
	"""Wrap a user message in the model's chat template. v2 uses IT models, so chat formatting matters."""
	messages = [{"role": "user", "content": user_text}]
	return tokenizer.apply_chat_template(
		messages, tokenize=False, add_generation_prompt=True
	)
