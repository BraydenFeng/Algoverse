"""Gemma model loading. Gated on HF; expects HF_TOKEN in env."""

from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_config


_DTYPE_MAP = {
	"bfloat16": torch.bfloat16,
	"float16": torch.float16,
	"float32": torch.float32,
}


def load_gemma(variant: Literal["primary", "fallback_it", "sanity"] = "primary"):
	"""Return (model, tokenizer) for the requested config variant.

	variant:
		"primary"     -> Gemma-3-12B PT (default)
		"fallback_it" -> Gemma-3-12B IT (if M0.0 says PT refuses too rarely)
		"sanity"      -> Gemma-2-2B (smoke test on T4)
	"""
	cfg = load_config()
	model_cfg = cfg["models"][variant]
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
