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


def load_gemma(variant: Literal["primary", "sanity"] = "primary"):
	"""Return (model, tokenizer) for the requested config variant.

	variant:
		"primary" -> Gemma-2-9B-IT (v2 target)
		"sanity"  -> Gemma-2-2B-IT (smoke test on T4)
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


def apply_chat_template(tokenizer, user_text: str) -> str:
	"""Wrap a user message in Gemma's chat template. v2 uses IT models, so chat formatting matters."""
	messages = [{"role": "user", "content": user_text}]
	return tokenizer.apply_chat_template(
		messages, tokenize=False, add_generation_prompt=True
	)
