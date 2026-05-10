"""Gemma Scope 2 SAE loader via sae_lens."""

from .config import load_config


def load_sae(layer: int | None = None, width: str | None = None, l0: str | None = None):
	"""Load a Gemma Scope 2 SAE for the configured layer.

	Returns (sae, cfg_dict, sparsity) per sae_lens convention.
	Defaults pulled from config.yaml; override per-call for sweeps.
	"""
	from sae_lens import SAE  # local import keeps the module importable on Windows w/o torch-cuda

	cfg = load_config()
	sae_cfg = cfg["sae"]
	layer = layer if layer is not None else sae_cfg["layer"]
	width = width or sae_cfg["width"]
	l0 = l0 or sae_cfg["l0_target"]

	sae_id = f"layer_{layer}/width_{width}/average_l0_{l0}"

	try:
		sae, cfg_dict, sparsity = SAE.from_pretrained(
			release=sae_cfg["release"],
			sae_id=sae_id,
		)
	except Exception as e:
		raise RuntimeError(
			f"failed to load SAE {sae_cfg['release']} / {sae_id}; "
			f"verify release name in sae_lens registry and SAE availability on HF"
		) from e

	return sae, cfg_dict, sparsity
