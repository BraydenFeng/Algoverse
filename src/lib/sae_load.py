"""Gemma Scope SAE loader via sae_lens.

v2 uses Gemma Scope (original, for Gemma-2), NOT Gemma Scope 2 (for Gemma-3).
This is what Ferrando et al. 2025 used for the unknown-entity latent.
Only Module 4 (mechanism tracing) needs this; M1-M3 don't load SAEs.
"""

from .config import load_config


def load_sae(layer: int | None = None, width: str | None = None, l0: str | None = None):
	"""Load a Gemma Scope SAE for Gemma-2-9B-IT.

	Returns (sae, cfg_dict, sparsity) per sae_lens convention.
	Defaults pulled from config.yaml; override per-call to scan layers in M4.
	"""
	from sae_lens import SAE  # local import keeps the module importable w/o torch-cuda

	cfg = load_config()
	sae_cfg = cfg["sae"]
	layer = layer if layer is not None else sae_cfg["layer"]
	width = width or sae_cfg["width"]
	l0 = l0 or sae_cfg["l0_target"]

	# Gemma Scope SAE IDs follow: layer_{N}/width_{W}/{l0_id}
	# verify exact format from sae_lens registry: SAE.from_pretrained() will error w/ valid options
	sae_id = f"layer_{layer}/width_{width}/{l0}"

	try:
		sae, cfg_dict, sparsity = SAE.from_pretrained(
			release=sae_cfg["release"],
			sae_id=sae_id,
		)
	except Exception as e:
		raise RuntimeError(
			f"failed to load SAE {sae_cfg['release']} / {sae_id}; "
			f"verify release name + sae_id in sae_lens registry "
			f"(run `from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory; "
			f"print(get_pretrained_saes_directory())` to list)"
		) from e

	return sae, cfg_dict, sparsity
