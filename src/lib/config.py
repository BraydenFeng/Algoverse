"""Load config.yaml with env-var overrides for paths."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config() -> dict:
	with open(CONFIG_PATH, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	# resolve paths relative to repo root unless overridden
	import os

	cfg["paths"]["data_dir"] = os.environ.get(
		"DC_DATA_DIR", str(REPO_ROOT / cfg["paths"]["data_dir"])
	)
	cfg["paths"]["outputs_dir"] = os.environ.get(
		"DC_OUTPUTS_DIR", str(REPO_ROOT / cfg["paths"]["outputs_dir"])
	)
	return cfg


def layer_suffix(cfg: dict | None = None, model_key: str = "primary") -> str:
	"""Per-layer subdirectory token.

	"primary" (Gemma)  -> "L21"          (backward-compat with all existing artifacts)
	"llama"            -> "llama_L21"    (Llama parallel track, isolated namespace)

	Used to namespace M1/M2/M3 outputs and HF-Hub paths so the same repo can
	hold artifacts from multiple extraction-layer runs and multiple models
	without overwriting. Both the steering vector and (Gemma-only) SAE mediator
	must live at the same layer for a given model_key in v2 scope.
	"""
	cfg = cfg if cfg is not None else load_config()
	layer = cfg["models"][model_key]["extraction_layer"]
	# preserve the pre-existing path scheme for "primary" so Gemma artifacts on
	# disk and on HF Hub don't need to be migrated
	if model_key == "primary":
		return f"L{layer}"
	return f"{model_key}_L{layer}"
