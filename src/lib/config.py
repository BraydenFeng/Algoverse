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


def layer_suffix(cfg: dict | None = None) -> str:
	"""Per-layer subdirectory token, e.g. "L21".

	Used to namespace M1/M2/M3 outputs and HF-Hub paths so the same repo can
	hold artifacts from multiple extraction-layer runs without overwriting.
	Always reads the primary model's extraction_layer — both the steering
	vector and the SAE mediator must live at the same layer in v2 scope.
	"""
	cfg = cfg if cfg is not None else load_config()
	return f"L{cfg['models']['primary']['extraction_layer']}"
