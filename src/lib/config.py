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
