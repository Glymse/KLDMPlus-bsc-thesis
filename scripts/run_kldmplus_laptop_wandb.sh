#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-configs/kldm_plus/mp_20/mp20_diffcsp_k_x0_soft_lattice_laptop.yaml}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-plus_mp20_diffcsp_k_x0_soft_lattice}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-plus_mp20_diffcsp_k_x0_soft_lattice_laptop_$(date +%Y%m%d_%H%M%S)}"

# Keep laptop runs light by default: metrics go to W&B, checkpoint artifacts do not.
LAPTOP_WANDB_CHECKPOINTS="${LAPTOP_WANDB_CHECKPOINTS:-false}"

TMP_CONFIG_DIR="${TMPDIR:-/tmp}/kldmplus_configs"
mkdir -p "$TMP_CONFIG_DIR"
TMP_CONFIG="${TMP_CONFIG_DIR}/${WANDB_RUN_NAME}.yaml"

"$PYTHON_BIN" - "$CONFIG_PATH" "$TMP_CONFIG" "$WANDB_PROJECT_NAME" "$WANDB_RUN_NAME" "$LAPTOP_WANDB_CHECKPOINTS" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
tmp_config = Path(sys.argv[2])
wandb_project = sys.argv[3]
wandb_run_name = sys.argv[4]
wandb_checkpoints = sys.argv[5].lower() in {"1", "true", "yes", "on"}

config = yaml.safe_load(config_path.read_text()) or {}
logging_cfg = dict(config.get("logging", {}) or {})
logging_cfg["wandb_project"] = wandb_project
logging_cfg["wandb_run_name"] = wandb_run_name
logging_cfg["wandb_checkpoints"] = wandb_checkpoints
config["logging"] = logging_cfg

tmp_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(f"prepared_config={tmp_config}")
print(f"wandb_project={wandb_project}")
print(f"wandb_run_name={wandb_run_name}")
print(f"wandb_checkpoints={wandb_checkpoints}")
PY

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

exec "$PYTHON_BIN" src/kldmPlus/run_experiment.py --config "$TMP_CONFIG"
