from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from kldmPlus import run_experiment


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG = WORKSPACE_ROOT / "configs" / "kldm_plus" / "mp_20" / "mp20_plus_conv_sg_k_x0_em_quad_fixed_ema_a100.yaml"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "artifacts" / "HPC" / "compare_configs"


@dataclass(frozen=True)
class CompareVariant:
    name: str
    lambda_conv_sg: float
    conv_enabled: bool
    control_mode: str = "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential KLDMPlus k-x0 conventional-SG ablation.")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG), help="Base YAML config to clone.")
    parser.add_argument("--project", default="kldm_plus_ablation", help="WandB project for all ablation runs.")
    parser.add_argument("--max-epochs", type=int, default=2000, help="Epochs per ablation run.")
    parser.add_argument("--val-seeds", type=int, default=10, help="Number of validation sampling seeds at final validation.")
    parser.add_argument("--sampling-max-graphs", type=int, default=1024, help="Max graphs for final seeded sampling validation.")
    parser.add_argument("--sym-lambdas", type=float, nargs="*", default=[1.0, 3.0], help="Real conventional-SG lambda values.")
    parser.add_argument("--fake-control-lambda", type=float, default=1.0, help="lambda_conv_sg for shuffled fake-control.")
    parser.add_argument("--skip-no-sym", action="store_true", help="Do not run the lambda=0 no-symmetry control.")
    parser.add_argument("--skip-real-sym", action="store_true", help="Do not run real conventional-SG lambda variants.")
    parser.add_argument("--skip-fake-control", action="store_true", help="Do not run the shuffled SG/conv_C fake-control.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Where derived configs are written.")
    return parser.parse_args()


def lambda_tag(value: float) -> str:
    text = f"{float(value):g}".replace("-", "neg").replace(".", "p")
    return f"lam{text}"


def make_variants(
    *,
    sym_lambdas: list[float],
    fake_control_lambda: float,
    skip_no_sym: bool,
    skip_real_sym: bool,
    skip_fake_control: bool,
) -> list[CompareVariant]:
    variants = []
    if not skip_no_sym:
        variants.append(
            CompareVariant(
                name="k_x0_no_sym",
                lambda_conv_sg=0.0,
                conv_enabled=False,
            )
        )
    if not skip_real_sym:
        variants.extend(
            CompareVariant(
                name=f"k_x0_convsg_{lambda_tag(value)}",
                lambda_conv_sg=float(value),
                conv_enabled=True,
            )
            for value in sym_lambdas
        )
    if not skip_fake_control:
        variants.append(
            CompareVariant(
                name=f"k_x0_convsg_{lambda_tag(fake_control_lambda)}_fake_shuffle",
                lambda_conv_sg=float(fake_control_lambda),
                conv_enabled=True,
                control_mode="shuffle_batch",
            )
        )
    if not variants:
        raise ValueError("No ablation variants selected. Disable fewer --skip-* flags or provide --sym-lambdas.")
    return variants


def make_compare_config(
    base_config: dict,
    *,
    variant: CompareVariant,
    project: str,
    max_epochs: int,
    val_seeds: int,
    sampling_max_graphs: int | None,
) -> dict:
    config = deepcopy(base_config)
    experiment_name = f"plus_mp20_ablation_{variant.name}"

    config.pop("sampler_config", None)
    config["experiment_name"] = experiment_name

    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg["lattice_representation"] = "diffcsp_k"
    dataset_cfg["train_subset_seed"] = 2002

    model_cfg = config.setdefault("model", {})
    model_cfg["lattice_representation"] = "diffcsp_k"
    model_cfg["lattice_parameterization"] = "x0"
    model_cfg["lambda_l"] = 1.0
    model_cfg["lattice_sg_lambda"] = 0.0
    model_cfg["lattice_debug"] = False
    conv_cfg = model_cfg.setdefault("conv_sg_aux", {})
    conv_cfg["enabled"] = bool(variant.conv_enabled)
    conv_cfg["mode"] = "direct"
    conv_cfg["lambda"] = float(variant.lambda_conv_sg)
    conv_cfg["time_weight"] = "alpha2"
    conv_cfg["require_valid_transform"] = True
    conv_cfg["control_mode"] = str(variant.control_mode)

    ema_cfg = config.setdefault("ema", {})
    ema_cfg["enabled"] = True
    ema_cfg["type"] = "fixed"
    ema_cfg["decay"] = 0.999
    ema_cfg["start_epoch"] = 500
    ema_cfg.pop("gamma", None)

    config.setdefault("training", {})["max_epochs"] = int(max_epochs)

    logging_cfg = config.setdefault("logging", {})
    logging_cfg["wandb_project"] = str(project)
    logging_cfg["wandb_run_name"] = experiment_name
    logging_cfg["wandb_checkpoints"] = True
    logging_cfg["train_every_epochs"] = 1

    validation_cfg = config.setdefault("validation", {})
    validation_cfg["every_n_epochs"] = int(max_epochs)
    validation_cfg["sampling_seed"] = 0
    validation_cfg["num_sampling_seeds"] = int(val_seeds)
    validation_cfg["loss_seed"] = 2019
    validation_cfg["ema_val"] = True
    validation_cfg["subset_size"] = None
    validation_cfg["subset_seed"] = 123
    validation_cfg["sampling_max_graphs"] = sampling_max_graphs

    checkpoint_cfg = config.setdefault("checkpoint", {})
    checkpoint_cfg["resume_from"] = None
    checkpoint_cfg["wandb_resume_id"] = None
    checkpoint_cfg["prune_wandb_artifact_cache"] = True

    return config


def main() -> None:
    args = parse_args()
    base_path, base_config = run_experiment.load_experiment_config(args.base_config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = make_variants(
        sym_lambdas=[float(value) for value in args.sym_lambdas],
        fake_control_lambda=float(args.fake_control_lambda),
        skip_no_sym=bool(args.skip_no_sym),
        skip_real_sym=bool(args.skip_real_sym),
        skip_fake_control=bool(args.skip_fake_control),
    )

    print(
        "compare_start "
        f"base_config={base_path} project={args.project} "
        f"max_epochs={args.max_epochs} val_seeds={args.val_seeds} "
        f"sampling_max_graphs={args.sampling_max_graphs} "
        f"variants={','.join(variant.name for variant in variants)}",
        flush=True,
    )

    for variant in variants:
        config = make_compare_config(
            base_config,
            variant=variant,
            project=str(args.project),
            max_epochs=int(args.max_epochs),
            val_seeds=int(args.val_seeds),
            sampling_max_graphs=None if args.sampling_max_graphs <= 0 else int(args.sampling_max_graphs),
        )
        config_path = output_dir / f"mp20_ablation_{variant.name}.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        print(
            "compare_run_start "
            f"variant={variant.name} "
            f"lambda_conv_sg={float(variant.lambda_conv_sg):g} "
            f"control_mode={variant.control_mode} "
            f"experiment={config['experiment_name']} config={config_path}",
            flush=True,
        )
        runner = run_experiment.ExperimentRunner(config_path)
        runner.run_training_loop()
        print(
            "compare_run_done "
            f"variant={variant.name} "
            f"lambda_conv_sg={float(variant.lambda_conv_sg):g} "
            f"experiment={config['experiment_name']}",
            flush=True,
        )
        if run_experiment.STOP_REQUESTED:
            print("compare_stop_requested stopping_before_next_lambda", flush=True)
            break

    print("compare_done", flush=True)


if __name__ == "__main__":
    main()
