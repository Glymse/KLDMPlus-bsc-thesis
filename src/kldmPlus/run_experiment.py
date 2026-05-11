from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import random
import signal
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from kldmPlus.utils.device import get_default_device
from kldmPlus.utils.time import sample_times
from kldmPlus.utils.time_sampler import (
    AdaptiveReinforceTimeSampler,
    AdaptiveReinforcePaperTimeSampler,
    AdaptiveReinforceVelocityFStatTimeSampler,
    KLDMUniformTimeSampler,
    LossSecondMomentTimeSampler,
)

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise ImportError("wandb is required for src/kldmPlus/run_experiment.py") from exc


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS_ROOT = WORKSPACE_ROOT / "artifacts" / "HPC" / "checkpoints" / "experiments"
WANDB_ARTIFACTS_ROOT = WORKSPACE_ROOT / "artifacts" / "HPC" / "wandb_artifacts"
TIME_LOWER_BOUND = 1e-3
STOP_REQUESTED = False
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TRAIN_SEED = 2002


def set_global_training_seed(seed: int = TRAIN_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker_factory(base_seed: int):
    def seed_worker(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return seed_worker


def _request_stop(_signum=None, _frame=None) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGTERM, _request_stop)
signal.signal(signal.SIGINT, _request_stop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a KLDM experiment from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file.")
    return parser.parse_args()


def load_experiment_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    # Load the main config once, then inline the sampler config so the rest of
    # the runner can always read from config["sampler"] when a training config
    # points to a separate sampler file.
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if "sampler" not in config and "sampler_config" in config:
        with (config_path.parent / str(config["sampler_config"])).expanduser().resolve().open("r", encoding="utf-8") as handle:
            config["sampler"] = yaml.safe_load(handle) or {}

    return config_path, config


def make_fixed_subset(dataset, subset_size: int | None, seed: int) -> Any:
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def should_stop(run) -> bool:
    if STOP_REQUESTED:
        return True
    if run is None:
        return False
    for attr in ("stopped", "_stopped"):
        value = getattr(run, attr, None)
        if isinstance(value, bool) and value:
            return True
    return False


def build_run_name() -> str:
    now = datetime.now()
    return f"trial_{now.strftime('%Y%m%d')}"


def _parse_wandb_artifact_url(reference: str) -> tuple[str, str, str, str, str, str] | None:
    parsed = urlparse(reference)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc not in {"wandb.ai", "www.wandb.ai"}:
        return None

    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 8:
        return None
    if parts[2] != "artifacts" or parts[6] != "files":
        return None

    entity = parts[0]
    project = parts[1]
    artifact_type = parts[3]
    artifact_name = parts[4]
    artifact_version = parts[5]
    file_path = "/".join(parts[7:])
    if not file_path:
        return None
    return entity, project, artifact_type, artifact_name, artifact_version, file_path


def resolve_checkpoint_reference(reference: str | Path, *, config_path: Path) -> Path:
    reference_str = str(reference)
    artifact_ref = _parse_wandb_artifact_url(reference_str)
    if artifact_ref is not None:
        entity, project, artifact_type, artifact_name, artifact_version, file_path = artifact_ref
        artifact_spec = f"{entity}/{project}/{artifact_name}:{artifact_version}"
        download_root = (
            WANDB_ARTIFACTS_ROOT
            / entity
            / project
            / artifact_type
            / artifact_name
            / artifact_version
        )
        expected_path = download_root / file_path
        if expected_path.exists():
            return expected_path.resolve()

        artifact = wandb.Api().artifact(artifact_spec, type=artifact_type)
        artifact_dir = Path(artifact.download(root=str(download_root)))
        checkpoint_path = artifact_dir / file_path
        if checkpoint_path.exists():
            print(
                f"checkpoint_downloaded=wandb artifact={artifact_spec} file={checkpoint_path}",
                flush=True,
            )
            return checkpoint_path.resolve()

        matches = sorted(artifact_dir.rglob(Path(file_path).name))
        if len(matches) == 1:
            chosen = matches[0].resolve()
            print(
                f"checkpoint_downloaded=wandb artifact={artifact_spec} file={chosen}",
                flush=True,
            )
            return chosen

        raise FileNotFoundError(
            "Downloaded WandB artifact but could not locate the requested checkpoint file: "
            f"artifact={artifact_spec} requested_file={file_path} download_dir={artifact_dir}"
        )

    candidate = Path(reference_str).expanduser()
    if not candidate.is_absolute():
        candidate = (config_path.parent / candidate).expanduser()
    candidate = candidate.resolve()
    if candidate.exists():
        return candidate

    if candidate.parent.exists():
        options = sorted(candidate.parent.glob("*.pt"))
        if options:
            chosen = options[-1].resolve()
            print(f"checkpoint_missing={candidate} fallback_latest={chosen}", flush=True)
            return chosen
    return candidate


def format_metric(value: float | int | None, fmt: str) -> str:
    if value is None:
        return "na"
    return format(value, fmt)


def set_validation_sampling_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_dir(config: dict[str, Any], experiment_name: str) -> Path:
    del config
    return CHECKPOINTS_ROOT / experiment_name


def save_named_checkpoint(
    *,
    model,
    optimizer: torch.optim.Optimizer,
    ema,
    time_sampler,
    config: dict[str, Any],
    experiment_name: str,
    epoch: int,
    metrics: Mapping[str, float | int | None],
    filename: str,
    keep_paths: list[Path] | None = None,
) -> Path:
    from kldmPlus.utils.model_loader import save_checkpoint

    output_dir = checkpoint_dir(config=config, experiment_name=experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    if output_path.exists():
        output_path.unlink()
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        ema=ema,
        time_sampler=time_sampler,
        output_path=output_path,
        config=config,
        epoch=epoch,
        metrics=metrics,
    )
    keep_names = {output_path.name}
    if keep_paths is not None:
        keep_names.update(path.name for path in keep_paths)
    for candidate in output_dir.iterdir():
        if candidate.is_file() and candidate.name not in keep_names:
            candidate.unlink(missing_ok=True)
    return output_path


def save_wandb_checkpoint(path: Path) -> None:
    if path.exists():
        wandb.save(str(path), policy="now")


class ExperimentRunner:
    def __init__(self, config_path: str | Path) -> None:
        from kldmPlus.utils.model_loader import build_training_components, load_checkpoint

        # -------------------------------------------------
        # Static experiment setup from config
        # -------------------------------------------------
        self.config_path, self.config = load_experiment_config(config_path)
        self.experiment_name = str(self.config["experiment_name"])

        self.sampler_cfg = dict(self.config["sampler"])
        self.logging_cfg = dict(self.config["logging"])
        self.validation_cfg = dict(self.config["validation"])
        self.checkpoint_cfg = dict(self.config["checkpoint"])
        self.wandb_project = str(self.logging_cfg.get("wandb_project", self.experiment_name))
        self.wandb_run_name = str(self.logging_cfg.get("wandb_run_name", self.experiment_name))
        set_global_training_seed(TRAIN_SEED)

        self.train_every_epochs = int(self.logging_cfg["train_every_epochs"])
        self.validate_every_epochs = int(self.validation_cfg["every_n_epochs"])

        # -------------------------------------------------
        # Runtime objects: device, data, model, optimizer, EMA
        # -------------------------------------------------
        self.device = get_default_device()
        self.train_loader, self.val_loader, self.lattice_transform = self.create_loaders()
        self.time_sampler = self.build_time_sampler()
        self._inject_mattergen_lattice_stats()

        self.model, self.optimizer, self.ema = build_training_components(
            config=self.config,
            device=self.device,
        )

        self.start_epoch = 0
        self.run = None
        self._last_validation_artifact = None
        self._warned_missing_wandb_image = False
        self.validation_ablation_history: dict[str, list[float]] = {
            "epoch": [],
            "rmse_mean": [],
            "rmse_std": [],
            "match_rate_mean": [],
            "match_rate_std": [],
            "valid_mean": [],
            "valid_std": [],
        }

        # Optional resume path for continuing training from a saved checkpoint.
        resume_from = self.checkpoint_cfg["resume_from"]
        if resume_from:
            checkpoint = load_checkpoint(
                checkpoint_path=resolve_checkpoint_reference(
                    resume_from,
                    config_path=self.config_path,
                ),
                model=self.model,
                optimizer=self.optimizer,
                ema=self.ema,
                device=self.device,
                prefer_ema_weights=False,
            )
            self.start_epoch = int(checkpoint["epoch"])
            self.time_sampler.load_state_dict(checkpoint.get("time_sampler_state_dict"))

    def build_time_sampler(self):
        cfg = dict(self.config.get("time_sampler", {}))
        sampler_type = str(cfg.get("type", "uniform"))

        if sampler_type == "uniform":
            return KLDMUniformTimeSampler(
                lower_bound=TIME_LOWER_BOUND,
                seed=TRAIN_SEED,
            )

        if sampler_type == "loss_second_moment":
            return LossSecondMomentTimeSampler(
                n_bins=int(cfg.get("n_bins", 64)),
                lower_bound=TIME_LOWER_BOUND,
                history_per_bin=int(cfg.get("history_per_bin", 10)),
                alpha=float(cfg.get("alpha", 0.5)),
                adaptive_power=float(cfg.get("adaptive_power", 0.5)),
                min_prob=float(cfg.get("min_prob", 0.002)),
                max_prob=float(cfg.get("max_prob", 0.10)),
                velocity_weight=float(cfg.get("velocity_weight", 0.7)),
                lattice_weight=float(cfg.get("lattice_weight", 0.3)),
                use_importance_weights=bool(cfg.get("use_importance_weights", False)),
                clip_importance_weights=bool(cfg.get("clip_importance_weights", True)),
                weight_clip_min=float(cfg.get("weight_clip_min", 0.5)),
                weight_clip_max=float(cfg.get("weight_clip_max", 2.0)),
                seed=TRAIN_SEED,
                device=self.device,
            )

        if sampler_type == "adaptive_reinforce":
            policy_warmup_steps = int(cfg.get("policy_warmup_steps", 5000))
            policy_warmup_epochs = cfg.get("policy_warmup_epochs")
            if policy_warmup_epochs is not None:
                batches_per_epoch = len(self.train_loader)
                policy_warmup_steps = int(policy_warmup_epochs) * batches_per_epoch
                self.config.setdefault("time_sampler", {})
                self.config["time_sampler"]["policy_warmup_steps_resolved"] = policy_warmup_steps
            return AdaptiveReinforceTimeSampler(
                lower_bound=TIME_LOWER_BOUND,
                policy_hidden_dim=int(cfg.get("policy_hidden_dim", 128)),
                policy_hidden_depth=int(cfg.get("policy_hidden_depth", 2)),
                min_concentration=float(cfg.get("min_concentration", 0.25)),
                policy_lr=float(cfg.get("policy_lr", 2e-5)),
                entropy_coef=float(cfg.get("entropy_coef", 1e-2)),
                policy_update_every=int(cfg.get("policy_update_every", 100)),
                policy_warmup_steps=policy_warmup_steps,
                reward_candidate_times=int(cfg.get("reward_candidate_times", 7)),
                reward_active_times=int(cfg.get("reward_active_times", 5)),
                reward_history_size=int(cfg.get("reward_history_size", 64)),
                use_baseline=bool(cfg.get("use_baseline", False)),
                reward_baseline_momentum=float(cfg.get("reward_baseline_momentum", 0.95)),
                reward_velocity_weight=float(cfg.get("reward_velocity_weight", 1.0)),
                reward_lattice_weight=float(cfg.get("reward_lattice_weight", 1.0)),
                reward_size_weight_power=float(cfg.get("reward_size_weight_power", 0.0)),
                reward_size_weight_max=float(cfg.get("reward_size_weight_max", 2.0)),
                reward_normalization_eps=float(cfg.get("reward_normalization_eps", 1e-6)),
                entropy_in_reward=bool(cfg.get("entropy_in_reward", True)),
                use_importance_weights=bool(cfg.get("use_importance_weights", True)),
                clip_importance_weights=bool(cfg.get("clip_importance_weights", True)),
                weight_clip_min=float(cfg.get("weight_clip_min", 0.25)),
                weight_clip_max=float(cfg.get("weight_clip_max", 4.0)),
                gradient_clip_norm=float(cfg.get("gradient_clip_norm", 1.0)),
                feature_selection_min_history=int(cfg.get("feature_selection_min_history", 32)),
                seed=TRAIN_SEED,
                device=self.device,
                reward_probe_times=cfg.get("reward_probe_times"),
            )

        if sampler_type == "adaptive_reinforce_paper":
            policy_warmup_steps = int(cfg.get("policy_warmup_steps", 0))
            policy_warmup_epochs = cfg.get("policy_warmup_epochs")
            if policy_warmup_epochs is not None:
                batches_per_epoch = len(self.train_loader)
                policy_warmup_steps = int(policy_warmup_epochs) * batches_per_epoch
                self.config.setdefault("time_sampler", {})
                self.config["time_sampler"]["policy_warmup_steps_resolved"] = policy_warmup_steps
            return AdaptiveReinforcePaperTimeSampler(
                lower_bound=TIME_LOWER_BOUND,
                policy_hidden_dim=int(cfg.get("policy_hidden_dim", 256)),
                policy_hidden_depth=int(cfg.get("policy_hidden_depth", 2)),
                min_concentration=float(cfg.get("min_concentration", 1e-5)),
                policy_lr=float(cfg.get("policy_lr", 1e-2)),
                entropy_coef=float(cfg.get("entropy_coef", 1e-2)),
                policy_update_every=int(cfg.get("policy_update_every", 40)),
                policy_warmup_steps=policy_warmup_steps,
                reward_candidate_times=int(cfg.get("reward_candidate_times", 13)),
                reward_active_times=int(cfg.get("reward_active_times", 3)),
                reward_history_size=int(cfg.get("reward_history_size", 5)),
                feature_selection_min_history=int(cfg.get("feature_selection_min_history", 2)),
                feature_queue_single_graph=bool(cfg.get("feature_queue_single_graph", True)),
                gradient_clip_norm=float(cfg.get("gradient_clip_norm", 1.0)),
                seed=TRAIN_SEED,
                device=self.device,
                reward_probe_times=cfg.get("reward_probe_times"),
            )

        if sampler_type == "adaptive_reinforce_kldm_velocity_fstat":
            policy_warmup_steps = int(cfg.get("policy_warmup_steps", 5000))
            policy_warmup_epochs = cfg.get("policy_warmup_epochs")
            if policy_warmup_epochs is not None:
                batches_per_epoch = len(self.train_loader)
                policy_warmup_steps = int(policy_warmup_epochs) * batches_per_epoch
                self.config.setdefault("time_sampler", {})
                self.config["time_sampler"]["policy_warmup_steps_resolved"] = policy_warmup_steps
            return AdaptiveReinforceVelocityFStatTimeSampler(
                lower_bound=TIME_LOWER_BOUND,
                policy_hidden_dim=int(cfg.get("policy_hidden_dim", 128)),
                policy_hidden_depth=int(cfg.get("policy_hidden_depth", 2)),
                min_concentration=float(cfg.get("min_concentration", 0.25)),
                policy_lr=float(cfg.get("policy_lr", 1e-4)),
                entropy_coef=float(cfg.get("entropy_coef", 3e-3)),
                policy_update_every=int(cfg.get("policy_update_every", 80)),
                policy_warmup_steps=policy_warmup_steps,
                reward_candidate_times=int(cfg.get("reward_candidate_times", 7)),
                reward_active_times=int(cfg.get("reward_active_times", 3)),
                reward_history_size=int(cfg.get("reward_history_size", 128)),
                use_baseline=bool(cfg.get("use_baseline", True)),
                reward_baseline_momentum=float(cfg.get("reward_baseline_momentum", 0.9)),
                reward_velocity_weight=float(cfg.get("reward_velocity_weight", 1.0)),
                reward_lattice_weight=float(cfg.get("reward_lattice_weight", 0.05)),
                reward_forgetting_penalty=float(cfg.get("reward_forgetting_penalty", 0.25)),
                reward_size_weight_power=float(cfg.get("reward_size_weight_power", 0.0)),
                reward_size_weight_max=float(cfg.get("reward_size_weight_max", 2.0)),
                reward_normalization_eps=float(cfg.get("reward_normalization_eps", 1e-6)),
                entropy_in_reward=bool(cfg.get("entropy_in_reward", False)),
                use_importance_weights=bool(cfg.get("use_importance_weights", True)),
                clip_importance_weights=bool(cfg.get("clip_importance_weights", True)),
                weight_clip_min=float(cfg.get("weight_clip_min", 0.5)),
                weight_clip_max=float(cfg.get("weight_clip_max", 2.0)),
                normalize_importance_weights=bool(
                    cfg.get("normalize_importance_weights", True)
                ),
                gradient_clip_norm=float(cfg.get("gradient_clip_norm", 1.0)),
                feature_selection_min_history=int(cfg.get("feature_selection_min_history", 32)),
                feature_selection_method=str(cfg.get("feature_selection_method", "f_stat")),
                seed=TRAIN_SEED,
                device=self.device,
                reward_probe_times=cfg.get("reward_probe_times"),
            )

        raise ValueError(f"Unknown time_sampler.type={sampler_type!r}")

    def create_loaders(self) -> tuple[DataLoader, DataLoader, Any]:
        from kldmPlus.data import CSPTask, resolve_data_root
        from kldmPlus.data.csp import validate_lattice_configuration

        dataset_cfg = dict(self.config["dataset"])
        model_cfg = dict(self.config["model"])
        # Guard against silent representation/diffusion mismatches before the
        # runner creates datasets or starts training.
        validate_lattice_configuration(
            lattice_representation=str(dataset_cfg.get("lattice_representation", "kldm")),
            lattice_parameterization=str(model_cfg["lattice_parameterization"]),
            lattice_diffusion_type=str(model_cfg.get("lattice_diffusion_type", "VP")),
        )

        task = CSPTask(
            dataset_name=str(dataset_cfg["name"]),
            lattice_parameterization=str(model_cfg["lattice_parameterization"]),
            lattice_representation=str(dataset_cfg.get("lattice_representation", "kldm")),
        )

        root = resolve_data_root(dataset_cfg["root"])
        batch_size = int(dataset_cfg["batch_size"])
        num_workers = int(dataset_cfg["num_workers"])
        pin_memory = bool(dataset_cfg["pin_memory"])
        train_generator = torch.Generator().manual_seed(TRAIN_SEED)
        worker_init_fn = _seed_worker_factory(TRAIN_SEED)

        # Keep training/validation splits fixed so experiment metrics are always
        # comparable across runs.
        train_loader = task.dataloader(
            root=root,
            split=TRAIN_SPLIT,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            download=True,
            generator=train_generator,
            worker_init_fn=worker_init_fn,
        )

        # Validation always uses the validation split, optionally with a fixed
        # subset for faster checks.
        val_dataset_full = task.fit_dataset(root=root, split=VAL_SPLIT, download=True)
        val_dataset = make_fixed_subset(
            val_dataset_full,
            subset_size=self.validation_cfg["subset_size"],
            seed=int(self.validation_cfg["subset_seed"]),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            collate_fn=val_dataset_full.collate_fn,
        )

        return train_loader, val_loader, task.make_lattice_transform(
            root=root,
            download=True,
            mattergen_limit_var_scaling_constant=model_cfg.get("mattergen_limit_var_scaling_constant"),
        )

    def _inject_mattergen_lattice_stats(self) -> None:
        if getattr(self.lattice_transform, "representation", None) != "mattergen":
            return
        if not hasattr(self.lattice_transform, "stats"):
            return
        c, nu = self.lattice_transform.stats()
        self.config.setdefault("model", {})
        self.config["model"]["mattergen_lattice_c"] = float(c)
        self.config["model"]["mattergen_lattice_nu"] = float(nu)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()

        total_loss_v = total_loss_l = total_loss_weighted = 0.0
        total_graphs = 0

        for batch in self.train_loader:
            if STOP_REQUESTED:
                break

            batch = batch.to(self.device)
            self.time_sampler.before_model_update(batch=batch, model=self.model)

            sampled_time = self.time_sampler.sample(batch)

            self.optimizer.zero_grad(set_to_none=True)
            loss, metrics = self.model.algorithm2_loss(
                batch=batch,
                t=sampled_time.t,
                time_weight=sampled_time.weights,
                debug=False,
            )
            loss.backward()
            self.optimizer.step()
            self.time_sampler.after_model_update(
                batch=batch,
                model=self.model,
                sampled_time=sampled_time,
                metrics=metrics,
            )

            if self.ema is not None:
                self.ema.update(self.model, current_epoch=epoch)

            total_loss_weighted += float(metrics["loss"]) * int(batch.num_graphs)
            total_loss_v += float(metrics["loss_v"]) * int(batch.num_graphs)
            total_loss_l += float(metrics["loss_l"]) * int(batch.num_graphs)
            total_graphs += int(batch.num_graphs)

        if total_graphs == 0:
            raise RuntimeError("Training stopped before any batches were processed.")

        metrics = {
            "loss": total_loss_weighted / total_graphs,
            "loss_v": total_loss_v / total_graphs,
            "loss_l": total_loss_l / total_graphs,
            "loss_weighted": total_loss_weighted / total_graphs,
        }
        metrics.update(self.time_sampler.diagnostics())
        return metrics

    def evaluate_loss(self) -> dict[str, float]:
        self.model.eval()

        total_loss_v = total_loss_l = total_loss_weighted = 0.0
        total_graphs = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)

            # Validation uses the same noisy-time sampling pattern as training.
            t_graph = sample_times(batch, lower_bound=TIME_LOWER_BOUND)

            with torch.no_grad():
                _, metrics = self.model.algorithm2_loss(batch=batch, t=t_graph, debug=False)

            total_loss_weighted += float(metrics["loss"]) * int(batch.num_graphs)
            total_loss_v += float(metrics["loss_v"]) * int(batch.num_graphs)
            total_loss_l += float(metrics["loss_l"]) * int(batch.num_graphs)
            total_graphs += int(batch.num_graphs)

        if total_graphs == 0:
            raise RuntimeError("Validation loader is empty.")

        return {
            "loss": total_loss_weighted / total_graphs,
            "loss_v": total_loss_v / total_graphs,
            "loss_l": total_loss_l / total_graphs,
            "loss_weighted": total_loss_weighted / total_graphs,
        }

    def run_sampling_evaluation(self) -> dict[str, Any]:
        from kldmPlus.sample_evaluation.sample_evaluation import (
            aggregate_csp_reconstruction_metrics,
            evaluate_csp_reconstruction,
        )

        self.model.eval()

        def collect_one_pass(*, seed: int | None = None) -> dict[str, Any]:
            if seed is not None:
                set_validation_sampling_seed(seed)
                print(f"validation_sampling_seed={seed} pass_start", flush=True)

            results = []
            num_graphs_seen = 0

            for batch in self.val_loader:
                batch = batch.to(self.device)

                with torch.no_grad():
                    sample_fn = (
                        self.model.sample_CSP_algorithm4
                        if str(self.sampler_cfg["method"]) == "pc"
                        else self.model.sample_CSP_algorithm3
                    )

                    sample_kwargs = {
                        "n_steps": int(self.sampler_cfg["n_steps"]),
                        "batch": batch,
                        "t_start": float(self.sampler_cfg["t_start"]),
                        "t_final": float(self.sampler_cfg["t_final"]),
                    }
                    if str(self.sampler_cfg["method"]) == "pc":
                        sample_kwargs["tau"] = float(self.sampler_cfg["tau"])
                        sample_kwargs["n_correction_steps"] = int(self.sampler_cfg["n_correction_steps"])

                    pos_t, _v_t, l_t, h_t = sample_fn(**sample_kwargs)

                ptr = batch.ptr.tolist()
                for graph_idx, (start_idx, end_idx) in enumerate(zip(ptr[:-1], ptr[1:])):
                    results.append(
                        evaluate_csp_reconstruction(
                            pred_f=pos_t[start_idx:end_idx],
                            pred_l=l_t[graph_idx],
                            pred_a=h_t[start_idx:end_idx],
                            target_f=batch.pos[start_idx:end_idx],
                            target_l=batch.l[graph_idx],
                            target_a=batch.atomic_numbers[start_idx:end_idx],
                            lattice_transform=self.lattice_transform,
                        )
                    )
                    num_graphs_seen += 1

                    if (
                        self.validation_cfg["sampling_max_graphs"] is not None
                        and num_graphs_seen >= self.validation_cfg["sampling_max_graphs"]
                    ):
                        break

                if (
                    self.validation_cfg["sampling_max_graphs"] is not None
                    and num_graphs_seen >= self.validation_cfg["sampling_max_graphs"]
                ):
                    break

            if seed is not None:
                print(
                    f"validation_sampling_seed={seed} pass_done num_graphs={len(results)}",
                    flush=True,
                )

            summary = aggregate_csp_reconstruction_metrics(results)
            return {
                "valid": summary.get("valid"),
                "match_rate": summary.get("match_rate"),
                "rmse": summary.get("rmse"),
                "num_samples": summary.get("num_samples"),
            }

        if not bool(self.validation_cfg.get("ablation", False)):
            return collect_one_pass()

        num_seeds = int(self.validation_cfg.get("ablation_num_seeds", 6))
        seed_offset = int(self.validation_cfg.get("ablation_seed_offset", 0))
        pass_summaries = []
        seed_summaries = []
        for seed_index in range(num_seeds):
            seed = seed_offset + seed_index
            print(
                f"validation_ablation_progress seed={seed_index + 1}/{num_seeds} actual_seed={seed}",
                flush=True,
            )
            one_pass = collect_one_pass(seed=seed)
            print(
                f"validation_ablation_seed_summary seed={seed} "
                f"valid={format_metric(one_pass['valid'], '.4f')} "
                f"match_rate={format_metric(one_pass['match_rate'], '.4f')} "
                f"rmse={format_metric(one_pass['rmse'], '.6f')}",
                flush=True,
            )
            pass_summaries.append(one_pass)
            seed_summaries.append({"seed": seed, **one_pass})

        def aggregate_scalar(metric_name: str) -> tuple[float | None, float | None]:
            values = [
                float(summary[metric_name])
                for summary in pass_summaries
                if summary.get(metric_name) is not None
            ]
            if not values:
                return None, None
            if len(values) == 1:
                return float(np.mean(values)), 0.0
            return float(np.mean(values)), float(np.std(values, ddof=1))

        valid_mean, valid_std = aggregate_scalar("valid")
        match_rate_mean, match_rate_std = aggregate_scalar("match_rate")
        rmse_mean, rmse_std = aggregate_scalar("rmse")

        return {
            "valid": valid_mean,
            "valid_std": valid_std,
            "match_rate": match_rate_mean,
            "match_rate_std": match_rate_std,
            "rmse": rmse_mean,
            "rmse_std": rmse_std,
            "num_samples": pass_summaries[0].get("num_samples") if pass_summaries else 0,
            "ablation_num_seeds": num_seeds,
            "seed_summaries": seed_summaries,
        }

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Mapping[str, float | int | None],
        filename: str,
        keep_paths: list[Path] | None = None,
        *,
        upload_to_wandb: bool = False,
    ) -> Path:
        path = save_named_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            ema=self.ema,
            time_sampler=self.time_sampler,
            config=self.config,
            experiment_name=self.experiment_name,
            epoch=epoch,
            metrics=metrics,
            filename=filename,
            keep_paths=keep_paths,
        )

        if upload_to_wandb and bool(self.logging_cfg["wandb_checkpoints"]):
            save_wandb_checkpoint(path)

        return path

    def save_validation_checkpoint_to_wandb(
        self,
        epoch: int,
        metrics: Mapping[str, float | int | None],
    ) -> None:
        from kldmPlus.utils.model_loader import save_checkpoint

        if not bool(self.logging_cfg["wandb_checkpoints"]):
            return

        with tempfile.TemporaryDirectory(prefix="kldm_val_ckpt_") as temp_dir_name:
            path = Path(temp_dir_name) / f"epoch_{epoch}.pt"
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                ema=self.ema,
                time_sampler=self.time_sampler,
                output_path=path,
                config=self.config,
                epoch=epoch,
                metrics=metrics,
            )
            artifact = wandb.Artifact(f"{self.experiment_name}_validation", type="model")
            artifact.add_file(str(path), name=path.name)
            logged_artifact = self.run.log_artifact(
                artifact,
                aliases=["latest-validation"],
            )
            logged_artifact.wait()

            previous_artifact = self._last_validation_artifact
            self._last_validation_artifact = logged_artifact

            if previous_artifact is not None:
                try:
                    previous_artifact.delete(delete_aliases=True)
                    print(
                        f"checkpoint_deleted=wandb previous_validation epoch={epoch}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"checkpoint_delete_warning=wandb previous_validation error={exc}",
                        flush=True,
                    )

    def _build_validation_ablation_band_figure(
        self,
        *,
        metric: str,
        title: str,
        y_label: str,
    ):
        history = self.validation_ablation_history
        x = np.asarray(history["epoch"], dtype=float)
        y = np.asarray(history[f"{metric}_mean"], dtype=float)
        s = np.asarray(history[f"{metric}_std"], dtype=float)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(x, y - s, y + s, alpha=0.25, label="±1 std")
        ax.plot(x, y, marker="o", linewidth=2, label=f"mean {y_label}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(y_label)
        ax.legend()
        fig.tight_layout()
        return fig

    def _log_validation_ablation_band_plots(
        self,
        *,
        epoch: int,
        val_sample_metrics: Mapping[str, Any],
    ) -> None:
        if self.run is None:
            return
        if not hasattr(wandb, "Image"):
            if not self._warned_missing_wandb_image:
                print(
                    "validation_ablation_plot_warning=wandb.Image unavailable; skipping mean/std band plots",
                    flush=True,
                )
                self._warned_missing_wandb_image = True
            return
        rmse_std = val_sample_metrics.get("rmse_std")
        match_rate_std = val_sample_metrics.get("match_rate_std")
        valid_std = val_sample_metrics.get("valid_std")
        if rmse_std is None or match_rate_std is None or valid_std is None:
            return

        history = self.validation_ablation_history
        history["epoch"].append(float(epoch))
        history["rmse_mean"].append(float(val_sample_metrics["rmse"]))
        history["rmse_std"].append(float(rmse_std))
        history["match_rate_mean"].append(float(val_sample_metrics["match_rate"]))
        history["match_rate_std"].append(float(match_rate_std))
        history["valid_mean"].append(float(val_sample_metrics["valid"]))
        history["valid_std"].append(float(valid_std))

        rmse_fig = self._build_validation_ablation_band_figure(
            metric="rmse",
            title="Validation RMSE across sampling seeds",
            y_label="RMSE",
        )
        match_rate_fig = self._build_validation_ablation_band_figure(
            metric="match_rate",
            title="Validation match rate across sampling seeds",
            y_label="Match rate",
        )
        valid_fig = self._build_validation_ablation_band_figure(
            metric="valid",
            title="Validation validity across sampling seeds",
            y_label="Validity",
        )

        self.run.log(
            {
                "epoch": epoch,
                "val_sampling/rmse_mean_std_plot": wandb.Image(rmse_fig),
                "val_sampling/match_rate_mean_std_plot": wandb.Image(match_rate_fig),
                "val_sampling/valid_mean_std_plot": wandb.Image(valid_fig),
            },
            step=epoch,
        )

        plt.close(rmse_fig)
        plt.close(match_rate_fig)
        plt.close(valid_fig)

    def validate_epoch(self, epoch: int) -> None:
        # Match facitKLDM semantics:
        #   - validation loss stays on the current/online model
        #   - validation sampling metrics (valid, match_rate, rmse) may use EMA
        # This keeps loss curves comparable to training while reporting
        # generation metrics from the smoother EMA model.
        ema_val = bool(self.validation_cfg["ema_val"])
        use_ema = ema_val and self.ema is not None and self.ema.num_updates > 0
        model_label = "EMA model for sampling metrics" if use_ema else "current model"

        print(f"epoch={epoch:04d} entering validation with {model_label}", flush=True)

        val_loss_metrics = self.evaluate_loss()
        sample_context = (
            self.ema.average_parameters(self.model)
            if use_ema and self.ema is not None
            else nullcontext()
        )
        with sample_context:
            val_sample_metrics = self.run_sampling_evaluation()

        merged_metrics = {
            "loss_v": val_loss_metrics["loss_v"],
            "loss_l": val_loss_metrics["loss_l"],
            "loss_weighted": val_loss_metrics["loss_weighted"],
            "valid": val_sample_metrics["valid"],
            "match_rate": val_sample_metrics["match_rate"],
            "rmse": val_sample_metrics["rmse"],
        }
        if "valid_std" in val_sample_metrics:
            merged_metrics["valid_std"] = val_sample_metrics["valid_std"]
        if "match_rate_std" in val_sample_metrics:
            merged_metrics["match_rate_std"] = val_sample_metrics["match_rate_std"]
        if "rmse_std" in val_sample_metrics:
            merged_metrics["rmse_std"] = val_sample_metrics["rmse_std"]
        log_data = {
            "epoch": epoch,
            "val/loss_v": merged_metrics["loss_v"],
            "val/loss_l": merged_metrics["loss_l"],
            "val/loss_weighted": merged_metrics["loss_weighted"],
            "val/valid": merged_metrics["valid"],
            "val/match_rate": merged_metrics["match_rate"],
            "val/rmse": merged_metrics["rmse"],
        }
        self.run.log(log_data, step=epoch)
        self._log_validation_ablation_band_plots(
            epoch=epoch,
            val_sample_metrics=val_sample_metrics,
        )

        self.save_validation_checkpoint_to_wandb(epoch, merged_metrics)

        print(
            f"validation_epoch={epoch:04d} val_loss_weighted={merged_metrics['loss_weighted']:.6f} "
            f"(loss_v={merged_metrics['loss_v']:.6f}, loss_l={merged_metrics['loss_l']:.6f}) "
            f"valid={format_metric(merged_metrics['valid'], '.4f')} "
            f"match_rate={format_metric(merged_metrics['match_rate'], '.4f')} "
            f"rmse={format_metric(merged_metrics['rmse'], '.6f')}",
            flush=True,
        )
        if "valid_std" in merged_metrics or "match_rate_std" in merged_metrics or "rmse_std" in merged_metrics:
            print(
                f"validation_epoch_std={epoch:04d} "
                f"valid_std={format_metric(merged_metrics.get('valid_std'), '.4f')} "
                f"match_rate_std={format_metric(merged_metrics.get('match_rate_std'), '.4f')} "
                f"rmse_std={format_metric(merged_metrics.get('rmse_std'), '.6f')}",
                flush=True,
            )
        seed_summaries = val_sample_metrics.get("seed_summaries", [])
        if seed_summaries:
            backup_payload = {
                "epoch": epoch,
                "num_seeds": val_sample_metrics.get("ablation_num_seeds"),
                "seeds": [int(summary["seed"]) for summary in seed_summaries],
                "valid_values": [summary.get("valid") for summary in seed_summaries],
                "valid_mean": merged_metrics.get("valid"),
                "valid_std": merged_metrics.get("valid_std"),
                "match_rate_values": [summary.get("match_rate") for summary in seed_summaries],
                "match_rate_mean": merged_metrics.get("match_rate"),
                "match_rate_std": merged_metrics.get("match_rate_std"),
                "rmse_values": [summary.get("rmse") for summary in seed_summaries],
                "rmse_mean": merged_metrics.get("rmse"),
                "rmse_std": merged_metrics.get("rmse_std"),
            }
            print(
                f"validation_ablation_backup={json.dumps(backup_payload, sort_keys=True)}",
                flush=True,
            )
        if bool(self.logging_cfg["wandb_checkpoints"]):
            print(f"checkpoint_uploaded=wandb epoch={epoch}", flush=True)

    def run_training_loop(self) -> None:
        # Start one wandb run for the whole experiment.
        wandb_resume_id = self.checkpoint_cfg.get("wandb_resume_id")
        init_kwargs = {
            "project": self.wandb_project,
            "config": self.config | {"start_epoch": self.start_epoch},
            "job_type": "train",
            "reinit": "create_new",
        }
        if wandb_resume_id:
            init_kwargs["id"] = str(wandb_resume_id)
            init_kwargs["resume"] = "must"
        else:
            init_kwargs["name"] = self.wandb_run_name

        self.run = wandb.init(
            **init_kwargs,
        )

        print(f"run_experiment config={self.config_path}", flush=True)
        print(f"device={self.device.type} experiment={self.experiment_name}", flush=True)
        print(f"data_splits train={TRAIN_SPLIT} val={VAL_SPLIT}", flush=True)
        print(f"time_sampler={self.config.get('time_sampler', {'type': 'uniform'})}", flush=True)
        print(f"sampler={self.sampler_cfg}", flush=True)

        epoch = self.start_epoch + 1
        try:
            while not should_stop(self.run):
                train_metrics = self.train_epoch(epoch)

                if epoch % self.train_every_epochs == 0:
                    log_data = {
                        "epoch": epoch,
                        "train/loss_v": train_metrics["loss_v"],
                        "train/loss_l": train_metrics["loss_l"],
                        "train/loss_weighted": train_metrics["loss_weighted"],
                    }
                    for key, value in train_metrics.items():
                        if key.startswith("time_sampler/"):
                            log_data[key] = value
                    self.run.log(log_data, step=epoch)

                    print(
                        f"epoch={epoch:04d} train_loss_weighted={train_metrics['loss_weighted']:.6f} "
                        f"(loss_v={train_metrics['loss_v']:.6f}, loss_l={train_metrics['loss_l']:.6f})",
                        flush=True,
                    )

                if not should_stop(self.run):
                    if epoch < 4000:
                        validate_now = epoch % 250 == 0
                        ablation_seeds = 1
                    else:
                        validate_now = epoch % 100 == 0
                        ablation_seeds = int(self.validation_cfg.get("ablation_num_seeds", 4))

                    if validate_now:
                        original_validation_cfg = dict(self.validation_cfg)
                        try:
                            self.validation_cfg["ablation"] = True
                            self.validation_cfg["ablation_num_seeds"] = ablation_seeds
                            self.validate_epoch(epoch)
                        finally:
                            self.validation_cfg = original_validation_cfg

                epoch += 1
        except KeyboardInterrupt:
            print("run_experiment interrupted", flush=True)
        finally:
            final_epoch = max(epoch - 1, self.start_epoch)
            final_filename = f"{self.experiment_name}_epoch_{final_epoch}.pt"

            # Save exactly one local checkpoint for the experiment: the final model.
            self.save_checkpoint(
                final_epoch,
                {"final_epoch": float(final_epoch)},
                final_filename,
                upload_to_wandb=False,
            )
            if self.run is not None:
                self.run.finish()


def main() -> None:
    ExperimentRunner(parse_args().config).run_training_loop()


if __name__ == "__main__":
    main()
