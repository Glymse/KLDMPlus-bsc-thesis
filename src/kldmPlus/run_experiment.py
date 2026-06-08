from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from datetime import datetime
import random
import re
import shutil
import signal
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from kldmPlus.utils.device import get_default_device
from kldmPlus.utils.time import sample_times
from kldmPlus.utils.time_sampler import TimeSampler

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


def _space_group_to_family(space_group_number: int) -> str:
    sg = int(space_group_number)
    if sg <= 2:
        return "triclinic"
    if sg <= 15:
        return "monoclinic"
    if sg <= 74:
        return "orthorhombic"
    if sg <= 142:
        return "tetragonal"
    if sg <= 167:
        return "trigonal"
    if sg <= 194:
        return "hexagonal"
    return "cubic"


def make_balanced_subset(dataset, subset_size: int | None, seed: int, *, group_key) -> Any:
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    grouped_indices: dict[Any, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        grouped_indices[group_key(dataset[idx])].append(idx)

    group_items = sorted(grouped_indices.items(), key=lambda item: str(item[0]))
    shuffled_groups: list[tuple[Any, list[int]]] = []
    for group, indices in group_items:
        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled_groups.append((group, [indices[pos] for pos in order]))

    selected: list[int] = []
    while len(selected) < int(subset_size):
        made_progress = False
        for _group, indices in shuffled_groups:
            if not indices:
                continue
            selected.append(indices.pop(0))
            made_progress = True
            if len(selected) >= int(subset_size):
                break
        if not made_progress:
            break

    return Subset(dataset, selected)


def make_balanced_subset_by_index(dataset, subset_size: int | None, seed: int, *, group_key_for_index) -> Any:
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    grouped_indices: dict[Any, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        grouped_indices[group_key_for_index(idx)].append(idx)

    group_items = sorted(grouped_indices.items(), key=lambda item: str(item[0]))
    shuffled_groups: list[tuple[Any, list[int]]] = []
    for group, indices in group_items:
        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled_groups.append((group, [indices[pos] for pos in order]))

    selected: list[int] = []
    while len(selected) < int(subset_size):
        made_progress = False
        for _group, indices in shuffled_groups:
            if not indices:
                continue
            selected.append(indices.pop(0))
            made_progress = True
            if len(selected) >= int(subset_size):
                break
        if not made_progress:
            break

    return Subset(dataset, selected)


def dataset_space_group_for_index(dataset, idx: int) -> int:
    if hasattr(dataset, "data") and hasattr(dataset.data, "structure_id") and hasattr(dataset, "_space_group_for_structure_id"):
        structure_id = str(dataset.data.structure_id[idx])
        return int(dataset._space_group_for_structure_id(structure_id))
    sample = dataset[idx]
    return int(torch.as_tensor(sample.space_group).reshape(-1)[0].item())


def make_fraction_subset(dataset, fraction: float | None, seed: int) -> Any:
    if fraction is None or float(fraction) <= 0.0 or float(fraction) >= 1.0:
        return dataset
    subset_size = max(1, int(round(len(dataset) * float(fraction))))
    return make_fixed_subset(dataset, subset_size=subset_size, seed=seed)


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
    if len(parts) < 6:
        return None
    if parts[2] != "artifacts":
        return None

    entity = parts[0]
    project = parts[1]
    artifact_type = parts[3]
    artifact_name = parts[4]
    artifact_version = parts[5]
    file_path = ""
    if len(parts) > 6:
        if parts[6] != "files":
            return None
        file_path = "/".join(parts[7:])
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
        if file_path and expected_path.exists():
            return expected_path.resolve()

        artifact = wandb.Api().artifact(artifact_spec, type=artifact_type)
        artifact_dir = Path(artifact.download(root=str(download_root)))
        checkpoint_path = artifact_dir / file_path if file_path else None
        if checkpoint_path is not None and checkpoint_path.exists():
            print(
                f"checkpoint_downloaded=wandb artifact={artifact_spec} file={checkpoint_path}",
                flush=True,
            )
            return checkpoint_path.resolve()

        matches = sorted(artifact_dir.rglob(Path(file_path).name if file_path else "*.pt"))
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
        latest_hint = f" latest_available={options[-1].resolve()}" if options else ""
        raise FileNotFoundError(f"Checkpoint path does not exist: {candidate}.{latest_hint}")
    raise FileNotFoundError(f"Checkpoint path parent does not exist: {candidate.parent}")


def prune_wandb_artifact_cache(checkpoint_path: Path) -> None:
    try:
        relative = checkpoint_path.resolve().relative_to(WANDB_ARTIFACTS_ROOT)
    except ValueError:
        return
    parts = relative.parts
    if len(parts) < 6:
        return

    artifact_dir = WANDB_ARTIFACTS_ROOT.joinpath(*parts[:4])
    keep_version_dir = artifact_dir / parts[4]
    if not artifact_dir.exists():
        return

    for candidate in artifact_dir.iterdir():
        if candidate == keep_version_dir:
            continue
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)


def prune_validation_artifact_cache(experiment_name: str) -> None:
    if not WANDB_ARTIFACTS_ROOT.exists():
        return
    artifact_name = f"{experiment_name}_validation"
    for candidate in WANDB_ARTIFACTS_ROOT.glob(f"*/*/model/{artifact_name}"):
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)


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


@contextmanager
def preserve_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@contextmanager
def isolated_rng_seed(seed: int):
    with preserve_rng_state():
        set_validation_sampling_seed(seed)
        yield


def epoch_from_checkpoint_name(path: Path) -> int | None:
    match = re.search(r"(?:^|[_-])epoch[_-](\d+)(?:\D|$)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def checkpoint_dir(config: dict[str, Any], experiment_name: str) -> Path:
    del config
    return CHECKPOINTS_ROOT / experiment_name


def clear_local_checkpoint_dir(config: dict[str, Any], experiment_name: str) -> None:
    candidate = checkpoint_dir(config=config, experiment_name=experiment_name)
    if candidate.exists():
        shutil.rmtree(candidate, ignore_errors=True)


def clear_wandb_artifact_cache() -> None:
    if WANDB_ARTIFACTS_ROOT.exists():
        shutil.rmtree(WANDB_ARTIFACTS_ROOT, ignore_errors=True)


def write_checkpoint_file(
    *,
    model,
    optimizer: torch.optim.Optimizer,
    ema,
    time_sampler,
    output_path: Path,
    config: dict[str, Any],
    epoch: int,
    metrics: Mapping[str, float | int | None],
) -> Path:
    from kldmPlus.utils.model_loader import save_checkpoint

    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    return output_path


class ExperimentRunner:
    def __init__(self, config_path: str | Path) -> None:
        from kldmPlus.utils.model_loader import build_training_components, load_checkpoint

        # -------------------------------------------------
        # Static experiment setup from config
        # -------------------------------------------------
        self.config_path, self.config = load_experiment_config(config_path)
        self.experiment_name = str(self.config["experiment_name"])

        self.sampler_cfg = dict(self.config["sampler"])
        self.training_cfg = dict(self.config.get("training", {}) or {})
        self.logging_cfg = dict(self.config["logging"])
        self.validation_cfg = dict(self.config["validation"])
        self.checkpoint_cfg = dict(self.config["checkpoint"])
        self.wandb_project = str(self.logging_cfg.get("wandb_project", self.experiment_name))
        self.wandb_run_name = str(self.logging_cfg.get("wandb_run_name", self.experiment_name))
        set_global_training_seed(TRAIN_SEED)
        clear_local_checkpoint_dir(self.config, self.experiment_name)
        clear_wandb_artifact_cache()

        self.train_every_epochs = int(self.logging_cfg["train_every_epochs"])
        self.profile_train_batches = int(self.logging_cfg.get("profile_train_batches", 0))
        self.validate_every_epochs = int(self.validation_cfg["every_n_epochs"])
        self.max_epochs = self.training_cfg.get("max_epochs")
        self.max_epochs = None if self.max_epochs is None else int(self.max_epochs)

        # -------------------------------------------------
        # Runtime objects: device, data, model, optimizer, EMA
        # -------------------------------------------------
        self.device = get_default_device()
        print("startup phase=create_loaders:start", flush=True)
        self.train_loader, self.val_loader, self.lattice_transform = self.create_loaders()
        print("startup phase=create_loaders:done", flush=True)
        print("startup phase=build_time_sampler:start", flush=True)
        self.time_sampler = self.build_time_sampler()
        print("startup phase=build_time_sampler:done", flush=True)
        print("startup phase=build_training_components:start", flush=True)
        self.model, self.optimizer, self.ema = build_training_components(
            config=self.config,
            device=self.device,
        )
        print("startup phase=build_training_components:done", flush=True)

        self.start_epoch = 0
        self.run = None
        self._last_validation_artifact = None
        self._last_validation_artifact_epoch: int | None = None
        self._last_validation_checkpoint_name: str | None = None

        # Optional resume path for continuing training from a saved checkpoint.
        resume_from = self.checkpoint_cfg["resume_from"]
        resolved_checkpoint_path: Path | None = None
        if resume_from:
            print("startup phase=resolve_checkpoint:start", flush=True)
            resolved_checkpoint_path = resolve_checkpoint_reference(
                resume_from,
                config_path=self.config_path,
            )
            print(
                f"startup phase=resolve_checkpoint:done path={resolved_checkpoint_path}",
                flush=True,
            )
            load_optimizer_state = bool(self.checkpoint_cfg.get("load_optimizer_state", True))
            load_ema_state = bool(self.checkpoint_cfg.get("load_ema_state", True))
            load_time_sampler_state = bool(self.checkpoint_cfg.get("load_time_sampler_state", True))
            print("startup phase=load_checkpoint:start", flush=True)
            checkpoint = load_checkpoint(
                checkpoint_path=resolved_checkpoint_path,
                model=self.model,
                optimizer=self.optimizer if load_optimizer_state else None,
                ema=self.ema if load_ema_state else None,
                device=self.device,
                prefer_ema_weights=False,
            )
            print("startup phase=load_checkpoint:done", flush=True)
            self.start_epoch = int(checkpoint["epoch"])
            filename_epoch = epoch_from_checkpoint_name(resolved_checkpoint_path)
            if filename_epoch is not None and filename_epoch != self.start_epoch:
                raise ValueError(
                    "Checkpoint filename epoch does not match checkpoint payload epoch: "
                    f"path={resolved_checkpoint_path} filename_epoch={filename_epoch} "
                    f"payload_epoch={self.start_epoch}. Refusing to resume from an "
                    "ambiguous checkpoint."
                )
            print(
                f"checkpoint_loaded path={resolved_checkpoint_path} epoch={self.start_epoch}",
                flush=True,
            )
            if load_time_sampler_state and checkpoint.get("time_sampler_state_dict") is not None:
                self.time_sampler.load_state_dict(checkpoint.get("time_sampler_state_dict"))
        if bool(self.checkpoint_cfg.get("prune_wandb_artifact_cache", False)) and resolved_checkpoint_path is not None:
            prune_wandb_artifact_cache(resolved_checkpoint_path)
        clear_wandb_artifact_cache()

    def build_time_sampler(self):
        cfg = dict(self.config.get("time_sampler", {}))
        sampler_type = str(cfg.get("type", "uniform"))
        if sampler_type not in {"uniform", "log_uniform"}:
            raise ValueError(
                "KLDMPlus now only supports time_sampler.type in {'uniform', 'log_uniform'}. "
                f"Got {sampler_type!r}."
            )
        return TimeSampler(
            mode=sampler_type,
            lower_bound=TIME_LOWER_BOUND,
            seed=TRAIN_SEED,
        )

    def create_loaders(self) -> tuple[DataLoader, DataLoader, Any]:
        from kldmPlus.data import CSPTask, resolve_data_root
        from kldmPlus.data.csp import validate_lattice_configuration

        dataset_cfg = dict(self.config["dataset"])
        model_cfg = dict(self.config["model"])
        dataset_lattice_representation = str(dataset_cfg.get("lattice_representation", "kldm"))
        model_lattice_representation = str(model_cfg.get("lattice_representation", dataset_lattice_representation))
        if model_lattice_representation != dataset_lattice_representation:
            raise ValueError(
                "dataset.lattice_representation and model.lattice_representation must match: "
                f"dataset={dataset_lattice_representation!r}, model={model_lattice_representation!r}."
            )
        # Guard against silent representation/diffusion mismatches before the
        # runner creates datasets or starts training.
        validate_lattice_configuration(
            lattice_representation=dataset_lattice_representation,
            lattice_parameterization=str(model_cfg["lattice_parameterization"]),
            lattice_diffusion_type=str(model_cfg.get("lattice_diffusion_type", "VP")),
        )

        task = CSPTask(
            dataset_name=str(dataset_cfg["name"]),
            lattice_parameterization=str(model_cfg["lattice_parameterization"]),
            lattice_representation=dataset_lattice_representation,
        )

        root = resolve_data_root(dataset_cfg["root"])
        batch_size = int(dataset_cfg["batch_size"])
        num_workers = int(dataset_cfg["num_workers"])
        pin_memory = bool(dataset_cfg["pin_memory"])
        train_generator = torch.Generator().manual_seed(TRAIN_SEED)
        worker_init_fn = _seed_worker_factory(TRAIN_SEED)

        # Keep training/validation splits fixed so experiment metrics are always
        # comparable across runs.
        train_dataset_full = task.fit_dataset(root=root, split=TRAIN_SPLIT, download=True)
        train_subset_fraction = dataset_cfg.get("train_subset_fraction")
        train_subset_seed = int(dataset_cfg.get("train_subset_seed", TRAIN_SEED))
        train_subset_strategy = str(dataset_cfg.get("train_subset_strategy", "random"))
        if train_subset_fraction is None or float(train_subset_fraction) <= 0.0 or float(train_subset_fraction) >= 1.0:
            train_dataset = train_dataset_full
        else:
            train_subset_size = max(1, int(round(len(train_dataset_full) * float(train_subset_fraction))))
            if train_subset_strategy == "balanced_space_group":
                train_dataset = make_balanced_subset_by_index(
                    train_dataset_full,
                    subset_size=train_subset_size,
                    seed=train_subset_seed,
                    group_key_for_index=lambda idx: dataset_space_group_for_index(train_dataset_full, int(idx)),
                )
            else:
                train_dataset = make_fixed_subset(
                    train_dataset_full,
                    subset_size=train_subset_size,
                    seed=train_subset_seed,
                )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            generator=train_generator,
            worker_init_fn=worker_init_fn,
            collate_fn=train_dataset_full.collate_fn,
        )

        # Validation always uses the validation split, optionally with a fixed
        # subset for faster checks.
        val_dataset_full = task.fit_dataset(root=root, split=VAL_SPLIT, download=True)
        val_subset_strategy = str(self.validation_cfg.get("subset_strategy", "random"))
        if val_subset_strategy == "balanced_family":
            val_dataset = make_balanced_subset_by_index(
                val_dataset_full,
                subset_size=self.validation_cfg["subset_size"],
                seed=int(self.validation_cfg["subset_seed"]),
                group_key_for_index=lambda idx: _space_group_to_family(
                    dataset_space_group_for_index(val_dataset_full, int(idx)),
                ),
            )
        else:
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
            persistent_workers=(num_workers > 0),
            worker_init_fn=worker_init_fn,
            collate_fn=val_dataset_full.collate_fn,
        )

        return train_loader, val_loader, task.make_lattice_transform(
            root=root,
            download=True,
        )

    def lattice_debug_enabled(self) -> bool:
        # Toggle expensive legacy primitive/orbit lattice diagnostics.
        return bool(getattr(self.model, "lattice_debug", False))

    def add_lattice_log_data(self, log_data: dict[str, Any], metrics: Mapping[str, Any], *, prefix: str) -> None:
        log_data.update(
            {
                f"{prefix}/loss_conv_sg": metrics["loss_conv_sg"],
                f"{prefix}/conv_weight": metrics["conv_weight_mean"],
                f"{prefix}/conv_proj_pred_k": metrics["conv_projection_error_pred_k"],
                f"{prefix}/conv_proj_gt_k": metrics["conv_projection_error_gt_k"],
            }
        )
        if not self.lattice_debug_enabled():
            return
        log_data.update(
            {
                f"{prefix}/loss_sg_lattice": metrics["loss_sg_lattice"],
                f"{prefix}/loss_sg_lattice_weighted": metrics["loss_sg_lattice_weighted"],
                f"{prefix}/loss_sg_lattice_lambda_scaled": metrics["loss_sg_lattice_lambda_scaled"],
                f"{prefix}/loss_conv_sg_weighted": metrics["loss_conv_sg_weighted"],
                f"{prefix}/loss_conv_sg_lambda_scaled": metrics["loss_conv_sg_lambda_scaled"],
                f"{prefix}/lambda_sg_lattice": metrics["lambda_sg_lattice"],
                f"{prefix}/lambda_conv_sg": metrics["lambda_conv_sg"],
                f"{prefix}/lattice_sg_time_weight_mean": metrics["lattice_sg_time_weight_mean"],
                f"{prefix}/conv_sg_time_weight_mean": metrics["conv_sg_time_weight_mean"],
                f"{prefix}/conv_weight_mean": metrics["conv_weight_mean"],
                f"{prefix}/projection_error_pred_k": metrics["projection_error_pred_k"],
                f"{prefix}/projection_error_gt_k": metrics["projection_error_gt_k"],
                f"{prefix}/primitive_projection_error_pred_k": metrics["primitive_projection_error_pred_k"],
                f"{prefix}/primitive_projection_error_gt_k": metrics["primitive_projection_error_gt_k"],
                f"{prefix}/conv_projection_error_pred_k": metrics["conv_projection_error_pred_k"],
                f"{prefix}/conv_projection_error_gt_k": metrics["conv_projection_error_gt_k"],
                f"{prefix}/projection_error_pred_direct_k": metrics["projection_error_pred_direct_k"],
                f"{prefix}/projection_error_gt_direct_k": metrics["projection_error_gt_direct_k"],
                f"{prefix}/projection_error_pred_orbit_k": metrics["projection_error_pred_orbit_k"],
                f"{prefix}/projection_error_gt_orbit_k": metrics["projection_error_gt_orbit_k"],
            }
        )

    def lattice_status_text(self, metrics: Mapping[str, Any]) -> str:
        essential = (
            f"loss_conv_sg={metrics['loss_conv_sg']:.6f}, "
            f"conv_weight={metrics['conv_weight_mean']:.6f}, "
            f"conv_proj_pred_k={metrics['conv_projection_error_pred_k']:.6f}, "
            f"conv_proj_gt_k={metrics['conv_projection_error_gt_k']:.6f}"
        )
        if not self.lattice_debug_enabled():
            return essential
        return (
            f"loss_sg_lattice={metrics['loss_sg_lattice']:.6f}, "
            f"loss_sg_lambda_scaled={metrics['loss_sg_lattice_lambda_scaled']:.6f}, "
            f"{essential}, "
            f"loss_conv_lambda_scaled={metrics['loss_conv_sg_lambda_scaled']:.6f}, "
            f"lambda_sg={metrics['lambda_sg_lattice']:.6f}, "
            f"lambda_conv_sg={metrics['lambda_conv_sg']:.6f}, "
            f"sg_time_weight={metrics['lattice_sg_time_weight_mean']:.6f}, "
            f"conv_time_weight={metrics['conv_sg_time_weight_mean']:.6f}, "
            f"proj_pred_k={metrics['projection_error_pred_k']:.6f}, "
            f"proj_gt_k={metrics['projection_error_gt_k']:.6f}, "
            f"proj_pred_orbit_k={metrics['projection_error_pred_orbit_k']:.6f}, "
            f"proj_gt_orbit_k={metrics['projection_error_gt_orbit_k']:.6f}"
        )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        train_generator = getattr(self.train_loader, "generator", None)
        if train_generator is not None:
            train_generator.manual_seed(TRAIN_SEED + int(epoch))

        total_loss_v = total_loss_l = total_loss_sg_lattice = 0.0
        total_loss_conv_sg = 0.0
        total_projection_error_pred_k = total_projection_error_gt_k = 0.0
        total_conv_projection_error_pred_k = total_conv_projection_error_gt_k = 0.0
        total_projection_error_pred_orbit_k = total_projection_error_gt_orbit_k = 0.0
        total_lattice_sg_time_weight_mean = 0.0
        total_conv_sg_time_weight_mean = total_conv_weight_mean = 0.0
        total_loss_v_weighted = total_loss_l_weighted = total_loss_sg_lattice_weighted = 0.0
        total_loss_sg_lattice_lambda_scaled = 0.0
        total_loss_conv_sg_weighted = total_loss_conv_sg_lambda_scaled = 0.0
        total_nodes = total_graphs = 0
        lambda_sg_lattice = 0.0
        lambda_conv_sg = 0.0
        total_data_wait_seconds = 0.0
        total_to_device_seconds = 0.0
        total_step_seconds = 0.0
        total_ema_seconds = 0.0
        epoch_start = time.perf_counter()
        batch_fetch_start = time.perf_counter()

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            data_wait_seconds = time.perf_counter() - batch_fetch_start
            total_data_wait_seconds += data_wait_seconds
            if STOP_REQUESTED:
                break

            to_device_start = time.perf_counter()
            batch = batch.to(self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            to_device_seconds = time.perf_counter() - to_device_start
            total_to_device_seconds += to_device_seconds

            step_start = time.perf_counter()
            sampled_t, sampled_weights = self.time_sampler.sample(batch)

            self.optimizer.zero_grad(set_to_none=True)
            loss, metrics = self.model.algorithm2_loss(
                batch=batch,
                t=sampled_t,
                time_weight=sampled_weights,
                debug=False,
            )
            loss.backward()
            self.optimizer.step()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            step_seconds = time.perf_counter() - step_start
            total_step_seconds += step_seconds

            if self.ema is not None:
                ema_start = time.perf_counter()
                self.ema.update(self.model, current_epoch=epoch)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                total_ema_seconds += time.perf_counter() - ema_start

            if self.profile_train_batches > 0 and batch_idx <= self.profile_train_batches:
                print(
                    f"train_batch_profile epoch={epoch:04d} batch={batch_idx} "
                    f"data_wait_s={data_wait_seconds:.3f} to_device_s={to_device_seconds:.3f} "
                    f"step_s={step_seconds:.3f} ema_s={total_ema_seconds:.3f} "
                    f"graphs={int(batch.num_graphs)} nodes={int(batch.pos.shape[0])}",
                    flush=True,
                )

            total_loss_v += float(metrics["loss_v"]) * int(batch.pos.shape[0])
            total_loss_l += float(metrics["loss_l"]) * int(batch.num_graphs)
            total_loss_sg_lattice += float(metrics.get("loss_sg_lattice", 0.0)) * int(batch.num_graphs)
            total_loss_conv_sg += float(metrics.get("loss_conv_sg", 0.0)) * int(batch.num_graphs)
            total_projection_error_pred_k += float(metrics.get("projection_error_pred_k", 0.0)) * int(batch.num_graphs)
            total_projection_error_gt_k += float(metrics.get("projection_error_gt_k", 0.0)) * int(batch.num_graphs)
            total_conv_projection_error_pred_k += float(metrics.get("conv_projection_error_pred_k", 0.0)) * int(batch.num_graphs)
            total_conv_projection_error_gt_k += float(metrics.get("conv_projection_error_gt_k", 0.0)) * int(batch.num_graphs)
            total_projection_error_pred_orbit_k += float(metrics.get("projection_error_pred_orbit_k", 0.0)) * int(batch.num_graphs)
            total_projection_error_gt_orbit_k += float(metrics.get("projection_error_gt_orbit_k", 0.0)) * int(batch.num_graphs)
            total_lattice_sg_time_weight_mean += float(metrics.get("lattice_sg_time_weight_mean", 0.0)) * int(batch.num_graphs)
            total_conv_sg_time_weight_mean += float(metrics.get("conv_sg_time_weight_mean", 0.0)) * int(batch.num_graphs)
            total_conv_weight_mean += float(metrics.get("conv_weight_mean", 0.0)) * int(batch.num_graphs)
            total_loss_v_weighted += float(metrics["loss_v_weighted"]) * int(batch.pos.shape[0])
            total_loss_l_weighted += float(metrics["loss_l_weighted"]) * int(batch.num_graphs)
            total_loss_sg_lattice_weighted += float(metrics.get("loss_sg_lattice_weighted", 0.0)) * int(batch.num_graphs)
            total_loss_sg_lattice_lambda_scaled += float(metrics.get("loss_sg_lattice_lambda_scaled", 0.0)) * int(batch.num_graphs)
            total_loss_conv_sg_weighted += float(metrics.get("loss_conv_sg_weighted", 0.0)) * int(batch.num_graphs)
            total_loss_conv_sg_lambda_scaled += float(metrics.get("loss_conv_sg_lambda_scaled", 0.0)) * int(batch.num_graphs)
            lambda_sg_lattice = float(metrics.get("lambda_sg_lattice", lambda_sg_lattice))
            lambda_conv_sg = float(metrics.get("lambda_conv_sg", lambda_conv_sg))
            total_nodes += int(batch.pos.shape[0])
            total_graphs += int(batch.num_graphs)
            batch_fetch_start = time.perf_counter()

        if total_nodes == 0 or total_graphs == 0:
            raise RuntimeError("Training stopped before any batches were processed.")

        mean_loss_v = total_loss_v / total_nodes
        mean_loss_l = total_loss_l / total_graphs
        mean_loss_sg_lattice = total_loss_sg_lattice / total_graphs
        mean_loss_conv_sg = total_loss_conv_sg / total_graphs
        mean_loss_v_weighted = total_loss_v_weighted / total_nodes
        mean_loss_l_weighted = total_loss_l_weighted / total_graphs
        mean_loss_sg_lattice_weighted = total_loss_sg_lattice_weighted / total_graphs
        mean_loss_sg_lattice_lambda_scaled = total_loss_sg_lattice_lambda_scaled / total_graphs
        mean_loss_conv_sg_weighted = total_loss_conv_sg_weighted / total_graphs
        mean_loss_conv_sg_lambda_scaled = total_loss_conv_sg_lambda_scaled / total_graphs
        mean_total_loss = (
            mean_loss_v_weighted
            + mean_loss_l_weighted
            + mean_loss_sg_lattice_lambda_scaled
            + mean_loss_conv_sg_lambda_scaled
        )

        metrics = {
            "loss": mean_total_loss,
            "loss_v": mean_loss_v,
            "loss_l": mean_loss_l,
            "loss_sg_lattice": mean_loss_sg_lattice,
            "loss_conv_sg": mean_loss_conv_sg,
            "projection_error_pred_k": total_projection_error_pred_k / total_graphs,
            "projection_error_gt_k": total_projection_error_gt_k / total_graphs,
            "primitive_projection_error_pred_k": total_projection_error_pred_k / total_graphs,
            "primitive_projection_error_gt_k": total_projection_error_gt_k / total_graphs,
            "conv_projection_error_pred_k": total_conv_projection_error_pred_k / total_graphs,
            "conv_projection_error_gt_k": total_conv_projection_error_gt_k / total_graphs,
            "projection_error_pred_direct_k": total_projection_error_pred_k / total_graphs,
            "projection_error_gt_direct_k": total_projection_error_gt_k / total_graphs,
            "projection_error_pred_orbit_k": total_projection_error_pred_orbit_k / total_graphs,
            "projection_error_gt_orbit_k": total_projection_error_gt_orbit_k / total_graphs,
            "lattice_sg_time_weight_mean": total_lattice_sg_time_weight_mean / total_graphs,
            "conv_sg_time_weight_mean": total_conv_sg_time_weight_mean / total_graphs,
            "conv_weight_mean": total_conv_weight_mean / total_graphs,
            "loss_v_weighted": mean_loss_v_weighted,
            "loss_l_weighted": mean_loss_l_weighted,
            "loss_sg_lattice_weighted": mean_loss_sg_lattice_weighted,
            "loss_sg_lattice_lambda_scaled": mean_loss_sg_lattice_lambda_scaled,
            "loss_conv_sg_weighted": mean_loss_conv_sg_weighted,
            "loss_conv_sg_lambda_scaled": mean_loss_conv_sg_lambda_scaled,
            "lambda_sg_lattice": lambda_sg_lattice,
            "lambda_conv_sg": lambda_conv_sg,
            "loss_weighted": mean_total_loss,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "data_wait_seconds": total_data_wait_seconds,
            "to_device_seconds": total_to_device_seconds,
            "step_seconds": total_step_seconds,
            "ema_seconds": total_ema_seconds,
        }
        return metrics

    def evaluate_loss(self) -> dict[str, float]:
        self.model.eval()

        total_loss_v = total_loss_l = total_loss_sg_lattice = 0.0
        total_loss_conv_sg = 0.0
        total_projection_error_pred_k = total_projection_error_gt_k = 0.0
        total_conv_projection_error_pred_k = total_conv_projection_error_gt_k = 0.0
        total_projection_error_pred_orbit_k = total_projection_error_gt_orbit_k = 0.0
        total_lattice_sg_time_weight_mean = 0.0
        total_conv_sg_time_weight_mean = total_conv_weight_mean = 0.0
        total_loss_v_weighted = total_loss_l_weighted = total_loss_sg_lattice_weighted = 0.0
        total_loss_sg_lattice_lambda_scaled = 0.0
        total_loss_conv_sg_weighted = total_loss_conv_sg_lambda_scaled = 0.0
        total_nodes = total_graphs = 0
        lambda_sg_lattice = 0.0
        lambda_conv_sg = 0.0
        loss_seed = int(self.validation_cfg.get("loss_seed", TRAIN_SEED + 17))

        with isolated_rng_seed(loss_seed):
            for batch in self.val_loader:
                batch = batch.to(self.device)

                # Validation uses the same noisy-time sampling pattern as training.
                t_graph = sample_times(batch, lower_bound=TIME_LOWER_BOUND)

                with torch.no_grad():
                    _, metrics = self.model.algorithm2_loss(batch=batch, t=t_graph, debug=True)

                total_loss_v += float(metrics["loss_v"]) * int(batch.pos.shape[0])
                total_loss_l += float(metrics["loss_l"]) * int(batch.num_graphs)
                total_loss_sg_lattice += float(metrics.get("loss_sg_lattice", 0.0)) * int(batch.num_graphs)
                total_loss_conv_sg += float(metrics.get("loss_conv_sg", 0.0)) * int(batch.num_graphs)
                total_projection_error_pred_k += float(metrics.get("projection_error_pred_k", 0.0)) * int(batch.num_graphs)
                total_projection_error_gt_k += float(metrics.get("projection_error_gt_k", 0.0)) * int(batch.num_graphs)
                total_conv_projection_error_pred_k += float(metrics.get("conv_projection_error_pred_k", 0.0)) * int(batch.num_graphs)
                total_conv_projection_error_gt_k += float(metrics.get("conv_projection_error_gt_k", 0.0)) * int(batch.num_graphs)
                total_projection_error_pred_orbit_k += float(metrics.get("projection_error_pred_orbit_k", 0.0)) * int(batch.num_graphs)
                total_projection_error_gt_orbit_k += float(metrics.get("projection_error_gt_orbit_k", 0.0)) * int(batch.num_graphs)
                total_lattice_sg_time_weight_mean += float(metrics.get("lattice_sg_time_weight_mean", 0.0)) * int(batch.num_graphs)
                total_conv_sg_time_weight_mean += float(metrics.get("conv_sg_time_weight_mean", 0.0)) * int(batch.num_graphs)
                total_conv_weight_mean += float(metrics.get("conv_weight_mean", 0.0)) * int(batch.num_graphs)
                total_loss_v_weighted += float(metrics["loss_v_weighted"]) * int(batch.pos.shape[0])
                total_loss_l_weighted += float(metrics["loss_l_weighted"]) * int(batch.num_graphs)
                total_loss_sg_lattice_weighted += float(metrics.get("loss_sg_lattice_weighted", 0.0)) * int(batch.num_graphs)
                total_loss_sg_lattice_lambda_scaled += float(metrics.get("loss_sg_lattice_lambda_scaled", 0.0)) * int(batch.num_graphs)
                total_loss_conv_sg_weighted += float(metrics.get("loss_conv_sg_weighted", 0.0)) * int(batch.num_graphs)
                total_loss_conv_sg_lambda_scaled += float(metrics.get("loss_conv_sg_lambda_scaled", 0.0)) * int(batch.num_graphs)
                lambda_sg_lattice = float(metrics.get("lambda_sg_lattice", lambda_sg_lattice))
                lambda_conv_sg = float(metrics.get("lambda_conv_sg", lambda_conv_sg))
                total_nodes += int(batch.pos.shape[0])
                total_graphs += int(batch.num_graphs)

        if total_nodes == 0 or total_graphs == 0:
            raise RuntimeError("Validation loader is empty.")

        mean_loss_v = total_loss_v / total_nodes
        mean_loss_l = total_loss_l / total_graphs
        mean_loss_sg_lattice = total_loss_sg_lattice / total_graphs
        mean_loss_conv_sg = total_loss_conv_sg / total_graphs
        mean_loss_v_weighted = total_loss_v_weighted / total_nodes
        mean_loss_l_weighted = total_loss_l_weighted / total_graphs
        mean_loss_sg_lattice_weighted = total_loss_sg_lattice_weighted / total_graphs
        mean_loss_sg_lattice_lambda_scaled = total_loss_sg_lattice_lambda_scaled / total_graphs
        mean_loss_conv_sg_weighted = total_loss_conv_sg_weighted / total_graphs
        mean_loss_conv_sg_lambda_scaled = total_loss_conv_sg_lambda_scaled / total_graphs
        mean_total_loss = (
            mean_loss_v_weighted
            + mean_loss_l_weighted
            + mean_loss_sg_lattice_lambda_scaled
            + mean_loss_conv_sg_lambda_scaled
        )

        return {
            "loss": mean_total_loss,
            "loss_v": mean_loss_v,
            "loss_l": mean_loss_l,
            "loss_sg_lattice": mean_loss_sg_lattice,
            "loss_conv_sg": mean_loss_conv_sg,
            "projection_error_pred_k": total_projection_error_pred_k / total_graphs,
            "projection_error_gt_k": total_projection_error_gt_k / total_graphs,
            "primitive_projection_error_pred_k": total_projection_error_pred_k / total_graphs,
            "primitive_projection_error_gt_k": total_projection_error_gt_k / total_graphs,
            "conv_projection_error_pred_k": total_conv_projection_error_pred_k / total_graphs,
            "conv_projection_error_gt_k": total_conv_projection_error_gt_k / total_graphs,
            "projection_error_pred_direct_k": total_projection_error_pred_k / total_graphs,
            "projection_error_gt_direct_k": total_projection_error_gt_k / total_graphs,
            "projection_error_pred_orbit_k": total_projection_error_pred_orbit_k / total_graphs,
            "projection_error_gt_orbit_k": total_projection_error_gt_orbit_k / total_graphs,
            "lattice_sg_time_weight_mean": total_lattice_sg_time_weight_mean / total_graphs,
            "conv_sg_time_weight_mean": total_conv_sg_time_weight_mean / total_graphs,
            "conv_weight_mean": total_conv_weight_mean / total_graphs,
            "loss_v_weighted": mean_loss_v_weighted,
            "loss_l_weighted": mean_loss_l_weighted,
            "loss_sg_lattice_weighted": mean_loss_sg_lattice_weighted,
            "loss_sg_lattice_lambda_scaled": mean_loss_sg_lattice_lambda_scaled,
            "loss_conv_sg_weighted": mean_loss_conv_sg_weighted,
            "loss_conv_sg_lambda_scaled": mean_loss_conv_sg_lambda_scaled,
            "lambda_sg_lattice": lambda_sg_lattice,
            "lambda_conv_sg": lambda_conv_sg,
            "loss_weighted": mean_total_loss,
        }

    def run_sampling_evaluation(self) -> dict[str, Any]:
        from kldmPlus.sample_evaluation.sample_evaluation import (
            aggregate_csp_reconstruction_metrics,
            evaluate_csp_reconstruction,
        )

        self.model.eval()

        def collect_one_pass(*, seed: int | None = None) -> dict[str, Any]:
            if seed is not None:
                print(f"validation_sampling_seed={seed} pass_start", flush=True)

            seed_context = isolated_rng_seed(seed) if seed is not None else preserve_rng_state()
            with seed_context:
                results = []
                num_graphs_seen = 0
                projection_sums = {
                    "projection_error_pred_direct_k": 0.0,
                    "projection_error_gt_direct_k": 0.0,
                    "projection_error_pred_orbit_k": 0.0,
                    "projection_error_gt_orbit_k": 0.0,
                }
                projection_count = 0

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
                        requested_sg = int(torch.as_tensor(batch.space_group).reshape(-1)[graph_idx].item())
                        results.append(
                            evaluate_csp_reconstruction(
                                pred_f=pos_t[start_idx:end_idx],
                                pred_l=l_t[graph_idx],
                                pred_a=h_t[start_idx:end_idx],
                                target_f=batch.pos[start_idx:end_idx],
                                target_l=batch.l[graph_idx],
                                target_a=batch.atomic_numbers[start_idx:end_idx],
                                lattice_transform=self.lattice_transform,
                                requested_space_group=requested_sg,
                            )
                        )
                        if self.lattice_debug_enabled() and self.model.lattice_representation == "diffcsp_k":
                            sg_t = torch.tensor([requested_sg], device=l_t.device, dtype=torch.long)
                            pred_l_graph = l_t[graph_idx].reshape(1, -1)
                            gt_l_graph = batch.l[graph_idx].reshape(1, -1)
                            projection_sums["projection_error_pred_direct_k"] += float(
                                self.model.lattice_symmetry.direct_sg_residual_abs_mean(pred_l_graph, sg_t).item()
                            )
                            projection_sums["projection_error_gt_direct_k"] += float(
                                self.model.lattice_symmetry.direct_sg_residual_abs_mean(gt_l_graph, sg_t).item()
                            )
                            projection_sums["projection_error_pred_orbit_k"] += float(
                                self.model.lattice_symmetry.orbit_sg_residual_abs_mean(
                                    pred_l_graph,
                                    sg_t,
                                    max_candidates=self.model.lattice_orbit_metric_max_candidates,
                                ).item()
                            )
                            projection_sums["projection_error_gt_orbit_k"] += float(
                                self.model.lattice_symmetry.orbit_sg_residual_abs_mean(
                                    gt_l_graph,
                                    sg_t,
                                    max_candidates=self.model.lattice_orbit_metric_max_candidates,
                                ).item()
                            )
                            projection_count += 1
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
            projection_metrics = {
                key: (None if projection_count == 0 else value / projection_count)
                for key, value in projection_sums.items()
            }
            return {
                "valid": summary.get("valid"),
                "match_rate": summary.get("match_rate"),
                "rmse": summary.get("rmse"),
                "frac_rmse": summary.get("frac_rmse"),
                "detected_family_agreement": summary.get("detected_family_agreement"),
                "detected_sg_agreement": summary.get("detected_sg_agreement"),
                "wyckoff_letter_agreement": summary.get("wyckoff_letter_agreement"),
                "predicted_wyckoff_dimensionality_distribution": summary.get("predicted_wyckoff_dimensionality_distribution"),
                "target_wyckoff_dimensionality_distribution": summary.get("target_wyckoff_dimensionality_distribution"),
                "lattice_lengths_rmse": summary.get("lattice_lengths_rmse"),
                "lattice_angles_rmse": summary.get("lattice_angles_rmse"),
                "volume_rel_error": summary.get("volume_rel_error"),
                "projection_error_pred_direct_k": projection_metrics["projection_error_pred_direct_k"],
                "projection_error_gt_direct_k": projection_metrics["projection_error_gt_direct_k"],
                "projection_error_pred_orbit_k": projection_metrics["projection_error_pred_orbit_k"],
                "projection_error_gt_orbit_k": projection_metrics["projection_error_gt_orbit_k"],
                "num_samples": summary.get("num_samples"),
            }

        first_seed = int(self.validation_cfg.get("sampling_seed", 0))
        num_sampling_seeds = int(self.validation_cfg.get("num_sampling_seeds", 1))
        if num_sampling_seeds <= 1:
            return collect_one_pass(seed=first_seed)

        seed_metrics = [collect_one_pass(seed=first_seed + offset) for offset in range(num_sampling_seeds)]
        merged: dict[str, Any] = {"num_sampling_seeds": num_sampling_seeds}
        keys = sorted({key for metrics in seed_metrics for key in metrics})
        for key in keys:
            values = [metrics.get(key) for metrics in seed_metrics]
            numeric_values = [
                float(value)
                for value in values
                if isinstance(value, (int, float)) and not isinstance(value, bool) and not np.isnan(float(value))
            ]
            if numeric_values:
                merged[key] = float(np.mean(np.asarray(numeric_values, dtype=float)))
                merged[f"{key}_std"] = float(np.std(np.asarray(numeric_values, dtype=float)))
            else:
                merged[key] = values[0] if values else None
        return merged

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Mapping[str, float | int | None],
        filename: str,
        *,
        artifact_name: str,
        aliases: list[str],
    ) -> tuple[str, bool]:
        if self.run is None or not bool(self.logging_cfg["wandb_checkpoints"]):
            return artifact_name, False

        with tempfile.TemporaryDirectory(prefix=f"{self.experiment_name}_checkpoint_") as temp_dir:
            local_path = write_checkpoint_file(
                model=self.model,
                optimizer=self.optimizer,
                ema=self.ema,
                time_sampler=self.time_sampler,
                output_path=Path(temp_dir) / filename,
                config=self.config,
                epoch=epoch,
                metrics=metrics,
            )
            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.add_file(str(local_path), name=filename)
            logged_artifact = self.run.log_artifact(artifact, aliases=aliases)
            logged_artifact.wait()
        clear_wandb_artifact_cache()
        return artifact_name, True

    def save_validation_checkpoint(
        self,
        epoch: int,
        metrics: Mapping[str, float | int | None],
    ) -> tuple[str | None, bool]:
        if not bool(self.logging_cfg["wandb_checkpoints"]) or self.run is None:
            return None, False

        filename = f"{self.experiment_name}_validation_epoch_{epoch}.pt"
        artifact_name = f"{self.experiment_name}_validation"
        previous_artifact = self._last_validation_artifact
        previous_epoch = self._last_validation_artifact_epoch
        if previous_artifact is None:
            try:
                entity = getattr(self.run, "entity", None)
                project = getattr(self.run, "project", None)
                if entity and project:
                    previous_artifact = wandb.Api().artifact(
                        f"{entity}/{project}/{artifact_name}:latest-validation",
                        type="model",
                    )
            except Exception:
                previous_artifact = None

        try:
            artifact_path, uploaded = self.save_checkpoint(
                epoch,
                metrics,
                filename,
                artifact_name=artifact_name,
                aliases=["latest-validation", f"epoch-{epoch}"],
            )
            if not uploaded:
                return None, False
            self._last_validation_artifact_epoch = int(epoch)
            self._last_validation_checkpoint_name = filename

            if previous_artifact is not None:
                try:
                    previous_artifact.delete(delete_aliases=True)
                    print(
                        f"checkpoint_deleted=wandb previous_validation epoch={previous_epoch}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"checkpoint_delete_warning=wandb previous_validation epoch={previous_epoch} "
                        f"error={exc}",
                        flush=True,
                    )
            prune_validation_artifact_cache(self.experiment_name)
            clear_wandb_artifact_cache()
            return artifact_path, True
        except Exception as exc:
            print(
                f"checkpoint_upload_warning=wandb validation epoch={epoch} error={exc}",
                flush=True,
            )
            clear_wandb_artifact_cache()
            return None, False

    def save_shutdown_checkpoint(
        self,
        *,
        epoch: int | None,
        metrics: Mapping[str, float | int | None] | None,
        reason: str,
    ) -> str | None:
        # Writes a final WandB checkpoint artifact so interrupted or completed runs can resume cleanly.
        if epoch is None or metrics is None or self.run is None or not bool(self.logging_cfg["wandb_checkpoints"]):
            return None

        checkpoint_metrics = dict(metrics)
        checkpoint_metrics["shutdown_requested"] = 1.0 if reason in {"stop_requested", "keyboard_interrupt"} else 0.0
        checkpoint_metrics["completed_run"] = 1.0 if reason == "completed" else 0.0

        filename = f"{self.experiment_name}_latest_epoch_{epoch}.pt"
        artifact_name = f"{self.experiment_name}_latest"
        previous_artifact = None
        try:
            entity = getattr(self.run, "entity", None)
            project = getattr(self.run, "project", None)
            if entity and project:
                previous_artifact = wandb.Api().artifact(
                    f"{entity}/{project}/{artifact_name}:latest",
                    type="model",
                )
        except Exception:
            previous_artifact = None

        artifact_path, uploaded = self.save_checkpoint(
            epoch=epoch,
            metrics=checkpoint_metrics,
            filename=filename,
            artifact_name=artifact_name,
            aliases=["latest", f"epoch-{epoch}", reason],
        )
        if not uploaded:
            return None
        try:
            if previous_artifact is None:
                return artifact_path
            previous_artifact.delete(delete_aliases=True)
        except Exception as exc:
            print(f"checkpoint_delete_warning=wandb previous_latest error={exc}", flush=True)
        clear_wandb_artifact_cache()
        return artifact_path

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
            "loss_sg_lattice": val_loss_metrics["loss_sg_lattice"],
            "loss_sg_lattice_weighted": val_loss_metrics["loss_sg_lattice_weighted"],
            "loss_sg_lattice_lambda_scaled": val_loss_metrics["loss_sg_lattice_lambda_scaled"],
            "loss_conv_sg": val_loss_metrics["loss_conv_sg"],
            "loss_conv_sg_weighted": val_loss_metrics["loss_conv_sg_weighted"],
            "loss_conv_sg_lambda_scaled": val_loss_metrics["loss_conv_sg_lambda_scaled"],
            "lambda_sg_lattice": val_loss_metrics["lambda_sg_lattice"],
            "lambda_conv_sg": val_loss_metrics["lambda_conv_sg"],
            "lattice_sg_time_weight_mean": val_loss_metrics["lattice_sg_time_weight_mean"],
            "conv_sg_time_weight_mean": val_loss_metrics["conv_sg_time_weight_mean"],
            "conv_weight_mean": val_loss_metrics["conv_weight_mean"],
            "projection_error_pred_k": val_loss_metrics["projection_error_pred_k"],
            "projection_error_gt_k": val_loss_metrics["projection_error_gt_k"],
            "primitive_projection_error_pred_k": val_loss_metrics["primitive_projection_error_pred_k"],
            "primitive_projection_error_gt_k": val_loss_metrics["primitive_projection_error_gt_k"],
            "conv_projection_error_pred_k": val_loss_metrics["conv_projection_error_pred_k"],
            "conv_projection_error_gt_k": val_loss_metrics["conv_projection_error_gt_k"],
            "projection_error_pred_orbit_k": val_loss_metrics["projection_error_pred_orbit_k"],
            "projection_error_gt_orbit_k": val_loss_metrics["projection_error_gt_orbit_k"],
            "loss_v_weighted": val_loss_metrics["loss_v_weighted"],
            "loss_l_weighted": val_loss_metrics["loss_l_weighted"],
            "loss_weighted": val_loss_metrics["loss_weighted"],
            "valid": val_sample_metrics["valid"],
            "match_rate": val_sample_metrics["match_rate"],
            "rmse": val_sample_metrics["rmse"],
            "frac_rmse": val_sample_metrics.get("frac_rmse"),
            "detected_family_agreement": val_sample_metrics.get("detected_family_agreement"),
            "detected_sg_agreement": val_sample_metrics.get("detected_sg_agreement"),
            "wyckoff_letter_agreement": val_sample_metrics.get("wyckoff_letter_agreement"),
            "predicted_wyckoff_dimensionality_distribution": val_sample_metrics.get("predicted_wyckoff_dimensionality_distribution"),
            "target_wyckoff_dimensionality_distribution": val_sample_metrics.get("target_wyckoff_dimensionality_distribution"),
            "lattice_lengths_rmse": val_sample_metrics.get("lattice_lengths_rmse"),
            "lattice_angles_rmse": val_sample_metrics.get("lattice_angles_rmse"),
            "volume_rel_error": val_sample_metrics.get("volume_rel_error"),
            "sample_projection_error_pred_direct_k": val_sample_metrics.get("projection_error_pred_direct_k"),
            "sample_projection_error_gt_direct_k": val_sample_metrics.get("projection_error_gt_direct_k"),
            "sample_projection_error_pred_orbit_k": val_sample_metrics.get("projection_error_pred_orbit_k"),
            "sample_projection_error_gt_orbit_k": val_sample_metrics.get("projection_error_gt_orbit_k"),
        }
        log_data = {
            "epoch": epoch,
            "val/loss_v": merged_metrics["loss_v"],
            "val/loss_l": merged_metrics["loss_l"],
            "val/loss_v_weighted": merged_metrics["loss_v_weighted"],
            "val/loss_l_weighted": merged_metrics["loss_l_weighted"],
            "val/loss_weighted": merged_metrics["loss_weighted"],
            "val/valid": merged_metrics["valid"],
            "val/match_rate": merged_metrics["match_rate"],
            "val/rmse": merged_metrics["rmse"],
            "val/frac_rmse": merged_metrics["frac_rmse"],
            "val/detected_family_agreement": merged_metrics["detected_family_agreement"],
            "val/detected_sg_agreement": merged_metrics["detected_sg_agreement"],
            "val/wyckoff_letter_agreement": merged_metrics["wyckoff_letter_agreement"],
            "val/lattice_lengths_rmse": merged_metrics["lattice_lengths_rmse"],
            "val/lattice_angles_rmse": merged_metrics["lattice_angles_rmse"],
            "val/volume_rel_error": merged_metrics["volume_rel_error"],
        }
        for metric_name in (
            "valid",
            "match_rate",
            "rmse",
            "frac_rmse",
            "detected_family_agreement",
            "detected_sg_agreement",
            "lattice_lengths_rmse",
            "lattice_angles_rmse",
            "volume_rel_error",
        ):
            std_key = f"{metric_name}_std"
            if std_key in val_sample_metrics:
                log_data[f"val/{std_key}"] = val_sample_metrics.get(std_key)
        if "num_sampling_seeds" in val_sample_metrics:
            log_data["val/num_sampling_seeds"] = val_sample_metrics.get("num_sampling_seeds")
        self.add_lattice_log_data(log_data, merged_metrics, prefix="val")
        if self.lattice_debug_enabled():
            log_data.update(
                {
                    "val/sample_projection_error_pred_direct_k": merged_metrics["sample_projection_error_pred_direct_k"],
                    "val/sample_projection_error_gt_direct_k": merged_metrics["sample_projection_error_gt_direct_k"],
                    "val/sample_projection_error_pred_orbit_k": merged_metrics["sample_projection_error_pred_orbit_k"],
                    "val/sample_projection_error_gt_orbit_k": merged_metrics["sample_projection_error_gt_orbit_k"],
                }
            )
        self.run.log(log_data, step=epoch)

        checkpoint_path, checkpoint_uploaded = self.save_validation_checkpoint(epoch, merged_metrics)

        print(
            f"validation_epoch={epoch:04d} val_loss_weighted={merged_metrics['loss_weighted']:.6f} "
            f"(loss_v={merged_metrics['loss_v']:.6f}, loss_l={merged_metrics['loss_l']:.6f}, "
            f"{self.lattice_status_text(merged_metrics)}, "
            f"loss_v_weighted={merged_metrics['loss_v_weighted']:.6f}, "
            f"loss_l_weighted={merged_metrics['loss_l_weighted']:.6f}) "
            f"valid={format_metric(merged_metrics['valid'], '.4f')} "
            f"match_rate={format_metric(merged_metrics['match_rate'], '.4f')} "
            f"rmse={format_metric(merged_metrics['rmse'], '.6f')} "
            f"frac_rmse={format_metric(merged_metrics['frac_rmse'], '.6f')} "
            f"family_agreement={format_metric(merged_metrics['detected_family_agreement'], '.4f')} "
            f"sg_agreement={format_metric(merged_metrics['detected_sg_agreement'], '.4f')} "
            f"wyckoff_letter_agreement={format_metric(merged_metrics['wyckoff_letter_agreement'], '.4f')} "
            f"lengths_rmse={format_metric(merged_metrics['lattice_lengths_rmse'], '.6f')} "
            f"angles_rmse={format_metric(merged_metrics['lattice_angles_rmse'], '.6f')} "
            f"volume_rel_error={format_metric(merged_metrics['volume_rel_error'], '.6f')}",
            flush=True,
        )
        if self.lattice_debug_enabled():
            print(
                "validation_lattice_debug "
                f"sample_proj_pred_direct_k={format_metric(merged_metrics['sample_projection_error_pred_direct_k'], '.6f')} "
                f"sample_proj_pred_orbit_k={format_metric(merged_metrics['sample_projection_error_pred_orbit_k'], '.6f')}",
                flush=True,
            )
        print(
            "validation_wyckoff_dimensionality "
            f"pred={merged_metrics['predicted_wyckoff_dimensionality_distribution']} "
            f"target={merged_metrics['target_wyckoff_dimensionality_distribution']}",
            flush=True,
        )
        print(f"validation_checkpoint_saved path={checkpoint_path} epoch={epoch}", flush=True)
        if checkpoint_uploaded:
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
        print(
            "model_lattice_weights "
            f"lambda_l={float(getattr(self.model, 'lambda_l', 1.0)):.6f} "
            f"lambda_sg_lattice={float(getattr(self.model, 'lattice_sg_lambda', 0.0)):.6f} "
            f"lattice_sg_time_weight={getattr(self.model, 'lattice_sg_time_weight', 'none')} "
            f"lambda_conv_sg={float(getattr(self.model, 'lambda_conv_sg', 0.0)):.6f} "
            f"conv_sg_time_weight={getattr(self.model, 'conv_sg_time_weight', 'none')} "
            f"lattice_debug={bool(getattr(self.model, 'lattice_debug', False))} "
            f"orbit_metric_max_candidates={getattr(self.model, 'lattice_orbit_metric_max_candidates', None)}",
            flush=True,
        )
        print(
            f"validation_schedule every_n_epochs={self.validate_every_epochs} "
            f"sampling_seed={int(self.validation_cfg.get('sampling_seed', 0))}",
            flush=True,
        )

        epoch = self.start_epoch + 1
        interrupted = False
        last_completed_epoch: int | None = None
        last_metrics: dict[str, float | int | None] | None = None
        shutdown_reason = "completed"
        try:
            while not should_stop(self.run) and (self.max_epochs is None or epoch <= self.max_epochs):
                train_metrics = self.train_epoch(epoch)
                last_completed_epoch = epoch
                last_metrics = dict(train_metrics)

                if epoch % self.train_every_epochs == 0:
                    log_data = {
                        "epoch": epoch,
                        "train/loss_v": train_metrics["loss_v"],
                        "train/loss_l": train_metrics["loss_l"],
                        "train/loss_v_weighted": train_metrics["loss_v_weighted"],
                        "train/loss_l_weighted": train_metrics["loss_l_weighted"],
                        "train/loss_weighted": train_metrics["loss_weighted"],
                        "train/epoch_seconds": train_metrics["epoch_seconds"],
                        "train/data_wait_seconds": train_metrics["data_wait_seconds"],
                        "train/to_device_seconds": train_metrics["to_device_seconds"],
                        "train/step_seconds": train_metrics["step_seconds"],
                        "train/ema_seconds": train_metrics["ema_seconds"],
                    }
                    self.add_lattice_log_data(log_data, train_metrics, prefix="train")
                    for key, value in train_metrics.items():
                        if key.startswith("time_sampler/"):
                            log_data[key] = value
                    self.run.log(log_data, step=epoch)

                    print(
                        f"epoch={epoch:04d} train_loss_weighted={train_metrics['loss_weighted']:.6f} "
                        f"(loss_v={train_metrics['loss_v']:.6f}, loss_l={train_metrics['loss_l']:.6f}, "
                        f"{self.lattice_status_text(train_metrics)}, "
                        f"loss_v_weighted={train_metrics['loss_v_weighted']:.6f}, "
                        f"loss_l_weighted={train_metrics['loss_l_weighted']:.6f}) "
                        f"timing(epoch_s={train_metrics['epoch_seconds']:.1f}, "
                        f"data_wait_s={train_metrics['data_wait_seconds']:.1f}, "
                        f"to_device_s={train_metrics['to_device_seconds']:.1f}, "
                        f"step_s={train_metrics['step_seconds']:.1f}, "
                        f"ema_s={train_metrics['ema_seconds']:.1f})",
                        flush=True,
                    )

                if not should_stop(self.run):
                    validate_now = self.validate_every_epochs > 0 and epoch % self.validate_every_epochs == 0
                    if validate_now:
                        self.validate_epoch(epoch)
                else:
                    shutdown_reason = "stop_requested"
                    break

                epoch += 1

            if should_stop(self.run) and shutdown_reason != "stop_requested":
                shutdown_reason = "stop_requested"
        except KeyboardInterrupt:
            interrupted = True
            shutdown_reason = "keyboard_interrupt"
            print("run_experiment interrupted", flush=True)
        finally:
            checkpoint_path = self.save_shutdown_checkpoint(
                epoch=last_completed_epoch,
                metrics=last_metrics,
                reason=shutdown_reason,
            )
            if checkpoint_path is not None:
                print(
                    f"shutdown_checkpoint_saved path={checkpoint_path} epoch={last_completed_epoch} "
                    f"reason={shutdown_reason}",
                    flush=True,
                )
            clear_local_checkpoint_dir(self.config, self.experiment_name)
            clear_wandb_artifact_cache()
            if self.run is not None:
                self.run.finish()


def main() -> None:
    ExperimentRunner(parse_args().config).run_training_loop()


if __name__ == "__main__":
    main()
