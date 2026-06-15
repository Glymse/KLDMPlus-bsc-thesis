from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import random
import re
import signal
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping
from urllib.parse import unquote

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from kldm.utils.device import get_default_device
from kldm.utils.time import sample_times

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise ImportError("wandb is required for src/kldm/run_experiment.py") from exc


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS_ROOT = WORKSPACE_ROOT / "artifacts" / "HPC" / "checkpoints" / "experiments"
TIME_LOWER_BOUND = 1e-3
STOP_REQUESTED = False
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TRAIN_SEED = 2002


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


def set_global_training_seed(seed: int = TRAIN_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker_factory(base_seed: int):
    def _seed_worker(worker_id: int) -> None:
        worker_seed = int(base_seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _seed_worker


def format_metric(value: float | int | None, fmt: str) -> str:
    if value is None:
        return "na"
    return format(value, fmt)


def checkpoint_dir(config: dict[str, Any], experiment_name: str) -> Path:
    del config
    return CHECKPOINTS_ROOT / experiment_name


def save_named_checkpoint(
    *,
    model,
    optimizer: torch.optim.Optimizer,
    ema,
    config: dict[str, Any],
    experiment_name: str,
    epoch: int,
    metrics: Mapping[str, float | int | None],
    filename: str,
    keep_paths: list[Path] | None = None,
) -> Path:
    from kldm.utils.model_loader import save_checkpoint

    output_dir = checkpoint_dir(config=config, experiment_name=experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    if output_path.exists():
        output_path.unlink()
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        ema=ema,
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


WANDB_ARTIFACT_URL_RE = re.compile(
    r"^https?://wandb\.ai/"
    r"(?P<entity>[^/]+)/"
    r"(?P<project>[^/]+)/"
    r"artifacts/"
    r"(?P<artifact_type>[^/]+)/"
    r"(?P<artifact_name>[^/]+)/"
    r"(?P<version>v[^/]+)"
    r"(?:/files/(?P<file_path>.+))?$"
)


def _artifact_cache_dir(experiment_name: str, entity: str, project: str, artifact_name: str, version: str) -> Path:
    safe_name = "__".join((entity, project, artifact_name, version))
    return CHECKPOINTS_ROOT / experiment_name / "resume_artifacts" / safe_name


def _parse_wandb_artifact_url(resume_from: str) -> dict[str, str] | None:
    match = WANDB_ARTIFACT_URL_RE.match(str(resume_from).strip())
    if match is None:
        return None
    parsed = match.groupdict()
    parsed["file_path"] = unquote(parsed.get("file_path") or "")
    return parsed


def resolve_resume_checkpoint(
    *,
    resume_from: str,
    config_path: Path,
    experiment_name: str,
) -> Path:
    artifact_info = _parse_wandb_artifact_url(resume_from)
    if artifact_info is None:
        return (config_path.parent / str(resume_from)).expanduser().resolve()

    artifact_cache_dir = _artifact_cache_dir(
        experiment_name=experiment_name,
        entity=artifact_info["entity"],
        project=artifact_info["project"],
        artifact_name=artifact_info["artifact_name"],
        version=artifact_info["version"],
    )
    requested_file = artifact_info["file_path"]
    if not requested_file:
        raise ValueError(
            "W&B artifact resume link must point to a specific checkpoint file under /files/..."
        )

    resolved_path = artifact_cache_dir / requested_file
    if resolved_path.exists():
        print(f"resume_checkpoint_cached={resolved_path}", flush=True)
        return resolved_path

    artifact_ref = (
        f"{artifact_info['entity']}/"
        f"{artifact_info['project']}/"
        f"{artifact_info['artifact_name']}:"
        f"{artifact_info['version']}"
    )

    artifact_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"resume_checkpoint_download=wandb artifact={artifact_ref}", flush=True)
    artifact = wandb.Api().artifact(
        artifact_ref,
        type=artifact_info["artifact_type"],
    )
    artifact.download(root=str(artifact_cache_dir))

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Downloaded W&B artifact but could not find requested file: {resolved_path}"
        )

    print(f"resume_checkpoint_downloaded={resolved_path}", flush=True)
    return resolved_path


class ExperimentRunner:
    def __init__(self, config_path: str | Path) -> None:
        from kldm.utils.model_loader import build_training_components, load_checkpoint

        # -------------------------------------------------
        # Static experiment setup from config
        # -------------------------------------------------
        self.config_path, self.config = load_experiment_config(config_path)
        self.experiment_name = str(self.config["experiment_name"])

        self.sampler_cfg = dict(self.config["sampler"])
        self.logging_cfg = dict(self.config["logging"])
        self.validation_cfg = dict(self.config["validation"])
        self.checkpoint_cfg = dict(self.config["checkpoint"])

        self.train_every_epochs = int(self.logging_cfg["train_every_epochs"])
        self.validate_every_epochs = int(self.validation_cfg["every_n_epochs"])

        # -------------------------------------------------
        # Runtime objects: device, data, model, optimizer, EMA
        # -------------------------------------------------
        set_global_training_seed(TRAIN_SEED)
        self.device = get_default_device()
        self.train_loader, self.val_loader, self.lattice_transform = self.create_loaders()

        self.model, self.optimizer, self.ema = build_training_components(
            config=self.config,
            device=self.device,
        )

        self.start_epoch = 0
        self.run = None
        self._last_validation_artifact = None

        # Optional resume path for continuing training from a saved checkpoint.
        resume_from = self.checkpoint_cfg["resume_from"]
        if resume_from:
            checkpoint_path = resolve_resume_checkpoint(
                resume_from=str(resume_from),
                config_path=self.config_path,
                experiment_name=self.experiment_name,
            )
            checkpoint = load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                ema=self.ema,
                device=self.device,
                prefer_ema_weights=False,
            )
            self.start_epoch = int(checkpoint["epoch"])

    def create_loaders(self) -> tuple[DataLoader, DataLoader, Any]:
        from kldm.data import CSPTask, resolve_data_root

        dataset_cfg = dict(self.config["dataset"])
        model_cfg = dict(self.config["model"])

        task = CSPTask(
            dataset_name=str(dataset_cfg["name"]),
            lattice_parameterization=str(model_cfg["lattice_parameterization"]),
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
            collate_fn=val_dataset_full.collate_fn,
            worker_init_fn=worker_init_fn,
        )

        return train_loader, val_loader, task.make_lattice_transform(root=root, download=True)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()

        total_loss_v = total_loss_l = 0.0
        total_nodes = total_graphs = 0

        for batch in self.train_loader:
            if STOP_REQUESTED:
                break

            batch = batch.to(self.device)

            # One shared diffusion time per material graph.
            t_graph = sample_times(batch, lower_bound=TIME_LOWER_BOUND)

            self.optimizer.zero_grad(set_to_none=True)
            loss, metrics = self.model.algorithm2_loss(batch=batch, t=t_graph, debug=False)
            loss.backward()
            self.optimizer.step()

            if self.ema is not None:
                self.ema.update(self.model, current_epoch=epoch)

            total_loss_v += float(metrics["loss_v"]) * int(batch.pos.shape[0])
            total_loss_l += float(metrics["loss_l"]) * int(batch.num_graphs)
            total_nodes += int(batch.pos.shape[0])
            total_graphs += int(batch.num_graphs)

        if total_nodes == 0 or total_graphs == 0:
            raise RuntimeError("Training stopped before any batches were processed.")

        return {
            "loss_v": total_loss_v / total_nodes,
            "loss_l": total_loss_l / total_graphs,
            "loss_weighted": (total_loss_v / total_nodes) + (total_loss_l / total_graphs),
        }

    def evaluate_loss(self) -> dict[str, float]:
        self.model.eval()

        total_loss_v = total_loss_l = 0.0
        total_nodes = total_graphs = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)

            # Validation uses the same noisy-time sampling pattern as training.
            t_graph = sample_times(batch, lower_bound=TIME_LOWER_BOUND)

            with torch.no_grad():
                _, metrics = self.model.algorithm2_loss(batch=batch, t=t_graph, debug=False)

            total_loss_v += float(metrics["loss_v"]) * int(batch.pos.shape[0])
            total_loss_l += float(metrics["loss_l"]) * int(batch.num_graphs)
            total_nodes += int(batch.pos.shape[0])
            total_graphs += int(batch.num_graphs)

        if total_nodes == 0 or total_graphs == 0:
            raise RuntimeError("Validation loader is empty.")

        return {
            "loss_v": total_loss_v / total_nodes,
            "loss_l": total_loss_l / total_graphs,
            "loss_weighted": (total_loss_v / total_nodes) + (total_loss_l / total_graphs),
        }

    def run_sampling_evaluation(self) -> dict[str, float | int | None]:
        from kldm.sample_evaluation.sample_evaluation import (
            aggregate_csp_reconstruction_metrics,
            evaluate_csp_reconstruction,
        )

        self.model.eval()

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

                # Stop early if validation sampling only wants a fixed number of graphs.
                if self.validation_cfg["sampling_max_graphs"] is not None and num_graphs_seen >= self.validation_cfg["sampling_max_graphs"]:
                    break

            if self.validation_cfg["sampling_max_graphs"] is not None and num_graphs_seen >= self.validation_cfg["sampling_max_graphs"]:
                break

        summary = aggregate_csp_reconstruction_metrics(results)
        return {
            "valid": summary.get("valid"),
            "match_rate": summary.get("match_rate"),
            "rmse": summary.get("rmse"),
            "num_samples": summary.get("num_samples"),
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
        from kldm.utils.model_loader import save_checkpoint

        if not bool(self.logging_cfg["wandb_checkpoints"]):
            return

        with tempfile.TemporaryDirectory(prefix="kldm_val_ckpt_") as temp_dir_name:
            path = Path(temp_dir_name) / f"epoch_{epoch}.pt"
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                ema=self.ema,
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

    def validate_epoch(self, epoch: int) -> None:
        # Validation can optionally swap in EMA weights, depending on the config.
        ema_val = bool(self.validation_cfg["ema_val"])
        use_ema = ema_val and self.ema is not None and self.ema.num_updates > 0
        context = (
            self.ema.average_parameters(self.model)
            if use_ema and self.ema is not None
            else nullcontext()
        )
        model_label = "EMA model" if use_ema else "current model"

        print(f"epoch={epoch:04d} entering validation with {model_label}", flush=True)

        with context:
            val_loss_metrics = self.evaluate_loss()
            val_sample_metrics = self.run_sampling_evaluation()

        merged_metrics = {
            "loss_v": val_loss_metrics["loss_v"],
            "loss_l": val_loss_metrics["loss_l"],
            "loss_weighted": val_loss_metrics["loss_weighted"],
            "valid": val_sample_metrics["valid"],
            "match_rate": val_sample_metrics["match_rate"],
            "rmse": val_sample_metrics["rmse"],
        }
        wandb.log(
            {
                "epoch": epoch,
                "val/loss_v": merged_metrics["loss_v"],
                "val/loss_l": merged_metrics["loss_l"],
                "val/loss_weighted": merged_metrics["loss_weighted"],
                "val/valid": merged_metrics["valid"],
                "val/match_rate": merged_metrics["match_rate"],
                "val/rmse": merged_metrics["rmse"],
            },
            step=epoch,
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
        if bool(self.logging_cfg["wandb_checkpoints"]):
            print(f"checkpoint_uploaded=wandb epoch={epoch}", flush=True)

    def run_training_loop(self) -> None:
        # Start one wandb run for the whole experiment.
        wandb_resume_id = self.checkpoint_cfg.get("wandb_resume_id")
        init_kwargs = {
            "project": self.experiment_name,
            "config": self.config | {"start_epoch": self.start_epoch},
        }
        if wandb_resume_id:
            init_kwargs["id"] = str(wandb_resume_id)
            init_kwargs["resume"] = "must"
        else:
            init_kwargs["name"] = build_run_name()

        self.run = wandb.init(
            **init_kwargs,
        )

        print(f"run_experiment config={self.config_path}", flush=True)
        print(f"device={self.device.type} experiment={self.experiment_name}", flush=True)
        print(f"data_splits train={TRAIN_SPLIT} val={VAL_SPLIT}", flush=True)
        print(f"sampler={self.sampler_cfg}", flush=True)

        epoch = self.start_epoch + 1
        try:
            while not should_stop(self.run):
                train_metrics = self.train_epoch(epoch)

                if epoch % self.train_every_epochs == 0:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train/loss_v": train_metrics["loss_v"],
                            "train/loss_l": train_metrics["loss_l"],
                            "train/loss_weighted": train_metrics["loss_weighted"],
                        },
                        step=epoch,
                    )

                    print(
                        f"epoch={epoch:04d} train_loss_weighted={train_metrics['loss_weighted']:.6f} "
                        f"(loss_v={train_metrics['loss_v']:.6f}, loss_l={train_metrics['loss_l']:.6f})",
                        flush=True,
                    )

                if epoch % self.validate_every_epochs == 0 and not should_stop(self.run):
                    self.validate_epoch(epoch)

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
            wandb.finish()


def main() -> None:
    ExperimentRunner(parse_args().config).run_training_loop()


if __name__ == "__main__":
    main()
