from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from kldmPlus.run_experiment import (
    WORKSPACE_ROOT,
    clear_wandb_artifact_cache,
    format_metric,
    load_experiment_config,
    make_fixed_subset,
    resolve_checkpoint_reference,
)
from kldmPlus.sample_evaluation import aggregate_csp_reconstruction_metrics, evaluate_csp_reconstruction
from kldmPlus.utils.device import get_default_device
from kldmPlus.utils.model_loader import build_model, load_checkpoint

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise ImportError("wandb is required for src/kldmPlus/run_sampler_ablation.py") from exc


DEFAULT_METHODS = ("em", "pc")
DEFAULT_STEPS = (300, 600, 1000)
PROGRESS_ROOT = WORKSPACE_ROOT / "artifacts" / "HPC" / "sampler_ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EM/PC sampler-step ablation for one KLDMPlus checkpoint."
    )
    parser.add_argument("--checkpoint-path", required=True, help="Local checkpoint path or WandB artifact file URL.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config. If omitted, the checkpoint's embedded config is used.",
    )
    parser.add_argument("--split", default="test", choices=("test",), help="Evaluation split.")
    parser.add_argument("--num-targets", type=int, default=256, help="Fixed test subset size.")
    parser.add_argument("--subset-seed", type=int, default=123, help="Seed for the fixed test subset.")
    parser.add_argument("--from-seed", type=int, default=0, help="First sampling seed.")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of one-sample seeds.")
    parser.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size.")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=("em", "pc"))
    parser.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--t-start", type=float, default=1.0)
    parser.add_argument("--t-final", type=float, default=1.0e-3)
    parser.add_argument("--pc-tau", type=float, default=0.15)
    parser.add_argument("--pc-corrections", type=int, default=1)
    parser.add_argument("--validity-cutoff", type=float, default=0.5)
    parser.add_argument("--progress-dir", default=str(PROGRESS_ROOT))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-project", default="kldmplus_sampler_ablation")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-disabled", action="store_true")
    parser.add_argument("--keep-artifact-cache", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(args: argparse.Namespace, checkpoint_path: Path) -> dict[str, Any]:
    if args.config is not None:
        _path, config = load_experiment_config(args.config)
        return config

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(
            "Checkpoint does not contain an embedded config. "
            "Pass --config configs/...yaml so the model and data pipeline can be rebuilt."
        )
    return config


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.config is not None:
        config_path = Path(args.config).expanduser().resolve()
    else:
        config_path = WORKSPACE_ROOT / "sampler_ablation_reference.yaml"
    return resolve_checkpoint_reference(args.checkpoint_path, config_path=config_path)


def stable_run_name(args: argparse.Namespace, checkpoint_path: Path) -> str:
    if args.run_name:
        return str(args.run_name)
    digest = hashlib.blake2b(str(checkpoint_path).encode("utf-8"), digest_size=5).hexdigest()
    return f"sampler_ablation_{checkpoint_path.stem}_{digest}"


def make_combo_key(method: str, steps: int) -> str:
    return f"{method.upper()}{int(steps)}"


def new_combo_state(num_targets: int) -> dict[str, Any]:
    return {
        "completed_seeds": [],
        "seed_summaries": [],
        "target_valid_count": [0 for _ in range(num_targets)],
        "target_hit_count": [0 for _ in range(num_targets)],
        "target_best_rmse": [None for _ in range(num_targets)],
    }


def load_state(path: Path, *, run_id: str, args: argparse.Namespace, checkpoint_path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        print(f"sampler_ablation_resume_state path={path}", flush=True)
        return state

    state = {
        "version": 1,
        "run_id": run_id,
        "checkpoint_path": str(checkpoint_path),
        "split": args.split,
        "num_targets": int(args.num_targets),
        "subset_seed": int(args.subset_seed),
        "from_seed": int(args.from_seed),
        "n_seeds": int(args.n_seeds),
        "methods": list(args.methods),
        "steps": [int(step) for step in args.steps],
        "combos": {},
    }
    save_state(state, path)
    return state


def save_state(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def update_combo_state(combo_state: dict[str, Any], *, seed: int, results: list[Any], summary: dict[str, Any]) -> None:
    completed = {int(value) for value in combo_state.get("completed_seeds", [])}
    if int(seed) in completed:
        return

    target_valid = list(combo_state.get("target_valid_count", []))
    target_hits = list(combo_state.get("target_hit_count", []))
    target_best_rmse = list(combo_state.get("target_best_rmse", []))
    if len(target_valid) != len(results) or len(target_hits) != len(results) or len(target_best_rmse) != len(results):
        raise ValueError(
            "Sampler ablation state target count mismatch: "
            f"valid={len(target_valid)} hits={len(target_hits)} best_rmse={len(target_best_rmse)} "
            f"results={len(results)}"
        )

    for target_idx, result in enumerate(results):
        if result.valid:
            target_valid[target_idx] = int(target_valid[target_idx]) + 1
        if not result.match or result.rmse is None:
            continue
        target_hits[target_idx] = int(target_hits[target_idx]) + 1
        current_rmse = float(result.rmse)
        best_rmse = target_best_rmse[target_idx]
        target_best_rmse[target_idx] = current_rmse if best_rmse is None else min(float(best_rmse), current_rmse)

    seed_summaries = [item for item in combo_state.get("seed_summaries", []) if int(item["seed"]) != int(seed)]
    seed_summary = {
        "seed": int(seed),
        "valid": none_or_float(summary.get("valid")),
        "match_rate": none_or_float(summary.get("match_rate")),
        "rmse": none_or_float(summary.get("rmse")),
        "frac_rmse": none_or_float(summary.get("frac_rmse")),
        "length_rmse": none_or_float(summary.get("lattice_lengths_rmse")),
        "angles_rmse": none_or_float(summary.get("lattice_angles_rmse")),
        "space_group_agreement": none_or_float(summary.get("requested_space_group_match_rate")),
        "family_agreement": none_or_float(summary.get("detected_family_agreement")),
        "relaxed_match_rate": none_or_float(summary.get("relaxed_match_rate")),
        "relaxed_rmse": none_or_float(summary.get("relaxed_rmse")),
    }
    seed_summaries.append(seed_summary)

    combo_state["completed_seeds"] = sorted([*completed, int(seed)])
    combo_state["seed_summaries"] = sorted(seed_summaries, key=lambda item: int(item["seed"]))
    combo_state["target_valid_count"] = target_valid
    combo_state["target_hit_count"] = target_hits
    combo_state["target_best_rmse"] = target_best_rmse


def none_or_float(value: Any) -> float | None:
    return None if value is None else float(value)


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array))


def summarize_combo(combo_key: str, combo_state: dict[str, Any]) -> dict[str, Any]:
    seed_summaries = list(combo_state.get("seed_summaries", []))
    target_valid = np.asarray(combo_state.get("target_valid_count", []), dtype=int)
    target_hits = np.asarray(combo_state.get("target_hit_count", []), dtype=int)
    best_rmse = [float(value) for value in combo_state.get("target_best_rmse", []) if value is not None]

    row: dict[str, Any] = {
        "combo": combo_key,
        "completed_seed_count": len(combo_state.get("completed_seeds", [])),
        "completed_seeds": ",".join(str(seed) for seed in combo_state.get("completed_seeds", [])),
        "at5_valid_rate": None if target_valid.size == 0 else float(np.mean(target_valid > 0)),
        "at5_match_rate": None if target_hits.size == 0 else float(np.mean(target_hits > 0)),
        "at5_rmse": None if not best_rmse else float(np.mean(best_rmse)),
    }
    for key in (
        "valid",
        "match_rate",
        "rmse",
        "frac_rmse",
        "length_rmse",
        "angles_rmse",
        "space_group_agreement",
        "family_agreement",
        "relaxed_match_rate",
        "relaxed_rmse",
    ):
        values = [float(item[key]) for item in seed_summaries if item.get(key) is not None]
        mean_value, std_value = mean_std(values)
        row[f"at1_{key}_mean"] = mean_value
        row[f"at1_{key}_std"] = std_value
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class SamplerAblationRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = get_default_device()
        self.checkpoint_path = resolve_checkpoint(args)
        self.config = load_config(args, self.checkpoint_path)
        self.run_id = stable_run_name(args, self.checkpoint_path)
        self.progress_dir = Path(args.progress_dir).expanduser().resolve()
        self.state_path = self.progress_dir / f"{self.run_id}.json"
        self.summary_csv_path = self.progress_dir / f"{self.run_id}_summary.csv"
        self.seed_csv_path = self.progress_dir / f"{self.run_id}_seeds.csv"
        self.loader, self.lattice_transform = self.build_loader()
        self.model = build_model(config=self.config, device=self.device)
        load_checkpoint(
            checkpoint_path=self.checkpoint_path,
            model=self.model,
            device=self.device,
            prefer_ema_weights=True,
        )
        self.model.eval()

    def build_loader(self) -> tuple[DataLoader, Any]:
        from kldmPlus.data import CSPTask, resolve_data_root
        from kldmPlus.data.csp import validate_lattice_configuration

        dataset_cfg = dict(self.config["dataset"])
        model_cfg = dict(self.config["model"])
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
        dataset_full = task.fit_dataset(root=root, split=self.args.split, download=True)
        dataset = make_fixed_subset(
            dataset_full,
            subset_size=int(self.args.num_targets),
            seed=int(self.args.subset_seed),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(self.args.batch_size),
            shuffle=False,
            num_workers=int(dataset_cfg.get("num_workers", 0)),
            pin_memory=bool(dataset_cfg.get("pin_memory", False)),
            collate_fn=dataset_full.collate_fn,
        )
        return loader, task.make_lattice_transform(root=root, download=True)

    def sample_batch(self, batch: Any, *, method: str, steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        kwargs = {
            "n_steps": int(steps),
            "batch": batch,
            "t_start": float(self.args.t_start),
            "t_final": float(self.args.t_final),
        }
        if method == "em":
            return self.model.sample_CSP_algorithm3(**kwargs)
        if method == "pc":
            return self.model.sample_CSP_algorithm4(
                **kwargs,
                tau=float(self.args.pc_tau),
                n_correction_steps=int(self.args.pc_corrections),
            )
        raise ValueError(f"Unsupported method={method!r}")

    def collect_seed(self, *, method: str, steps: int, seed: int) -> tuple[list[Any], dict[str, Any]]:
        set_seed(seed)
        self.model.eval()
        results: list[Any] = []
        started_at = time.perf_counter()
        total_batches = len(self.loader)
        print(
            f"sampler_ablation_seed_start combo={make_combo_key(method, steps)} seed={seed} "
            f"batches={total_batches}",
            flush=True,
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.loader, start=1):
                batch = batch.to(self.device)
                pos_t, _v_t, l_t, h_t = self.sample_batch(batch, method=method, steps=steps)
                ptr = batch.ptr.tolist()
                for graph_idx, (start_idx, end_idx) in enumerate(zip(ptr[:-1], ptr[1:])):
                    requested_sg = None
                    if hasattr(batch, "space_group"):
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
                            validity_cutoff=float(self.args.validity_cutoff),
                        )
                    )
                print(
                    f"sampler_ablation_batch combo={make_combo_key(method, steps)} seed={seed} "
                    f"batch={batch_idx}/{total_batches} elapsed_s={time.perf_counter() - started_at:.1f}",
                    flush=True,
                )

        summary = aggregate_csp_reconstruction_metrics(results)
        print(
            f"sampler_ablation_seed_done combo={make_combo_key(method, steps)} seed={seed} "
            f"valid={format_metric(summary.get('valid'), '.4f')} "
            f"match_rate={format_metric(summary.get('match_rate'), '.4f')} "
            f"rmse={format_metric(summary.get('rmse'), '.6f')} "
            f"frac_rmse={format_metric(summary.get('frac_rmse'), '.6f')} "
            f"length_rmse={format_metric(summary.get('lattice_lengths_rmse'), '.6f')} "
            f"angles_rmse={format_metric(summary.get('lattice_angles_rmse'), '.6f')} "
            f"elapsed_s={time.perf_counter() - started_at:.1f}",
            flush=True,
        )
        return results, summary

    def run(self) -> None:
        state = load_state(
            self.state_path,
            run_id=self.run_id,
            args=self.args,
            checkpoint_path=self.checkpoint_path,
        )

        wandb_run = None
        if not self.args.wandb_disabled:
            wandb_run = wandb.init(
                project=str(self.args.wandb_project),
                name=str(self.args.wandb_run_name or self.run_id),
                config={
                    "checkpoint_path": str(self.checkpoint_path),
                    "run_id": self.run_id,
                    "split": self.args.split,
                    "num_targets": int(self.args.num_targets),
                    "subset_seed": int(self.args.subset_seed),
                    "from_seed": int(self.args.from_seed),
                    "n_seeds": int(self.args.n_seeds),
                    "methods": list(self.args.methods),
                    "steps": [int(step) for step in self.args.steps],
                    "t_start": float(self.args.t_start),
                    "t_final": float(self.args.t_final),
                    "pc_tau": float(self.args.pc_tau),
                    "pc_corrections": int(self.args.pc_corrections),
                },
            )

        for method in self.args.methods:
            for steps in self.args.steps:
                combo_key = make_combo_key(str(method), int(steps))
                combo_state = state["combos"].setdefault(combo_key, new_combo_state(len(self.loader.dataset)))
                completed = {int(seed) for seed in combo_state.get("completed_seeds", [])}
                for seed in range(int(self.args.from_seed), int(self.args.from_seed) + int(self.args.n_seeds)):
                    if seed in completed:
                        print(f"sampler_ablation_skip combo={combo_key} seed={seed}", flush=True)
                        continue
                    results, summary = self.collect_seed(method=str(method), steps=int(steps), seed=int(seed))
                    update_combo_state(combo_state, seed=int(seed), results=results, summary=summary)
                    save_state(state, self.state_path)
                    if wandb_run is not None:
                        wandb.log(
                            {
                                "seed/combo": combo_key,
                                "seed/seed": int(seed),
                                f"{combo_key}/latest_valid": summary.get("valid"),
                                f"{combo_key}/latest_match_rate": summary.get("match_rate"),
                                f"{combo_key}/latest_rmse": summary.get("rmse"),
                                f"{combo_key}/latest_frac_rmse": summary.get("frac_rmse"),
                                f"{combo_key}/latest_length_rmse": summary.get("lattice_lengths_rmse"),
                                f"{combo_key}/latest_angles_rmse": summary.get("lattice_angles_rmse"),
                            }
                        )
                    completed = {int(seed_value) for seed_value in combo_state.get("completed_seeds", [])}

        summary_rows = [summarize_combo(key, value) for key, value in sorted(state["combos"].items())]
        seed_rows = []
        for combo_key, combo_state in sorted(state["combos"].items()):
            for seed_summary in combo_state.get("seed_summaries", []):
                seed_rows.append({"combo": combo_key, **seed_summary})
        write_csv(summary_rows, self.summary_csv_path)
        write_csv(seed_rows, self.seed_csv_path)

        if wandb_run is not None:
            summary_table = wandb.Table(columns=sorted({key for row in summary_rows for key in row}))
            for row in summary_rows:
                summary_table.add_data(*[row.get(column) for column in summary_table.columns])
            seed_table = wandb.Table(columns=sorted({key for row in seed_rows for key in row}))
            for row in seed_rows:
                seed_table.add_data(*[row.get(column) for column in seed_table.columns])
            log_data: dict[str, Any] = {
                "ablation/summary": summary_table,
                "ablation/seeds": seed_table,
            }
            for row in summary_rows:
                combo = str(row["combo"])
                log_data[f"{combo}/at1_match_rate_mean"] = row.get("at1_match_rate_mean")
                log_data[f"{combo}/at1_match_rate_std"] = row.get("at1_match_rate_std")
                log_data[f"{combo}/at1_rmse_mean"] = row.get("at1_rmse_mean")
                log_data[f"{combo}/at1_rmse_std"] = row.get("at1_rmse_std")
                log_data[f"{combo}/at5_match_rate"] = row.get("at5_match_rate")
                log_data[f"{combo}/at5_rmse"] = row.get("at5_rmse")
            wandb.log(log_data)
            wandb.finish()

        print(f"sampler_ablation_summary_csv={self.summary_csv_path}", flush=True)
        print(f"sampler_ablation_seed_csv={self.seed_csv_path}", flush=True)
        for row in summary_rows:
            print(
                f"{row['combo']} "
                f"seeds={row.get('completed_seeds')} "
                f"@1_match={format_metric(row.get('at1_match_rate_mean'), '.4f')}±"
                f"{format_metric(row.get('at1_match_rate_std'), '.4f')} "
                f"@1_rmse={format_metric(row.get('at1_rmse_mean'), '.6f')}±"
                f"{format_metric(row.get('at1_rmse_std'), '.6f')} "
                f"@5_match={format_metric(row.get('at5_match_rate'), '.4f')} "
                f"@5_rmse={format_metric(row.get('at5_rmse'), '.6f')}",
                flush=True,
            )

        if not self.args.keep_artifact_cache:
            clear_wandb_artifact_cache()


def main() -> None:
    SamplerAblationRunner(parse_args()).run()


if __name__ == "__main__":
    main()
