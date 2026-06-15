from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from kldmPlus.run_experiment import (
    format_metric,
    load_experiment_config,
    make_fixed_subset,
    resolve_checkpoint_reference,
    should_stop,
)
from kldmPlus.sample_evaluation import prepare_visualization_pair
from kldmPlus.utils.device import get_default_device

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise ImportError("wandb is required for src/kldmPlus/run_sampling.py") from exc


TEST_SPLIT = "test"
AT_K = 20
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SAMPLING_PROGRESS_ROOT = WORKSPACE_ROOT / "artifacts" / "HPC" / "sampling_eval"
EVAL_EXTRA_METRICS = (
    ("requested_space_group_match_rate", "space_group_agreement"),
    ("detected_family_agreement", "family_agreement"),
    ("frac_rmse", "frac_rmse"),
    ("lattice_angles_rmse", "angles_rmse"),
    ("lattice_lengths_rmse", "length_rmse"),
    ("volume_rel_error", "volume_rel_error"),
)
EVAL_RELAXED_METRICS = (
    ("relaxed_match_rate", "match_rate"),
    ("relaxed_rmse", "rmse"),
    ("relaxed_near_miss_rate", "near_miss_rate"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KLDM checkpoint sampling from config.")
    parser.add_argument("--config", required=True, help="Path to the sampling YAML file.")
    return parser.parse_args()


# Returns the lowest-RMSE matched result, or a valid fallback if nothing matched.
def _best_result(results: list[Any]) -> Any:
    matched = [result for result in results if result.match and result.rmse is not None]
    if matched:
        return min(matched, key=lambda result: float(result.rmse))
    return next((result for result in results if result.valid), results[0])


# Reduces repeated evaluation passes into per-target hit counts and best RMSE values.
def _merge_pass_statistics(pass_results: list[list[Any]]) -> tuple[float | None, float | None]:
    if not pass_results:
        return None, None

    target_count = len(pass_results[0])
    hit_count = np.zeros(target_count, dtype=int)
    best_rmse = np.full(target_count, np.inf, dtype=float)

    for one_pass in pass_results:
        for target_idx, result in enumerate(one_pass):
            if not result.match or result.rmse is None:
                continue
            hit_count[target_idx] += 1
            best_rmse[target_idx] = min(best_rmse[target_idx], float(result.rmse))

    reached = hit_count > 0
    match_rate = None if target_count == 0 else float(np.mean(reached))
    rmse = None if not reached.any() else float(best_rmse[reached].mean())
    return match_rate, rmse


# Applies Python, NumPy, and Torch seeding for one repeated evaluation pass.
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _space_group_to_family(space_group_number: int | None) -> str | None:
    if space_group_number is None:
        return None
    sg = int(space_group_number)
    if not 1 <= sg <= 230:
        return None
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


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array))


def _result_metric_payload(result: Any) -> dict[str, float | None]:
    requested_family = _space_group_to_family(getattr(result, "requested_space_group", None))
    detected_family = _space_group_to_family(getattr(result, "detected_space_group", None))
    family_agreement = (
        None if requested_family is None or detected_family is None
        else float(requested_family == detected_family)
    )
    return {
        "requested_space_group_match_rate": _json_float(getattr(result, "requested_space_group_match", None)),
        "detected_family_agreement": family_agreement,
        "frac_rmse": _json_float(getattr(result, "frac_rmse", None)),
        "lattice_angles_rmse": _json_float(getattr(result, "lattice_angles_rmse", None)),
        "lattice_lengths_rmse": _json_float(getattr(result, "lattice_lengths_rmse", None)),
        "volume_rel_error": _json_float(getattr(result, "volume_rel_error", None)),
    }


class SamplingRunner:
    def __init__(self, config_path: str | Path) -> None:
        from kldmPlus.utils.model_loader import build_model, load_checkpoint

        self.config_path, self.config = load_experiment_config(config_path)
        self.experiment_name = str(self.config["experiment_name"])
        self.sampling_cfg = dict(self.config["sampling"])
        self.eval_cfg = dict(self.config["sampling_eval"])
        self.validity_cutoff = float(self.eval_cfg.get("validity_cutoff", 0.5))
        self.evaluation = bool(self.eval_cfg["evaluation"])
        self.sample_count = int(self.sampling_cfg["n_samples"])
        self.eval_seed_count = int(self.eval_cfg.get("n_seeds", AT_K))
        self.eval_from_seed = int(self.eval_cfg.get("from_seed", 0))
        self.eval_progress_dir = self._progress_dir(self.eval_cfg.get("progress_dir"))
        self.wandb_project = str(self.eval_cfg.get("wandb_project", "mp_20_sampling"))
        self.wandb_run_name = str(
            self.eval_cfg.get(
                "wandb_run_name",
                f"{'EVAL' if self.evaluation else 'SAMPLES'}_{self.experiment_name}",
            )
        )
        self.wandb_resume_id = self.eval_cfg.get("wandb_resume_id")
        self.device = get_default_device()
        self.checkpoint_path = self._checkpoint_path(self.sampling_cfg["checkpoint_path"])
        self.loader, self.lattice_transform = self._build_loader()
        self.model = build_model(config=self.config, device=self.device)
        load_checkpoint(
            checkpoint_path=self.checkpoint_path,
            model=self.model,
            device=self.device,
            prefer_ema_weights=True,
        )

    # Resolves the checkpoint path from the config file location and falls back to the latest file.
    def _checkpoint_path(self, checkpoint_path: str | Path) -> Path:
        return resolve_checkpoint_reference(checkpoint_path, config_path=self.config_path)

    def _progress_dir(self, configured_path: str | Path | None) -> Path:
        if configured_path is None:
            return SAMPLING_PROGRESS_ROOT
        candidate = Path(configured_path).expanduser()
        if not candidate.is_absolute():
            candidate = (self.config_path.parent / candidate).expanduser()
        return candidate.resolve()

    # Builds the fixed test-set loader used by both sample mode and @1/@20 evaluation mode.
    def _build_loader(self) -> tuple[DataLoader, Any]:
        from kldmPlus.data import CSPTask, resolve_data_root
        from kldmPlus.data.csp import validate_lattice_configuration

        dataset_cfg = dict(self.config["dataset"])
        model_cfg = dict(self.config["model"])
        # Sampling must use the same representation/diffusion pairing as
        # training, or the lattice branch becomes physically meaningless.
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

        requested_split = str(self.eval_cfg.get("split", TEST_SPLIT))
        if requested_split != TEST_SPLIT:
            raise ValueError(f"run_sampling always uses split={TEST_SPLIT!r}, got {requested_split!r}")

        root = resolve_data_root(dataset_cfg["root"])
        dataset_full = task.fit_dataset(root=root, split=TEST_SPLIT, download=True)
        subset_size = int(self.eval_cfg["num_targets"]) if self.evaluation else self.sample_count
        dataset = make_fixed_subset(
            dataset_full,
            subset_size=subset_size,
            seed=int(self.eval_cfg["subset_seed"]),
        )

        loader = DataLoader(
            dataset,
            batch_size=int(self.eval_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(dataset_cfg["num_workers"]),
            pin_memory=bool(dataset_cfg["pin_memory"]),
            collate_fn=dataset_full.collate_fn,
        )
        return loader, task.make_lattice_transform(
            root=root,
            download=True,
        )

    # Samples one batch with the configured KLDM sampler.
    def _sample_batch(self, batch):
        method = str(self.sampling_cfg["method"])
        sampling_algorithm = int(self.sampling_cfg.get("sampling_algorithm", 4 if method == "pc" else 3))
        kwargs = {
            "n_steps": int(self.sampling_cfg["n_steps"]),
            "batch": batch,
            "t_start": float(self.sampling_cfg["t_start"]),
            "t_final": float(self.sampling_cfg["t_final"]),
        }
        if sampling_algorithm == 4:
            kwargs["tau"] = float(self.sampling_cfg["tau"])
            kwargs["n_correction_steps"] = int(self.sampling_cfg["n_correction_steps"])
            sample_fn = self.model.sample_CSP_algorithm4
        elif sampling_algorithm == 3:
            sample_fn = self.model.sample_CSP_algorithm3
        else:
            raise ValueError(
                f"Unsupported sampling_algorithm={sampling_algorithm}. "
                "Supported sampler paths are original KLDMplus (3/4)."
            )

        return sample_fn(**kwargs)

    # Renders a predicted/actual structure pair to a small side-by-side PNG.
    @staticmethod
    def _render_pair(predicted_structure, target_structure, png_path: Path) -> None:
        from ase.visualize.plot import plot_atoms
        import matplotlib.pyplot as plt
        from pymatgen.io.ase import AseAtomsAdaptor

        def to_atoms(structure):
            atoms = AseAtomsAdaptor.get_atoms(structure)
            try:
                atoms.wrap()
            except Exception:
                pass
            return atoms

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        plot_atoms(to_atoms(predicted_structure), axes[0])
        plot_atoms(to_atoms(target_structure), axes[1])
        axes[0].set_title("Predicted")
        axes[1].set_title("Actual")
        for axis in axes:
            axis.set_axis_off()
        fig.tight_layout(pad=0.3)
        fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    # Evaluates one full loader pass, optionally seeding the random state first.
    def _collect(self, *, samples_per_target: int, seed: int | None = None) -> list[list[Any]]:
        from kldmPlus.sample_evaluation import evaluate_csp_reconstruction

        self.model.eval()
        started_at = time.perf_counter()
        if seed is not None:
            _set_seed(seed)
            print(
                f"eval_seed={seed} sampling_pass_start samples_per_target={samples_per_target}",
                flush=True,
            )
        else:
            total_batches = len(self.loader)
            print(
                f"sampling_progress phase=start total_batches={total_batches} "
                f"samples_per_target={samples_per_target}",
                flush=True,
            )

        results: list[list[Any]] = []
        total_batches = len(self.loader)
        for batch_idx, batch in enumerate(self.loader, start=1):
            batch = batch.to(self.device)
            per_graph = [[] for _ in range(batch.num_graphs)]

            if seed is not None:
                elapsed_s = time.perf_counter() - started_at
                print(
                    f"eval_seed={seed} batch={batch_idx}/{total_batches} "
                    f"graphs_in_batch={batch.num_graphs} elapsed_s={elapsed_s:.1f}",
                    flush=True,
                )
            else:
                elapsed_s = time.perf_counter() - started_at
                print(
                    f"sampling_progress phase=batch batch={batch_idx}/{total_batches} "
                    f"graphs_in_batch={batch.num_graphs} elapsed_s={elapsed_s:.1f}",
                    flush=True,
                )

            for _ in range(samples_per_target):
                pos_t, _v_t, l_t, h_t = self._sample_batch(batch)

                ptr = batch.ptr.tolist()
                for graph_idx, (start_idx, end_idx) in enumerate(zip(ptr[:-1], ptr[1:])):
                    per_graph[graph_idx].append(
                        evaluate_csp_reconstruction(
                            pred_f=pos_t[start_idx:end_idx],
                            pred_l=l_t[graph_idx],
                            pred_a=h_t[start_idx:end_idx],
                            target_f=batch.pos[start_idx:end_idx],
                            target_l=batch.l[graph_idx],
                            target_a=batch.atomic_numbers[start_idx:end_idx],
                            lattice_transform=self.lattice_transform,
                            requested_space_group=int(torch.as_tensor(batch.space_group).reshape(-1)[graph_idx].item()),
                            validity_cutoff=self.validity_cutoff,
                        )
                    )

            results.extend(per_graph)

        total_elapsed_s = time.perf_counter() - started_at
        if seed is not None:
            print(
                f"eval_seed={seed} sampling_pass_done targets={len(results)} "
                f"elapsed_s={total_elapsed_s:.1f}",
                flush=True,
            )
        else:
            print(
                f"sampling_progress phase=done targets={len(results)} elapsed_s={total_elapsed_s:.1f}",
                flush=True,
            )
        return results

    def _progress_paths(self, run_id: str) -> tuple[Path, Path]:
        self.eval_progress_dir.mkdir(parents=True, exist_ok=True)
        return (
            self.eval_progress_dir / f"{run_id}.json",
            self.eval_progress_dir / f"{run_id}.log",
        )

    def _new_eval_state(self, run_id: str) -> dict[str, Any]:
        target_count = len(self.loader.dataset)
        return {
            "version": 1,
            "run_id": run_id,
            "experiment_name": self.experiment_name,
            "config_path": str(self.config_path),
            "checkpoint_path": str(self.checkpoint_path),
            "num_targets": target_count,
            "subset_seed": int(self.eval_cfg["subset_seed"]),
            "n_seeds": self.eval_seed_count,
            "from_seed": self.eval_from_seed,
            "completed_seeds": [],
            "seed_summaries": [],
            "target_valid_count": [0 for _ in range(target_count)],
            "target_hit_count": [0 for _ in range(target_count)],
            "target_best_rmse": [None for _ in range(target_count)],
            "target_relaxed_hit_count": [0 for _ in range(target_count)],
            "target_best_relaxed_rmse": [None for _ in range(target_count)],
            "target_best_metrics": [None for _ in range(target_count)],
            "first_seed_matches": None,
        }

    def _save_eval_state(self, state: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)

    def _load_eval_state(self, run_id: str) -> tuple[dict[str, Any], Path, Path]:
        state_path, log_path = self._progress_paths(run_id)
        if state_path.exists():
            with state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
            print(
                f"evaluation_resume_state path={state_path} completed_seeds={state.get('completed_seeds', [])}",
                flush=True,
            )
            return state, state_path, log_path

        state = self._new_eval_state(run_id)
        self._save_eval_state(state, state_path)
        return state, state_path, log_path

    def _append_eval_log(self, path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip() + "\n")

    def _update_eval_state(
        self,
        state: dict[str, Any],
        *,
        seed: int,
        results: list[Any],
        summary: dict[str, Any],
    ) -> None:
        completed = {int(value) for value in state.get("completed_seeds", [])}
        if seed in completed:
            return

        target_valid = list(state.get("target_valid_count") or [0 for _ in range(len(results))])
        target_hits = list(state.get("target_hit_count", []))
        target_best_rmse = list(state.get("target_best_rmse", []))
        target_relaxed_hits = list(state.get("target_relaxed_hit_count") or [0 for _ in range(len(results))])
        target_best_relaxed_rmse = list(state.get("target_best_relaxed_rmse") or [None for _ in range(len(results))])
        target_best_metrics = list(state.get("target_best_metrics") or [None for _ in range(len(results))])
        if (
            len(target_valid) != len(results)
            or len(target_relaxed_hits) != len(results)
            or len(target_best_relaxed_rmse) != len(results)
            or len(target_hits) != len(results)
            or len(target_best_rmse) != len(results)
            or len(target_best_metrics) != len(results)
        ):
            raise ValueError(
                "Evaluation state target count mismatch "
                f"state_valid={len(target_valid)} state_hits={len(target_hits)} "
                f"state_rmse={len(target_best_rmse)} state_relaxed_hits={len(target_relaxed_hits)} "
                f"state_relaxed_rmse={len(target_best_relaxed_rmse)} "
                f"state_metrics={len(target_best_metrics)} results={len(results)}"
            )

        for target_idx, result in enumerate(results):
            if result.valid:
                target_valid[target_idx] = int(target_valid[target_idx]) + 1
            if getattr(result, "relaxed_match", False) and getattr(result, "relaxed_rmse", None) is not None:
                target_relaxed_hits[target_idx] = int(target_relaxed_hits[target_idx]) + 1
                current_relaxed_rmse = float(result.relaxed_rmse)
                best_relaxed_rmse = target_best_relaxed_rmse[target_idx]
                target_best_relaxed_rmse[target_idx] = (
                    current_relaxed_rmse
                    if best_relaxed_rmse is None
                    else min(float(best_relaxed_rmse), current_relaxed_rmse)
                )
            if not result.match or result.rmse is None:
                continue
            target_hits[target_idx] = int(target_hits[target_idx]) + 1
            best_rmse = target_best_rmse[target_idx]
            current_rmse = float(result.rmse)
            if best_rmse is None or current_rmse <= float(best_rmse):
                target_best_rmse[target_idx] = current_rmse
                target_best_metrics[target_idx] = _result_metric_payload(result)

        if state.get("first_seed_matches") is None:
            state["first_seed_matches"] = [int(result.match) for result in results]

        state["target_valid_count"] = target_valid
        state["target_hit_count"] = target_hits
        state["target_best_rmse"] = target_best_rmse
        state["target_relaxed_hit_count"] = target_relaxed_hits
        state["target_best_relaxed_rmse"] = target_best_relaxed_rmse
        state["target_best_metrics"] = target_best_metrics
        state["completed_seeds"] = sorted([*completed, seed])
        seed_summaries = [item for item in state.get("seed_summaries", []) if int(item["seed"]) != seed]
        seed_summary = {
            "seed": seed,
            "valid": _json_float(summary.get("valid")),
            "match_rate": _json_float(summary.get("match_rate")),
            "rmse": _json_float(summary.get("rmse")),
            "composition_match_rate": _json_float(summary.get("composition_match_rate")),
        }
        for source_key, _display_key in EVAL_EXTRA_METRICS:
            seed_summary[source_key] = _json_float(summary.get(source_key))
        for source_key, _display_key in EVAL_RELAXED_METRICS:
            seed_summary[source_key] = _json_float(summary.get(source_key))
        seed_summaries.append(seed_summary)
        state["seed_summaries"] = sorted(seed_summaries, key=lambda item: int(item["seed"]))

    def _summary_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        seed_summaries = list(state.get("seed_summaries", []))
        valid_values = [float(item["valid"]) for item in seed_summaries if item.get("valid") is not None]
        match_values = [float(item["match_rate"]) for item in seed_summaries if item.get("match_rate") is not None]
        rmse_values = [float(item["rmse"]) for item in seed_summaries if item.get("rmse") is not None]
        target_valid = np.asarray(state.get("target_valid_count") or [], dtype=int)
        target_hits = np.asarray(state.get("target_hit_count", []), dtype=int)
        target_relaxed_hits = np.asarray(state.get("target_relaxed_hit_count") or [], dtype=int)
        reached = target_hits > 0
        valid_reached = target_valid > 0
        relaxed_reached = target_relaxed_hits > 0
        best_rmse_values = [float(value) for value in state.get("target_best_rmse", []) if value is not None]
        best_relaxed_rmse_values = [
            float(value) for value in state.get("target_best_relaxed_rmse", []) if value is not None
        ]
        completed_seed_count = len(state.get("completed_seeds", []))
        match_frequencies = (
            target_hits.astype(float) / float(completed_seed_count)
            if completed_seed_count > 0 and target_hits.size > 0
            else np.asarray([], dtype=float)
        )

        at_1_summary = {
            "valid_mean": None if not valid_values else float(np.mean(valid_values)),
            "valid_std": None if not valid_values else float(np.std(valid_values)),
            "match_rate_mean": None if not match_values else float(np.mean(match_values)),
            "match_rate_std": None if not match_values else float(np.std(match_values)),
            "rmse_mean": None if not rmse_values else float(np.mean(rmse_values)),
            "rmse_std": None if not rmse_values else float(np.std(rmse_values)),
        }
        at_k_summary = {
            "valid_rate": None if target_valid.size == 0 else float(np.mean(valid_reached)),
            "match_rate": None if target_hits.size == 0 else float(np.mean(reached)),
            "rmse": None if not best_rmse_values else float(np.mean(best_rmse_values)),
            "relaxed_match_rate": None if target_relaxed_hits.size == 0 else float(np.mean(relaxed_reached)),
            "relaxed_rmse": (
                None if not best_relaxed_rmse_values else float(np.mean(best_relaxed_rmse_values))
            ),
            "material_stability_mean": None if match_frequencies.size == 0 else float(np.mean(match_frequencies)),
            "material_stability_std": None if match_frequencies.size == 0 else float(np.std(match_frequencies)),
            "matched_material_stability_mean": (
                None if not reached.any() else float(np.mean(match_frequencies[reached]))
            ),
        }
        if at_k_summary["relaxed_match_rate"] is None or at_k_summary["match_rate"] is None:
            at_k_summary["relaxed_near_miss_rate"] = None
        else:
            at_k_summary["relaxed_near_miss_rate"] = (
                float(at_k_summary["relaxed_match_rate"]) - float(at_k_summary["match_rate"])
            )
        target_best_metrics = [item for item in state.get("target_best_metrics", []) if isinstance(item, dict)]
        for source_key, display_key in EVAL_EXTRA_METRICS:
            seed_values = [float(item[source_key]) for item in seed_summaries if item.get(source_key) is not None]
            mean_value, std_value = _mean_std(seed_values)
            at_1_summary[f"{display_key}_mean"] = mean_value
            at_1_summary[f"{display_key}_std"] = std_value

            best_values = [
                float(item[source_key])
                for item in target_best_metrics
                if item.get(source_key) is not None
            ]
            best_mean, best_std = _mean_std(best_values)
            at_k_summary[f"{display_key}_mean"] = best_mean
            at_k_summary[f"{display_key}_std"] = best_std
        for source_key, display_key in EVAL_RELAXED_METRICS:
            seed_values = [float(item[source_key]) for item in seed_summaries if item.get(source_key) is not None]
            mean_value, std_value = _mean_std(seed_values)
            at_1_summary[f"relaxed_{display_key}_mean"] = mean_value
            at_1_summary[f"relaxed_{display_key}_std"] = std_value

        return {
            "completed_seeds": [int(seed) for seed in state.get("completed_seeds", [])],
            "at_1_summary": at_1_summary,
            "at_k_summary": at_k_summary,
            "at_1_matches": list(state.get("first_seed_matches") or []),
            "at_k_matches": reached.astype(int).tolist(),
            "at_1_rmses": rmse_values,
            "at_k_rmses": best_rmse_values,
        }

    # Runs repeated single-sample passes and aggregates them into @1/@20 summaries.
    def _evaluate(self, run) -> dict[str, Any]:
        from kldmPlus.sample_evaluation import aggregate_csp_reconstruction_metrics

        state, state_path, log_path = self._load_eval_state(run.id)
        completed = {int(seed) for seed in state.get("completed_seeds", [])}

        for seed in range(self.eval_from_seed, self.eval_seed_count):
            if seed in completed:
                print(f"evaluation_skip seed={seed} reason=already_completed", flush=True)
                continue
            print(f"evaluation_progress seed={seed + 1}/{self.eval_seed_count}", flush=True)
            results = [graph_results[0] for graph_results in self._collect(samples_per_target=1, seed=seed)]
            summary = aggregate_csp_reconstruction_metrics(results)
            print(
                f"evaluation_seed_summary seed={seed} "
                f"valid={format_metric(summary['valid'], '.4f')} "
                f"match_rate={format_metric(summary['match_rate'], '.4f')} "
                f"rmse={format_metric(summary['rmse'], '.6f')} "
                f"composition_match_rate={format_metric(summary.get('composition_match_rate'), '.4f')} "
                f"requested_sg_match_rate={format_metric(summary.get('requested_space_group_match_rate'), '.4f')} "
                f"family_agreement={format_metric(summary.get('detected_family_agreement'), '.4f')} "
                f"frac_rmse={format_metric(summary.get('frac_rmse'), '.6f')} "
                f"length_rmse={format_metric(summary.get('lattice_lengths_rmse'), '.6f')} "
                f"angles_rmse={format_metric(summary.get('lattice_angles_rmse'), '.6f')} "
                f"relaxed_match_rate={format_metric(summary.get('relaxed_match_rate'), '.4f')} "
                f"relaxed_rmse={format_metric(summary.get('relaxed_rmse'), '.6f')} "
                f"relaxed_near_miss_rate={format_metric(summary.get('relaxed_near_miss_rate'), '.4f')}",
                flush=True,
            )
            self._update_eval_state(state, seed=seed, results=results, summary=summary)
            self._save_eval_state(state, state_path)
            self._append_eval_log(
                log_path,
                (
                    f"seed={seed} valid={format_metric(summary['valid'], '.4f')} "
                    f"match_rate={format_metric(summary['match_rate'], '.4f')} "
                    f"rmse={format_metric(summary['rmse'], '.6f')} "
                    f"relaxed_match_rate={format_metric(summary.get('relaxed_match_rate'), '.4f')} "
                    f"relaxed_rmse={format_metric(summary.get('relaxed_rmse'), '.6f')}"
                ),
            )
            wandb.log(
                {
                    "evaluation/latest_seed": seed,
                    "evaluation/completed_seed_count": len(state["completed_seeds"]),
                    "evaluation/latest_seed_valid": summary["valid"],
                    "evaluation/latest_seed_match_rate": summary["match_rate"],
                    "evaluation/latest_seed_rmse": summary["rmse"],
                    "evaluation/latest_seed_space_group_agreement": summary.get("requested_space_group_match_rate"),
                    "evaluation/latest_seed_family_agreement": summary.get("detected_family_agreement"),
                    "evaluation/latest_seed_frac_rmse": summary.get("frac_rmse"),
                    "evaluation/latest_seed_length_rmse": summary.get("lattice_lengths_rmse"),
                    "evaluation/latest_seed_angles_rmse": summary.get("lattice_angles_rmse"),
                    "evaluation/latest_seed_relaxed_match_rate": summary.get("relaxed_match_rate"),
                    "evaluation/latest_seed_relaxed_rmse": summary.get("relaxed_rmse"),
                    "evaluation/latest_seed_relaxed_near_miss_rate": summary.get("relaxed_near_miss_rate"),
                }
            )
            completed = {int(value) for value in state.get("completed_seeds", [])}
            if should_stop(run):
                print(
                    f"evaluation_stop_requested completed_seeds={state['completed_seeds']} state_path={state_path}",
                    flush=True,
                )
                break

        summary = self._summary_from_state(state)
        summary["state_path"] = str(state_path)
        summary["log_path"] = str(log_path)
        summary["complete"] = len(summary["completed_seeds"]) >= self.eval_seed_count
        return summary

    # Creates a compact artifact with a few representative evaluation structures.
    def _log_eval_examples(self, summary: dict[str, Any], temp_dir: Path) -> None:
        artifact = wandb.Artifact(f"structures_{self.experiment_name}", type="structure")

        for prefix, results in (("at1", summary["at_1_results"]), ("at20", summary["at_k_results"])):
            for index, result in enumerate(results[:3], start=1):
                if result.predicted_structure is None or result.target_structure is None:
                    continue
                predicted_vis, target_vis = prepare_visualization_pair(
                    result.predicted_structure,
                    result.target_structure,
                )
                pred_path = temp_dir / f"{prefix}_{index:02d}_predicted.cif"
                target_path = temp_dir / f"{prefix}_{index:02d}_actual.cif"
                png_path = temp_dir / f"{prefix}_{index:02d}.png"
                predicted_vis.to(fmt="cif", filename=str(pred_path))
                target_vis.to(fmt="cif", filename=str(target_path))
                self._render_pair(predicted_vis, target_vis, png_path)
                artifact.add_file(str(pred_path), name=pred_path.name)
                artifact.add_file(str(target_path), name=target_path.name)
                wandb.log({f"structures/{prefix}_{index:02d}": wandb.Image(str(png_path))})

        if artifact.manifest.entries:
            wandb.log_artifact(artifact)

    # Builds a stable material label used in sample mode outputs.
    @staticmethod
    def _material_name(index: int, result) -> str:
        rmse = "na" if result.rmse is None else f"{float(result.rmse):.6f}"
        return f"material_{index:02d}_rmse_{rmse}_match_{int(result.match)}_valid_{int(result.valid)}"

    # Logs individual sample-mode materials and returns a summary payload.
    def _log_samples(self, results: list[Any], temp_dir: Path) -> dict[str, Any]:
        from kldmPlus.sample_evaluation import aggregate_csp_reconstruction_metrics

        artifact = wandb.Artifact(f"materials_{self.experiment_name}", type="structure")
        table = wandb.Table(columns=["material", "rmse", "match", "valid"])
        rows = []

        for index, result in enumerate(results, start=1):
            if result.predicted_structure is None or result.target_structure is None:
                continue

            name = self._material_name(index, result)
            predicted_vis, target_vis = prepare_visualization_pair(
                result.predicted_structure,
                result.target_structure,
            )
            pred_path = temp_dir / f"{name}_predicted.cif"
            target_path = temp_dir / f"{name}_actual.cif"
            png_path = temp_dir / f"{name}.png"
            predicted_vis.to(fmt="cif", filename=str(pred_path))
            target_vis.to(fmt="cif", filename=str(target_path))
            self._render_pair(predicted_vis, target_vis, png_path)
            artifact.add_file(str(pred_path), name=pred_path.name)
            artifact.add_file(str(target_path), name=target_path.name)
            wandb.log({f"materials/{name}": wandb.Image(str(png_path))})
            table.add_data(name, None if result.rmse is None else float(result.rmse), int(result.match), int(result.valid))
            rows.append({"material": name, "rmse": result.rmse, "match": result.match, "valid": result.valid})

        if len(table.data) > 0:
            wandb.log({"materials/summary": table})
        if artifact.manifest.entries:
            wandb.log_artifact(artifact)

        return {
            "materials": rows,
            "reconstruction_summary": aggregate_csp_reconstruction_metrics(results),
        }

    # Runs either sample-mode export or repeated-pass @1/@20 evaluation.
    def run(self) -> None:
        init_kwargs = {
            "project": self.wandb_project,
            "name": self.wandb_run_name,
            "config": {
                "experiment_name": self.experiment_name,
                "config_path": str(self.config_path),
                "checkpoint_path": str(self.checkpoint_path),
                "evaluation": self.evaluation,
                "n_samples": self.sample_count,
                "sampling": self.sampling_cfg,
                "sampling_eval": self.eval_cfg,
            },
        }
        if self.wandb_resume_id:
            init_kwargs["id"] = str(self.wandb_resume_id)
            init_kwargs["resume"] = "must"
        run = wandb.init(**init_kwargs)
        print(f"data_split sample={TEST_SPLIT}", flush=True)

        with tempfile.TemporaryDirectory(prefix="kldm_sampling_") as tmp:
            temp_dir = Path(tmp)
            if not self.evaluation:
                material_results = [graph_results[0] for graph_results in self._collect(samples_per_target=1)]
                summary = self._log_samples(material_results, temp_dir)
                print(f"saved {len(material_results)} materials to wandb", flush=True)
                for material in summary["materials"]:
                    print(
                        f"{material['material']} "
                        f"rmse={format_metric(material['rmse'], '.6f')} "
                        f"match={int(material['match'])} "
                        f"valid={int(material['valid'])}",
                        flush=True,
                    )
                recon = summary["reconstruction_summary"]
                print(
                    f"samples valid={format_metric(recon['valid'], '.4f')} "
                    f"match_rate={format_metric(recon['match_rate'], '.4f')} "
                    f"rmse={format_metric(recon['rmse'], '.6f')} "
                    f"composition_match_rate={format_metric(recon.get('composition_match_rate'), '.4f')} "
                    f"requested_sg_match_rate={format_metric(recon.get('requested_space_group_match_rate'), '.4f')}",
                    flush=True,
                )
            else:
                summary = self._evaluate(run)
                at_1 = summary["at_1_summary"]
                at_k = summary["at_k_summary"]
                log_data = {
                    "facit_core/@1_valid_mean": at_1["valid_mean"],
                    "facit_core/@1_valid_std": at_1["valid_std"],
                    "facit_core/@1_match_rate_mean": at_1["match_rate_mean"],
                    "facit_core/@1_match_rate_std": at_1["match_rate_std"],
                    "facit_core/@1_rmse_mean": at_1["rmse_mean"],
                    "facit_core/@1_rmse_std": at_1["rmse_std"],
                    "facit_core/@20_valid_rate": at_k["valid_rate"],
                    "facit_core/@20_match_rate": at_k["match_rate"],
                    "facit_core/@20_rmse": at_k["rmse"],
                    "@1/valid_mean": at_1["valid_mean"],
                    "@1/valid_std": at_1["valid_std"],
                    "@1/match_rate_mean": at_1["match_rate_mean"],
                    "@1/match_rate_std": at_1["match_rate_std"],
                    "@1/rmse_mean": at_1["rmse_mean"],
                    "@1/rmse_std": at_1["rmse_std"],
                    "histograms/@1_match": wandb.Histogram(summary["at_1_matches"]),
                    "@20/valid_rate": at_k["valid_rate"],
                    "@20/match_rate": at_k["match_rate"],
                    "@20/rmse": at_k["rmse"],
                    "relaxed_gt/@1_match_rate_mean": at_1.get("relaxed_match_rate_mean"),
                    "relaxed_gt/@1_match_rate_std": at_1.get("relaxed_match_rate_std"),
                    "relaxed_gt/@1_rmse_mean": at_1.get("relaxed_rmse_mean"),
                    "relaxed_gt/@1_rmse_std": at_1.get("relaxed_rmse_std"),
                    "relaxed_gt/@1_near_miss_rate_mean": at_1.get("relaxed_near_miss_rate_mean"),
                    "relaxed_gt/@1_near_miss_rate_std": at_1.get("relaxed_near_miss_rate_std"),
                    "relaxed_gt/@20_match_rate": at_k.get("relaxed_match_rate"),
                    "relaxed_gt/@20_rmse": at_k.get("relaxed_rmse"),
                    "relaxed_gt/@20_near_miss_rate": at_k.get("relaxed_near_miss_rate"),
                    "stability/@20_material_stability_mean": at_k["material_stability_mean"],
                    "stability/@20_material_stability_std": at_k["material_stability_std"],
                    "stability/@20_matched_material_stability_mean": at_k["matched_material_stability_mean"],
                    "histograms/@20_match": wandb.Histogram(summary["at_k_matches"]),
                }
                for _source_key, display_key in EVAL_EXTRA_METRICS:
                    log_data[f"diagnostics/@1_{display_key}_mean"] = at_1.get(f"{display_key}_mean")
                    log_data[f"diagnostics/@1_{display_key}_std"] = at_1.get(f"{display_key}_std")
                    log_data[f"diagnostics/@20_{display_key}_mean"] = at_k.get(f"{display_key}_mean")
                    log_data[f"diagnostics/@20_{display_key}_std"] = at_k.get(f"{display_key}_std")
                if summary["at_1_rmses"]:
                    log_data["histograms/@1_rmse"] = wandb.Histogram(summary["at_1_rmses"])
                if summary["at_k_rmses"]:
                    log_data["histograms/@20_rmse"] = wandb.Histogram(summary["at_k_rmses"])
                log_data["evaluation/completed_seed_count"] = len(summary["completed_seeds"])
                log_data["evaluation/completed_seeds"] = ",".join(str(seed) for seed in summary["completed_seeds"])
                log_data["evaluation/complete"] = int(bool(summary["complete"]))
                wandb.log(log_data)
                print(
                    f"@1 valid_mean={format_metric(at_1['valid_mean'], '.4f')} "
                    f"valid_std={format_metric(at_1['valid_std'], '.4f')} "
                    f"match_rate_mean={format_metric(at_1['match_rate_mean'], '.4f')} "
                    f"match_rate_std={format_metric(at_1['match_rate_std'], '.4f')} "
                    f"rmse_mean={format_metric(at_1['rmse_mean'], '.6f')} "
                    f"rmse_std={format_metric(at_1['rmse_std'], '.6f')} "
                    f"space_group_agreement={format_metric(at_1.get('space_group_agreement_mean'), '.4f')} "
                    f"family_agreement={format_metric(at_1.get('family_agreement_mean'), '.4f')} "
                    f"frac_rmse={format_metric(at_1.get('frac_rmse_mean'), '.6f')} "
                    f"length_rmse={format_metric(at_1.get('length_rmse_mean'), '.6f')} "
                    f"angles_rmse={format_metric(at_1.get('angles_rmse_mean'), '.6f')} "
                    f"relaxed_match_rate={format_metric(at_1.get('relaxed_match_rate_mean'), '.4f')} "
                    f"relaxed_rmse={format_metric(at_1.get('relaxed_rmse_mean'), '.6f')} "
                    f"relaxed_near_miss_rate={format_metric(at_1.get('relaxed_near_miss_rate_mean'), '.4f')}",
                    flush=True,
                )
                print(
                    f"@20 valid_rate={format_metric(at_k.get('valid_rate'), '.4f')} "
                    f"match_rate={format_metric(at_k['match_rate'], '.4f')} "
                    f"rmse={format_metric(at_k['rmse'], '.6f')} "
                    f"space_group_agreement={format_metric(at_k.get('space_group_agreement_mean'), '.4f')} "
                    f"family_agreement={format_metric(at_k.get('family_agreement_mean'), '.4f')} "
                    f"frac_rmse={format_metric(at_k.get('frac_rmse_mean'), '.6f')} "
                    f"length_rmse={format_metric(at_k.get('length_rmse_mean'), '.6f')} "
                    f"angles_rmse={format_metric(at_k.get('angles_rmse_mean'), '.6f')} "
                    f"relaxed_match_rate={format_metric(at_k.get('relaxed_match_rate'), '.4f')} "
                    f"relaxed_rmse={format_metric(at_k.get('relaxed_rmse'), '.6f')} "
                    f"relaxed_near_miss_rate={format_metric(at_k.get('relaxed_near_miss_rate'), '.4f')} "
                    f"material_stability={format_metric(at_k.get('material_stability_mean'), '.4f')} "
                    f"completed_seeds={summary['completed_seeds']} "
                    f"state_path={summary['state_path']} "
                    f"log_path={summary['log_path']}",
                    flush=True,
                )

        wandb.finish()


def main() -> None:
    SamplingRunner(parse_args().config).run()


if __name__ == "__main__":
    main()
