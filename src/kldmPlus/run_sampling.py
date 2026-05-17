from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import pickle
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
        self.template_prior = None
        self._template_prior_initialized = False
        self._inject_mattergen_lattice_stats()
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

    def _ensure_template_prior(self):
        if not self._template_prior_initialized:
            self.template_prior = self._build_template_prior()
            self._template_prior_initialized = True
        return self.template_prior

    def _build_template_prior(self):
        from kldmPlus.data import CSPTask, resolve_data_root
        from kldmPlus.symmetry import build_dataset_template_prior
        from kldmPlus.symmetry.template_prior import _anonymous_count_key
        from kldmPlus.symmetry.wyckoff_templates import requested_composition_key

        sampling_algorithm = int(self.sampling_cfg.get("sampling_algorithm", 4 if str(self.sampling_cfg["method"]) == "pc" else 3))
        if sampling_algorithm not in {6, 7, 8}:
            return None
        if sampling_algorithm == 6:
            prior_cfg = dict(self.sampling_cfg.get("pcs", {}))
        elif sampling_algorithm == 7:
            prior_cfg = dict(self.sampling_cfg.get("sgdpnp", {}))
        else:
            prior_cfg = dict(self.sampling_cfg.get("dpnpsvd", {}))
        if not bool(prior_cfg.get("template_prior_enabled", True)):
            return None
        if float(prior_cfg.get("template_prior_weight", 1.0)) <= 0.0:
            return None

        dataset_cfg = dict(self.config["dataset"])
        model_cfg = dict(self.config["model"])
        root = resolve_data_root(dataset_cfg["root"])
        task = CSPTask(
            dataset_name=str(dataset_cfg["name"]),
            lattice_parameterization=str(model_cfg["lattice_parameterization"]),
            lattice_representation=str(dataset_cfg.get("lattice_representation", "kldm")),
        )
        train_dataset = task.fit_dataset(root=root, split="train", download=True)
        max_samples = int(prior_cfg.get("template_prior_max_samples", 2000))
        if max_samples <= 0:
            return None
        match_targets_only = bool(prior_cfg.get("template_prior_match_targets_only", True))
        allowed_keys = None
        if match_targets_only:
            allowed_keys = {
                requested_composition_key(
                    space_group_number=int(torch.as_tensor(sample.space_group).reshape(-1)[0].item()),
                    atomic_numbers=sample.atomic_numbers,
                )
                for sample in self.loader.dataset
            }
        cache_path = None
        if bool(prior_cfg.get("template_prior_cache", True)):
            cache_dir = Path.cwd() / ".cache" / "kldmPlus" / "template_prior"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_payload = {
                "dataset": str(dataset_cfg["name"]),
                "lattice_representation": str(dataset_cfg.get("lattice_representation", "kldm")),
                "lattice_parameterization": str(model_cfg["lattice_parameterization"]),
                "max_samples": int(max_samples),
                "allowed_keys": sorted(map(repr, allowed_keys or [])),
            }
            cache_hash = hashlib.sha1(repr(cache_payload).encode("utf-8")).hexdigest()[:16]
            cache_path = cache_dir / f"{cache_hash}.pkl"
            if cache_path.exists():
                with cache_path.open("rb") as handle:
                    prior = pickle.load(handle)
                print(
                    f"template_prior_cache_hit path={cache_path} records={len(prior)}",
                    flush=True,
                )
                return prior
        allowed_render = 0 if allowed_keys is None else len(allowed_keys)
        anonymous_render = 0 if allowed_keys is None else len({_anonymous_count_key(key) for key in allowed_keys})
        print(
            f"template_prior_build start max_samples={max_samples} "
            f"match_targets_only={int(match_targets_only)} allowed_keys={allowed_render} "
            f"allowed_anonymous_keys={anonymous_render}",
            flush=True,
        )
        started_at = time.perf_counter()
        prior = build_dataset_template_prior(
            dataset=train_dataset,
            lattice_transform=self.lattice_transform,
            max_samples=max_samples,
            allowed_keys=allowed_keys,
        )
        elapsed_s = time.perf_counter() - started_at
        print(
            f"template_prior_build done records={len(prior)} elapsed_s={elapsed_s:.1f}",
            flush=True,
        )
        if cache_path is not None:
            with cache_path.open("wb") as handle:
                pickle.dump(prior, handle)
            print(f"template_prior_cache_write path={cache_path}", flush=True)
        return prior

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
        if sampling_algorithm == 8:
            if method != "em":
                raise ValueError(
                    "sampling_algorithm=8 currently extends the EM sampler, so sampling.method must be 'em'.",
                )
            kwargs["lattice_transform"] = self.lattice_transform
            kwargs["dpnpsvd_config"] = dict(self.sampling_cfg.get("dpnpsvd", {}))
            kwargs["template_prior"] = self._ensure_template_prior()
            sample_fn = self.model.sample_CSP_algorithm8
        elif sampling_algorithm == 7:
            if method != "em":
                raise ValueError(
                    "sampling_algorithm=7 currently extends the EM sampler, so sampling.method must be 'em'.",
                )
            kwargs["lattice_transform"] = self.lattice_transform
            kwargs["sgdpnp_config"] = dict(self.sampling_cfg.get("sgdpnp", {}))
            kwargs["template_prior"] = self._ensure_template_prior()
            sample_fn = self.model.sample_CSP_algorithm7
        elif sampling_algorithm == 6:
            if method != "em":
                raise ValueError(
                    "sampling_algorithm=6 currently extends the EM sampler, so sampling.method must be 'em'.",
                )
            pcs_cfg = dict(self.sampling_cfg.get("pcs", {}))
            kwargs["lattice_transform"] = self.lattice_transform
            kwargs["pcs_standardization"] = str(pcs_cfg.get("standardization", "conventional"))
            kwargs["pcs_symprec"] = float(pcs_cfg.get("symprec", 1e-2))
            kwargs["pcs_angle_tolerance"] = float(pcs_cfg.get("angle_tolerance", 5.0))
            kwargs["pcs_max_templates"] = int(pcs_cfg.get("max_templates", 256))
            kwargs["pcs_template_eval_limit"] = int(pcs_cfg.get("template_eval_limit", 32))
            kwargs["pcs_optimization_steps"] = int(pcs_cfg.get("optimization_steps", 150))
            kwargs["pcs_learning_rate"] = float(pcs_cfg.get("learning_rate", 5e-2))
            kwargs["pcs_coord_weight"] = float(pcs_cfg.get("coord_weight", 1.0))
            kwargs["pcs_lattice_weight"] = float(pcs_cfg.get("lattice_weight", 0.25))
            kwargs["pcs_pairdist_weight"] = float(pcs_cfg.get("pairdist_weight", 0.0))
            kwargs["pcs_template_init_pairdist_weight"] = (
                None
                if "template_init_pairdist_weight" not in pcs_cfg
                else float(pcs_cfg["template_init_pairdist_weight"])
            )
            kwargs["pcs_pairdist_bins"] = int(pcs_cfg.get("pairdist_bins", 32))
            kwargs["pcs_pairdist_max_distance"] = float(pcs_cfg.get("pairdist_max_distance", 8.0))
            kwargs["pcs_pairdist_bandwidth"] = float(pcs_cfg.get("pairdist_bandwidth", 0.25))
            kwargs["pcs_steric_weight"] = float(pcs_cfg.get("steric_weight", 0.0))
            kwargs["pcs_steric_min_distance"] = float(pcs_cfg.get("steric_min_distance", 0.8))
            kwargs["pcs_volume_weight"] = float(pcs_cfg.get("volume_weight", 0.0))
            kwargs["pcs_volume_ratio_min"] = float(pcs_cfg.get("volume_ratio_min", 0.0))
            kwargs["pcs_volume_ratio_max"] = float(pcs_cfg.get("volume_ratio_max", 0.0))
            kwargs["pcs_k6_weight"] = float(pcs_cfg.get("k6_weight", 0.0))
            kwargs["pcs_hard_min_distance"] = float(pcs_cfg.get("hard_min_distance", 0.0))
            kwargs["pcs_hard_volume_ratio_min"] = float(pcs_cfg.get("hard_volume_ratio_min", 0.0))
            kwargs["pcs_hard_volume_ratio_max"] = float(pcs_cfg.get("hard_volume_ratio_max", 0.0))
            kwargs["pcs_freeze_lattice"] = bool(pcs_cfg.get("freeze_lattice", False))
            kwargs["pcs_initialization"] = str(pcs_cfg.get("initialization", "repair"))
            kwargs["pcs_quick_templates"] = bool(pcs_cfg.get("quick_templates", False))
            kwargs["pcs_top_k_templates"] = int(pcs_cfg.get("top_k_templates", 1))
            kwargs["pcs_mala_steps"] = int(pcs_cfg.get("mala_steps", 8))
            kwargs["pcs_mala_step_size"] = float(pcs_cfg.get("mala_step_size", 5e-2))
            kwargs["pcs_debug_template_candidates"] = bool(pcs_cfg.get("debug_template_candidates", False))
            if "pcs_debug_high_prior_templates" in inspect.signature(
                self.model.sample_CSP_algorithm6
            ).parameters:
                kwargs["pcs_debug_high_prior_templates"] = bool(
                    pcs_cfg.get("debug_high_prior_templates", False)
                )
                kwargs["pcs_debug_high_prior_min_score"] = int(
                    pcs_cfg.get("debug_high_prior_min_score", 1)
                )
            if "pcs_allow_soft_physics_fallback" in inspect.signature(
                self.model.sample_CSP_algorithm6
            ).parameters:
                kwargs["pcs_allow_soft_physics_fallback"] = bool(
                    pcs_cfg.get("allow_soft_physics_fallback", True)
                )
            if "pcs_branch_selection_temperature" in inspect.signature(
                self.model.sample_CSP_algorithm6
            ).parameters:
                kwargs["pcs_branch_selection_temperature"] = float(
                    pcs_cfg.get("branch_selection_temperature", 1.0)
                )
            kwargs["pcs_oracle_template_orbit_rerank"] = bool(
                pcs_cfg.get("oracle_template_orbit_rerank", False)
            )
            kwargs["pcs_oracle_template_fit_target"] = bool(
                pcs_cfg.get("oracle_template_fit_target", False)
            )
            kwargs["pcs_dds_repair"] = bool(pcs_cfg.get("dds_repair", True))
            kwargs["pcs_dds_n_steps"] = int(pcs_cfg.get("dds_n_steps", 60))
            kwargs["pcs_dds_t_final"] = float(pcs_cfg.get("dds_t_final", kwargs["t_final"]))
            kwargs["pcs_outer_steps"] = int(pcs_cfg.get("outer_steps", 1))
            kwargs["pcs_outer_eta_start"] = float(pcs_cfg.get("outer_eta_start", pcs_cfg.get("dds_t_start", 0.2)))
            kwargs["pcs_outer_eta_end"] = float(pcs_cfg.get("outer_eta_end", pcs_cfg.get("dds_t_start", 0.2)))
            kwargs["pcs_outer_eta_k_start"] = int(pcs_cfg.get("outer_eta_k_start", 0))
            kwargs["pcs_outer_eta_rho"] = float(pcs_cfg.get("outer_eta_rho", 1.0))
            kwargs["pcs_final_projection"] = bool(pcs_cfg.get("final_projection", True))
            kwargs["pcs_validate_requested_space_group"] = bool(pcs_cfg.get("validate_requested_space_group", True))
            kwargs["pcs_return_last_pcs_on_validation_failure"] = bool(
                pcs_cfg.get("return_last_pcs_on_validation_failure", False)
            )
            kwargs["pcs_template_prior"] = self._ensure_template_prior()
            kwargs["pcs_template_prior_weight"] = float(pcs_cfg.get("template_prior_weight", 1.0))
            sample_fn = self.model.sample_CSP_algorithm6
        elif sampling_algorithm == 5:
            if method != "em":
                raise ValueError(
                    "sampling_algorithm=5 currently extends the EM sampler, so sampling.method must be 'em'.",
                )
            guidance_cfg = dict(self.sampling_cfg.get("symmetry_guidance", {}))
            kwargs["lattice_transform"] = self.lattice_transform
            kwargs["coord_scale"] = float(guidance_cfg.get("coord_scale", 2e-3))
            kwargs["lattice_scale"] = float(guidance_cfg.get("lattice_scale", 1e-5))
            kwargs["guidance_interval"] = int(guidance_cfg.get("guidance_interval", 5))
            kwargs["guidance_start_fraction"] = float(guidance_cfg.get("guidance_start_fraction", 0.5))
            kwargs["coord_grad_clip"] = guidance_cfg.get("coord_grad_clip", 5.0)
            kwargs["lattice_grad_clip"] = guidance_cfg.get("lattice_grad_clip", 0.5)
            kwargs["coord_max_step"] = guidance_cfg.get("coord_max_step", 2e-2)
            kwargs["lattice_max_step"] = guidance_cfg.get("lattice_max_step", 2e-3)
            sample_fn = self.model.sample_CSP_algorithm5
        elif sampling_algorithm == 4:
            kwargs["tau"] = float(self.sampling_cfg["tau"])
            kwargs["n_correction_steps"] = int(self.sampling_cfg["n_correction_steps"])
            sample_fn = self.model.sample_CSP_algorithm4
        elif sampling_algorithm == 3:
            sample_fn = self.model.sample_CSP_algorithm3
        else:
            raise ValueError(f"Unsupported sampling_algorithm={sampling_algorithm}.")

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
            "target_hit_count": [0 for _ in range(target_count)],
            "target_best_rmse": [None for _ in range(target_count)],
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

        target_hits = list(state.get("target_hit_count", []))
        target_best_rmse = list(state.get("target_best_rmse", []))
        if len(target_hits) != len(results) or len(target_best_rmse) != len(results):
            raise ValueError(
                "Evaluation state target count mismatch "
                f"state_hits={len(target_hits)} state_rmse={len(target_best_rmse)} results={len(results)}"
            )

        for target_idx, result in enumerate(results):
            if not result.match or result.rmse is None:
                continue
            target_hits[target_idx] = int(target_hits[target_idx]) + 1
            best_rmse = target_best_rmse[target_idx]
            current_rmse = float(result.rmse)
            target_best_rmse[target_idx] = current_rmse if best_rmse is None else min(float(best_rmse), current_rmse)

        if state.get("first_seed_matches") is None:
            state["first_seed_matches"] = [int(result.match) for result in results]

        state["target_hit_count"] = target_hits
        state["target_best_rmse"] = target_best_rmse
        state["completed_seeds"] = sorted([*completed, seed])
        seed_summaries = [item for item in state.get("seed_summaries", []) if int(item["seed"]) != seed]
        seed_summaries.append(
            {
                "seed": seed,
                "valid": _json_float(summary.get("valid")),
                "match_rate": _json_float(summary.get("match_rate")),
                "rmse": _json_float(summary.get("rmse")),
                "composition_match_rate": _json_float(summary.get("composition_match_rate")),
                "requested_space_group_match_rate": _json_float(summary.get("requested_space_group_match_rate")),
            }
        )
        state["seed_summaries"] = sorted(seed_summaries, key=lambda item: int(item["seed"]))

    def _summary_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        seed_summaries = list(state.get("seed_summaries", []))
        valid_values = [float(item["valid"]) for item in seed_summaries if item.get("valid") is not None]
        match_values = [float(item["match_rate"]) for item in seed_summaries if item.get("match_rate") is not None]
        rmse_values = [float(item["rmse"]) for item in seed_summaries if item.get("rmse") is not None]
        target_hits = np.asarray(state.get("target_hit_count", []), dtype=int)
        reached = target_hits > 0
        best_rmse_values = [float(value) for value in state.get("target_best_rmse", []) if value is not None]

        return {
            "completed_seeds": [int(seed) for seed in state.get("completed_seeds", [])],
            "at_1_summary": {
                "valid_mean": None if not valid_values else float(np.mean(valid_values)),
                "valid_std": None if not valid_values else float(np.std(valid_values)),
                "match_rate_mean": None if not match_values else float(np.mean(match_values)),
                "match_rate_std": None if not match_values else float(np.std(match_values)),
                "rmse_mean": None if not rmse_values else float(np.mean(rmse_values)),
                "rmse_std": None if not rmse_values else float(np.std(rmse_values)),
            },
            "at_k_summary": {
                "match_rate": None if target_hits.size == 0 else float(np.mean(reached)),
                "rmse": None if not best_rmse_values else float(np.mean(best_rmse_values)),
            },
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
                f"requested_sg_match_rate={format_metric(summary.get('requested_space_group_match_rate'), '.4f')}",
                flush=True,
            )
            self._update_eval_state(state, seed=seed, results=results, summary=summary)
            self._save_eval_state(state, state_path)
            self._append_eval_log(
                log_path,
                (
                    f"seed={seed} valid={format_metric(summary['valid'], '.4f')} "
                    f"match_rate={format_metric(summary['match_rate'], '.4f')} "
                    f"rmse={format_metric(summary['rmse'], '.6f')}"
                ),
            )
            wandb.log(
                {
                    "evaluation/latest_seed": seed,
                    "evaluation/completed_seed_count": len(state["completed_seeds"]),
                    "evaluation/latest_seed_valid": summary["valid"],
                    "evaluation/latest_seed_match_rate": summary["match_rate"],
                    "evaluation/latest_seed_rmse": summary["rmse"],
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
                    "@1/valid_mean": at_1["valid_mean"],
                    "@1/valid_std": at_1["valid_std"],
                    "@1/match_rate_mean": at_1["match_rate_mean"],
                    "@1/match_rate_std": at_1["match_rate_std"],
                    "@1/rmse_mean": at_1["rmse_mean"],
                    "@1/rmse_std": at_1["rmse_std"],
                    "@1/match_hist": wandb.Histogram(summary["at_1_matches"]),
                    "@20/match_rate": at_k["match_rate"],
                    "@20/rmse": at_k["rmse"],
                    "@20/match_hist": wandb.Histogram(summary["at_k_matches"]),
                }
                if summary["at_1_rmses"]:
                    log_data["@1/rmse_hist"] = wandb.Histogram(summary["at_1_rmses"])
                if summary["at_k_rmses"]:
                    log_data["@20/rmse_hist"] = wandb.Histogram(summary["at_k_rmses"])
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
                    f"rmse_std={format_metric(at_1['rmse_std'], '.6f')}",
                    flush=True,
                )
                print(
                    f"@20 match_rate={format_metric(at_k['match_rate'], '.4f')} "
                    f"rmse={format_metric(at_k['rmse'], '.6f')} "
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
