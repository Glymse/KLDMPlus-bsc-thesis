from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import tempfile
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
)
from kldmPlus.sample_evaluation import prepare_visualization_pair
from kldmPlus.utils.device import get_default_device

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise ImportError("wandb is required for src/kldmPlus/run_sampling.py") from exc


TEST_SPLIT = "test"
AT_K = 20


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


class SamplingRunner:
    def __init__(self, config_path: str | Path) -> None:
        from kldmPlus.utils.model_loader import build_model, load_checkpoint

        self.config_path, self.config = load_experiment_config(config_path)
        self.experiment_name = str(self.config["experiment_name"])
        self.sampling_cfg = dict(self.config["sampling"])
        self.eval_cfg = dict(self.config["sampling_eval"])
        self.evaluation = bool(self.eval_cfg["evaluation"])
        self.sample_count = int(self.sampling_cfg["n_samples"])
        self.device = get_default_device()
        self.checkpoint_path = self._checkpoint_path(self.sampling_cfg["checkpoint_path"])
        self.loader, self.lattice_transform = self._build_loader()
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

    # Samples one batch with either EM or PC, depending on the config.
    def _sample_batch(self, batch):
        method = str(self.sampling_cfg["method"])
        sample_fn = self.model.sample_CSP_algorithm4 if method == "pc" else self.model.sample_CSP_algorithm3
        kwargs = {
            "n_steps": int(self.sampling_cfg["n_steps"]),
            "batch": batch,
            "t_start": float(self.sampling_cfg["t_start"]),
            "t_final": float(self.sampling_cfg["t_final"]),
        }
        if method == "pc":
            kwargs["tau"] = float(self.sampling_cfg["tau"])
            kwargs["n_correction_steps"] = int(self.sampling_cfg["n_correction_steps"])
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
        if seed is not None:
            _set_seed(seed)
            print(f"eval_seed={seed} sampling_pass_start samples_per_target={samples_per_target}", flush=True)

        results: list[list[Any]] = []
        total_batches = len(self.loader)
        for batch_idx, batch in enumerate(self.loader, start=1):
            batch = batch.to(self.device)
            per_graph = [[] for _ in range(batch.num_graphs)]

            if seed is not None:
                print(
                    f"eval_seed={seed} batch={batch_idx}/{total_batches} graphs_in_batch={batch.num_graphs}",
                    flush=True,
                )

            for _ in range(samples_per_target):
                with torch.no_grad():
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
                        )
                    )

            results.extend(per_graph)

        if seed is not None:
            print(f"eval_seed={seed} sampling_pass_done targets={len(results)}", flush=True)
        return results

    # Runs repeated single-sample passes and aggregates them into @1/@20 summaries.
    def _evaluate(self) -> dict[str, Any]:
        from kldmPlus.sample_evaluation import aggregate_csp_reconstruction_metrics

        pass_summaries = []
        pass_results = []

        for seed in range(AT_K):
            print(f"evaluation_progress seed={seed + 1}/{AT_K}", flush=True)
            results = [graph_results[0] for graph_results in self._collect(samples_per_target=1, seed=seed)]
            summary = aggregate_csp_reconstruction_metrics(results)
            print(
                f"evaluation_seed_summary seed={seed} "
                f"valid={format_metric(summary['valid'], '.4f')} "
                f"match_rate={format_metric(summary['match_rate'], '.4f')} "
                f"rmse={format_metric(summary['rmse'], '.6f')}",
                flush=True,
            )
            pass_summaries.append(summary)
            pass_results.append(results)

        first_results = pass_results[0] if pass_results else []
        best_results = []
        if pass_results:
            for target_idx in range(len(pass_results[0])):
                best_results.append(_best_result([one_pass[target_idx] for one_pass in pass_results]))

        valid_values = [summary["valid"] for summary in pass_summaries if summary["valid"] is not None]
        match_values = [summary["match_rate"] for summary in pass_summaries if summary["match_rate"] is not None]
        rmse_values = [summary["rmse"] for summary in pass_summaries if summary["rmse"] is not None]
        at_k_match_rate, at_k_rmse = _merge_pass_statistics(pass_results)

        return {
            "at_1_summary": {
                "valid_mean": None if not valid_values else float(np.mean(valid_values)),
                "valid_std": None if not valid_values else float(np.std(valid_values)),
                "match_rate_mean": None if not match_values else float(np.mean(match_values)),
                "match_rate_std": None if not match_values else float(np.std(match_values)),
                "rmse_mean": None if not rmse_values else float(np.mean(rmse_values)),
                "rmse_std": None if not rmse_values else float(np.std(rmse_values)),
            },
            "at_k_summary": {
                "match_rate": at_k_match_rate,
                "rmse": at_k_rmse,
            },
            "at_1_results": first_results,
            "at_k_results": best_results,
            "at_1_matches": [int(result.match) for result in first_results],
            "at_k_matches": [int(result.match) for result in best_results],
            "at_1_rmses": rmse_values,
            "at_k_rmses": [float(result.rmse) for result in best_results if result.rmse is not None],
        }

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
        wandb.init(
            project="mp_20_sampling",
            name=f"{'EVAL' if self.evaluation else 'SAMPLES'}_{self.experiment_name}",
            config={
                "experiment_name": self.experiment_name,
                "config_path": str(self.config_path),
                "checkpoint_path": str(self.checkpoint_path),
                "evaluation": self.evaluation,
                "n_samples": self.sample_count,
                "sampling": self.sampling_cfg,
                "sampling_eval": self.eval_cfg,
            },
        )
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
                    f"rmse={format_metric(recon['rmse'], '.6f')}",
                    flush=True,
                )
            else:
                summary = self._evaluate()
                self._log_eval_examples(summary, temp_dir)
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
                    f"rmse={format_metric(at_k['rmse'], '.6f')}",
                    flush=True,
                )

        wandb.finish()


def main() -> None:
    SamplingRunner(parse_args().config).run()


if __name__ == "__main__":
    main()
