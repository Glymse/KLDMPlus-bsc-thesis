from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path
import random
from contextlib import redirect_stderr, redirect_stdout
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from kldmPlus.sample_evaluation import (
    aggregate_csp_reconstruction_metrics,
    evaluate_csp_reconstruction,
)
from kldmPlus.run_experiment import resolve_checkpoint_reference
from kldmPlus.utils.device import get_default_device


TEST_SPLIT = "test"


class _TeeTextIO:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare KLDM samplers without WandB.")
    parser.add_argument("--config", required=True, help="Path to the compare YAML file.")
    return parser.parse_args()


def load_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if "sampler" not in config and "sampler_config" in config:
        sampler_path = (config_path.parent / str(config["sampler_config"])).expanduser().resolve()
        with sampler_path.open("r", encoding="utf-8") as handle:
            config["sampler"] = yaml.safe_load(handle) or {}

    return config_path, config


def resolve_checkpoint_path(reference: str | Path, *, config_path: Path) -> Path:
    return resolve_checkpoint_reference(reference, config_path=config_path)


def make_fixed_subset(dataset, subset_size: int | None, seed: int) -> Any:
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_metric(value: float | int | None, fmt: str) -> str:
    if value is None:
        return "na"
    return format(value, fmt)


def _format_counts(mapping: dict[str, int] | None) -> str:
    if not mapping:
        return "none"
    return ",".join(f"{key}:{value}" for key, value in sorted(mapping.items()))


def _format_vector(values: Any, fmt: str = ".4f") -> str:
    if values is None:
        return "na"
    array = np.asarray(values, dtype=float).reshape(-1)
    return "[" + ",".join(format(float(value), fmt) for value in array.tolist()) + "]"


class SamplingCompareRunner:
    def __init__(self, config_path: str | Path) -> None:
        from kldmPlus.utils.model_loader import build_model, load_checkpoint

        self.config_path, self.config = load_config(config_path)
        self.experiment_name = str(self.config["experiment_name"])
        self.compare_cfg = dict(self.config["sampling_compare"])
        self.validity_cutoff = float(self.compare_cfg.get("validity_cutoff", 0.5))
        self.debug_diagnostics = bool(self.compare_cfg.get("debug_diagnostics", int(self.compare_cfg.get("num_targets", 0)) <= 10))
        self.debug_matcher = bool(self.compare_cfg.get("debug_matcher", self.debug_diagnostics))
        self.device = get_default_device()
        self.checkpoint_path = resolve_checkpoint_path(
            self.compare_cfg["checkpoint_path"],
            config_path=self.config_path,
        )

        self.loader, self.lattice_transform = self._build_loader()
        self.template_prior = None
        self._template_prior_initialized = False
        self.model = build_model(config=self.config, device=self.device)
        load_checkpoint(
            checkpoint_path=self.checkpoint_path,
            model=self.model,
            device=self.device,
            prefer_ema_weights=True,
        )

    def _build_loader(self) -> tuple[DataLoader, Any]:
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

        requested_split = str(self.compare_cfg.get("split", TEST_SPLIT))
        if requested_split != TEST_SPLIT:
            raise ValueError(f"run_sampling_compare always uses split={TEST_SPLIT!r}, got {requested_split!r}")

        root = resolve_data_root(dataset_cfg.get("root"))
        dataset_full = task.fit_dataset(root=root, split=TEST_SPLIT, download=True)
        dataset = make_fixed_subset(
            dataset_full,
            subset_size=int(self.compare_cfg["num_targets"]),
            seed=int(self.compare_cfg["subset_seed"]),
        )

        batch_size = int(self.compare_cfg.get("batch_size", self.compare_cfg["num_targets"]))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(dataset_cfg.get("num_workers", 0)),
            pin_memory=bool(dataset_cfg.get("pin_memory", False)),
            collate_fn=dataset_full.collate_fn,
        )

        lattice_transform = task.make_lattice_transform(
            root=root,
            download=True,
        )
        return loader, lattice_transform

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

        prior_cfgs: list[dict[str, Any]] = []
        for cfg in self.compare_cfg.get("samplers", []):
            algorithm = int(cfg.get("sampling_algorithm", 4 if str(cfg["method"]) == "pc" else 3))
            if algorithm == 10:
                prior_cfgs.append(dict(cfg.get("algorithm10", {})))
        if not prior_cfgs:
            return None
        template_prior_modes = {
            str(cfg.get("template_prior_mode", "dataset")).strip().lower() or "dataset"
            for cfg in prior_cfgs
        }
        if template_prior_modes and template_prior_modes <= {"none", "oracle_surrogate"}:
            print(
                "template_prior_build skip "
                f"modes={sorted(template_prior_modes)} source=non_dataset_prior",
                flush=True,
            )
            return None
        if not any(bool(cfg.get("template_prior_enabled", True)) and float(cfg.get("template_prior_weight", 1.0)) > 0.0 for cfg in prior_cfgs):
            return None

        dataset_cfg = dict(self.config["dataset"])
        model_cfg = dict(self.config["model"])
        root = resolve_data_root(dataset_cfg.get("root"))
        task = CSPTask(
            dataset_name=str(dataset_cfg["name"]),
            lattice_parameterization=str(model_cfg["lattice_parameterization"]),
            lattice_representation=str(dataset_cfg.get("lattice_representation", "kldm")),
        )
        train_dataset = task.fit_dataset(root=root, split="train", download=True)
        max_samples = max((int(cfg.get("template_prior_max_samples", 2000)) for cfg in prior_cfgs), default=2000)
        if max_samples <= 0:
            return None
        match_targets_only = any(bool(cfg.get("template_prior_match_targets_only", True)) for cfg in prior_cfgs)
        allowed_keys = None
        if match_targets_only:
            allowed_keys = {
                requested_composition_key(
                    space_group_number=int(torch.as_tensor(sample.space_group).reshape(-1)[0].item()),
                    atomic_numbers=sample.atomic_numbers,
                )
                for sample in self.loader.dataset
            }
        cache_enabled = any(bool(cfg.get("template_prior_cache", True)) for cfg in prior_cfgs)
        cache_path = None
        if cache_enabled:
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

    def _sample_batch(self, batch, sampler_cfg: dict[str, Any]):
        method = str(sampler_cfg["method"])
        sampling_algorithm = int(sampler_cfg.get("sampling_algorithm", 4 if method == "pc" else 3))
        kwargs = {
            "n_steps": int(sampler_cfg["n_steps"]),
            "batch": batch,
            "t_start": float(sampler_cfg["t_start"]),
            "t_final": float(sampler_cfg["t_final"]),
        }

        if sampling_algorithm == 10:
            if method not in {"casal", "em"}:
                raise ValueError(
                    "sampling_algorithm=10 expects sampling.method in {'casal', 'em'} for KLDM reverse-step compatibility.",
                )
            kwargs["lattice_transform"] = self.lattice_transform
            kwargs["algorithm10_config"] = dict(sampler_cfg.get("algorithm10", {}))
            kwargs["template_prior"] = self._ensure_template_prior()
            sample_fn = self.model.sample_CSP_algorithm10
        elif sampling_algorithm in {5, 6, 7, 8, 9}:
            raise ValueError(
                f"sampling_algorithm={sampling_algorithm} was removed during cleanup. "
                "Supported sampler paths are original KLDMplus (3/4) and CASAL/CASCAL (10).",
            )
        elif sampling_algorithm == 4:
            kwargs["tau"] = float(sampler_cfg["tau"])
            kwargs["n_correction_steps"] = int(sampler_cfg["n_correction_steps"])
            if method == "facit_pc":
                sample_fn = self.model.sample_CSP_algorithm4_facit
            else:
                sample_fn = self.model.sample_CSP_algorithm4
        elif sampling_algorithm == 3:
            if method == "facit_em":
                sample_fn = self.model.sample_CSP_algorithm3_facit
            else:
                sample_fn = self.model.sample_CSP_algorithm3
        else:
            raise ValueError(f"Unsupported sampling_algorithm={sampling_algorithm}.")

        return sample_fn(**kwargs)

    def _collect(self, sampler_cfg: dict[str, Any]) -> list[Any]:
        self.model.eval()
        results: list[Any] = []
        sampler_name = str(sampler_cfg["name"])
        total_batches = len(self.loader)
        started_at = time.perf_counter()

        print(
            f"sampler_progress name={sampler_name} phase=start total_batches={total_batches}",
            flush=True,
        )

        for batch_idx, batch in enumerate(self.loader, start=1):
            batch = batch.to(self.device)
            pos_t, _v_t, l_t, h_t = self._sample_batch(batch, sampler_cfg=sampler_cfg)

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
                        requested_space_group=int(torch.as_tensor(batch.space_group).reshape(-1)[graph_idx].item()),
                        validity_cutoff=self.validity_cutoff,
                    )
                )

            elapsed_s = time.perf_counter() - started_at
            print(
                f"sampler_progress name={sampler_name} phase=batch "
                f"batch={batch_idx}/{total_batches} graphs_in_batch={batch.num_graphs} "
                f"elapsed_s={elapsed_s:.1f}",
                flush=True,
            )

        total_elapsed_s = time.perf_counter() - started_at
        print(
            f"sampler_progress name={sampler_name} phase=done num_results={len(results)} "
            f"elapsed_s={total_elapsed_s:.1f}",
            flush=True,
        )
        return results

    def _print_sample_diagnostics(self, *, sampler_name: str, results: list[Any]) -> None:
        if not self.debug_diagnostics:
            for index, result in enumerate(results, start=1):
                diagnostics = result.matcher_diagnostics
                should_explain = (
                    (not result.match)
                    or (
                        result.requested_space_group_match is not None
                        and not bool(result.requested_space_group_match)
                    )
                )
                if not should_explain or diagnostics is None:
                    continue
                print(
                    f"sample_failure sampler={sampler_name} idx={index} "
                    f"diagnosis={diagnostics.diagnosis} "
                    f"requested_sg={result.requested_space_group if result.requested_space_group is not None else 'na'} "
                    f"detected_sg={result.detected_space_group if result.detected_space_group is not None else 'na'} "
                    f"requested_sg_match={int(result.requested_space_group_match) if result.requested_space_group_match is not None else 'na'} "
                    f"frac_rmse={format_metric(result.frac_rmse, '.6f')} "
                    f"standardized_frac_rmse={format_metric(diagnostics.standardized_frac_rmse, '.6f')} "
                    f"lattice_lengths_mae={format_metric(result.lattice_lengths_mae, '.6f')} "
                    f"lattice_angles_mae={format_metric(result.lattice_angles_mae, '.6f')}",
                    flush=True,
                )
            return
        for index, result in enumerate(results, start=1):
            predicted = result.predicted_structure
            target = result.target_structure
            print(
                f"sample_debug sampler={sampler_name} idx={index} "
                f"valid={int(result.valid)} match={int(result.match)} "
                f"rmse={format_metric(result.rmse, '.6f')} "
                f"composition_match={int(result.composition_match) if result.composition_match is not None else 'na'} "
                f"requested_sg={result.requested_space_group if result.requested_space_group is not None else 'na'} "
                f"detected_sg={result.detected_space_group if result.detected_space_group is not None else 'na'} "
                f"requested_sg_match={int(result.requested_space_group_match) if result.requested_space_group_match is not None else 'na'} "
                f"validity_reason={result.validity_reason or 'na'} "
                f"min_pair_distance={format_metric(result.min_pair_distance, '.4f')} "
                f"volume={format_metric(result.volume, '.4f')} "
                f"max_lattice_length={format_metric(result.max_lattice_length, '.4f')} "
                f"frac_rmse={format_metric(result.frac_rmse, '.6f')} "
                f"frac_status={result.frac_rmse_status or 'na'} "
                f"lattice_lengths_mae={format_metric(result.lattice_lengths_mae, '.6f')} "
                f"lattice_angles_mae={format_metric(result.lattice_angles_mae, '.6f')}",
                flush=True,
            )
            print(
                f"sample_lattice sampler={sampler_name} idx={index} "
                f"pred_formula={predicted.composition.formula if predicted is not None else 'na'} "
                f"target_formula={target.composition.formula if target is not None else 'na'} "
                f"pred_abc={_format_vector(None if predicted is None else predicted.lattice.abc)} "
                f"target_abc={_format_vector(None if target is None else target.lattice.abc)} "
                f"pred_angles={_format_vector(None if predicted is None else predicted.lattice.angles)} "
                f"target_angles={_format_vector(None if target is None else target.lattice.angles)}",
                flush=True,
            )
            diagnostics = result.matcher_diagnostics if self.debug_matcher else None
            if diagnostics is None:
                continue
            pred_std = diagnostics.predicted_standardized_structure
            target_std = diagnostics.target_standardized_structure
            print(
                f"sample_matcher sampler={sampler_name} idx={index} "
                f"diagnosis={diagnostics.diagnosis} "
                f"standardized_match={int(diagnostics.conventional_match)} "
                f"standardized_rmse={format_metric(diagnostics.conventional_rmse, '.6f')} "
                f"primitive_match={int(diagnostics.primitive_match)} "
                f"primitive_rmse={format_metric(diagnostics.primitive_rmse, '.6f')} "
                f"standardized_pred_sg={diagnostics.standardized_predicted_space_group if diagnostics.standardized_predicted_space_group is not None else 'na'} "
                f"standardized_target_sg={diagnostics.standardized_target_space_group if diagnostics.standardized_target_space_group is not None else 'na'} "
                f"standardized_frac_rmse={format_metric(diagnostics.standardized_frac_rmse, '.6f')} "
                f"standardized_frac_status={diagnostics.standardized_frac_status or 'na'}",
                flush=True,
            )
            print(
                f"sample_matcher_lattice sampler={sampler_name} idx={index} "
                f"pred_std_abc={_format_vector(None if pred_std is None else pred_std.lattice.abc)} "
                f"target_std_abc={_format_vector(None if target_std is None else target_std.lattice.abc)} "
                f"pred_std_angles={_format_vector(None if pred_std is None else pred_std.lattice.angles)} "
                f"target_std_angles={_format_vector(None if target_std is None else target_std.lattice.angles)}",
                flush=True,
            )
            for species_diag in diagnostics.species_errors:
                print(
                    f"sample_matcher_species sampler={sampler_name} idx={index} "
                    f"species={species_diag.symbol} "
                    f"count={species_diag.count} "
                    f"rmse={format_metric(species_diag.rmse, '.6f')} "
                    f"mean_distance={format_metric(species_diag.mean_distance, '.6f')} "
                    f"max_distance={format_metric(species_diag.max_distance, '.6f')} "
                    f"mean_shift={_format_vector(species_diag.mean_torus_shift)} "
                    f"shift_spread={format_metric(species_diag.max_shift_deviation, '.6f')} "
                    f"pred_orbits={species_diag.predicted_orbits if species_diag.predicted_orbits else 'na'} "
                    f"target_orbits={species_diag.target_orbits if species_diag.target_orbits else 'na'}",
                    flush=True,
                )

    def run(self) -> None:
        sampler_specs = list(self.compare_cfg["samplers"])
        sample_seed = int(self.compare_cfg.get("sample_seed", 0))

        print(f"experiment={self.experiment_name}", flush=True)
        print(f"checkpoint={self.checkpoint_path}", flush=True)
        print(
            f"subset split={TEST_SPLIT} num_targets={int(self.compare_cfg['num_targets'])} "
            f"subset_seed={int(self.compare_cfg['subset_seed'])} sample_seed={sample_seed}",
            flush=True,
        )

        summaries: list[tuple[str, dict[str, Any]]] = []
        for sampler_cfg in sampler_specs:
            sampler_name = str(sampler_cfg["name"])
            set_seed(sample_seed)
            results = self._collect(sampler_cfg=sampler_cfg)
            summary = aggregate_csp_reconstruction_metrics(results)
            summaries.append((sampler_name, summary))
            self._print_sample_diagnostics(sampler_name=sampler_name, results=results)
            print(
                f"sampler={sampler_name} "
                f"valid={format_metric(summary['valid'], '.4f')} "
                f"match_rate={format_metric(summary['match_rate'], '.4f')} "
                f"rmse={format_metric(summary['rmse'], '.6f')} "
                f"composition_match_rate={format_metric(summary.get('composition_match_rate'), '.4f')} "
                f"requested_sg_match_rate={format_metric(summary.get('requested_space_group_match_rate'), '.4f')} "
                f"num_samples={summary['num_samples']}",
                flush=True,
            )
            print(
                f"sampler_debug={sampler_name} "
                f"frac_rmse={format_metric(summary.get('frac_rmse'), '.6f')} "
                f"standardized_frac_rmse={format_metric(summary.get('standardized_frac_rmse'), '.6f')} "
                f"rmse_defined={summary.get('rmse_defined_count', 0)}/{summary['num_samples']} "
                f"frac_defined={summary.get('frac_rmse_defined_count', 0)}/{summary['num_samples']} "
                f"std_frac_defined={summary.get('standardized_frac_rmse_defined_count', 0)}/{summary['num_samples']} "
                f"matcher_failures={_format_counts(summary.get('matcher_diagnosis_counts'))}",
                flush=True,
            )

        if len(summaries) >= 2:
            baseline_name, baseline = summaries[0]
            challenger_name, challenger = summaries[1]

            def _delta(key: str) -> float | None:
                left = baseline.get(key)
                right = challenger.get(key)
                if left is None or right is None:
                    return None
                return float(right) - float(left)

            print(
                f"delta {challenger_name}-{baseline_name} "
                f"valid={format_metric(_delta('valid'), '+.4f')} "
                f"match_rate={format_metric(_delta('match_rate'), '+.4f')} "
                f"rmse={format_metric(_delta('rmse'), '+.6f')} "
                f"frac_rmse={format_metric(_delta('frac_rmse'), '+.6f')} "
                f"standardized_frac_rmse={format_metric(_delta('standardized_frac_rmse'), '+.6f')} "
                f"composition_match_rate={format_metric(_delta('composition_match_rate'), '+.4f')} "
                f"requested_sg_match_rate={format_metric(_delta('requested_space_group_match_rate'), '+.4f')}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    log_path = Path.cwd() / "sampling-compare.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        tee_stdout = _TeeTextIO(sys.stdout, log_handle)
        tee_stderr = _TeeTextIO(sys.stderr, log_handle)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            print(f"sampling_compare_log_path={log_path}", flush=True)
            SamplingCompareRunner(config_path=config_path).run()


if __name__ == "__main__":
    main()
