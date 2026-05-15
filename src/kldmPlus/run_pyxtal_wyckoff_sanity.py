from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from kldmPlus.sample_evaluation.sample_evaluation import build_structure_from_sample
from kldmPlus.symmetry import build_pyxtal_wyckoff_result, species_aware_torus_rmse

try:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Element, Lattice, Structure
except ImportError:  # pragma: no cover
    Element = Lattice = StructureMatcher = Structure = None


TEST_SPLIT = "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyXtal Wyckoff sanity checks on GT structures.")
    parser.add_argument("--config", required=True, help="Path to the PyXtal sanity YAML file.")
    return parser.parse_args()


def load_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config_path, config


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


def _structure_id_from_batch(batch) -> str:
    value = getattr(batch, "structure_id", None)
    if value is None:
        return "unknown"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    try:
        return str(value[0])
    except Exception:
        return str(value)


def _require_pymatgen_structure_tools() -> None:
    if None in (Element, Lattice, Structure, StructureMatcher):
        raise ImportError("run_pyxtal_wyckoff_sanity requires pymatgen structure tools.")


def _build_structure_from_pyxtal_result(result) -> Any:
    _require_pymatgen_structure_tools()
    species = [Element.from_Z(int(z)).symbol for z in result.expanded_atomic_numbers.tolist()]
    return Structure(
        lattice=Lattice.from_parameters(*result.lattice_parameters),
        species=species,
        coords=result.expanded_frac_coords.tolist(),
        coords_are_cartesian=False,
    ).get_sorted_structure()


def _structure_match_metrics(
    *,
    predicted_structure,
    target_structure,
    stol: float,
    angle_tol: float,
    ltol: float,
) -> tuple[bool, float | None, str]:
    _require_pymatgen_structure_tools()
    matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
    try:
        matched = bool(matcher.fit(target_structure, predicted_structure))
    except Exception as exc:
        return False, None, f"matcher_fit_error:{type(exc).__name__}"

    try:
        rms = matcher.get_rms_dist(target_structure, predicted_structure)
    except Exception as exc:
        return matched, None, f"matcher_rms_error:{type(exc).__name__}"

    if rms is None:
        return matched, None, "matcher_rms_none"
    return matched, float(rms[0]), "ok"


class PyXtalWyckoffSanityRunner:
    def __init__(self, config_path: str | Path) -> None:
        from kldmPlus.data import CSPTask, resolve_data_root
        from kldmPlus.data.csp import validate_lattice_configuration

        self.config_path, self.config = load_config(config_path)
        self.experiment_name = str(self.config["experiment_name"])
        self.sanity_cfg = dict(self.config["pyxtal_wyckoff_sanity"])

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

        requested_split = str(self.sanity_cfg.get("split", TEST_SPLIT))
        if requested_split != TEST_SPLIT:
            raise ValueError(f"run_pyxtal_wyckoff_sanity always uses split={TEST_SPLIT!r}, got {requested_split!r}")

        root = resolve_data_root(dataset_cfg.get("root"))
        dataset_full = task.fit_dataset(root=root, split=TEST_SPLIT, download=True)
        dataset = make_fixed_subset(
            dataset_full,
            subset_size=int(self.sanity_cfg["num_targets"]),
            seed=int(self.sanity_cfg["subset_seed"]),
        )
        self.loader = DataLoader(
            dataset,
            batch_size=int(self.sanity_cfg.get("batch_size", 1)),
            shuffle=False,
            num_workers=int(dataset_cfg.get("num_workers", 0)),
            pin_memory=bool(dataset_cfg.get("pin_memory", False)),
            collate_fn=dataset_full.collate_fn,
        )
        self.lattice_transform = task.make_lattice_transform(
            root=root,
            download=True,
            mattergen_limit_var_scaling_constant=model_cfg.get("mattergen_limit_var_scaling_constant"),
        )

    def run(self) -> None:
        sample_seed = int(self.sanity_cfg.get("sample_seed", 0))
        set_seed(sample_seed)

        print(f"experiment={self.experiment_name}", flush=True)
        print(
            f"subset split={TEST_SPLIT} num_targets={int(self.sanity_cfg['num_targets'])} "
            f"subset_seed={int(self.sanity_cfg['subset_seed'])} sample_seed={sample_seed}",
            flush=True,
        )

        sg_matches: list[int] = []
        successful: list[int] = []
        anchor_ratios: list[float] = []
        raw_coord_rmses: list[float] = []
        matcher_rmses: list[float] = []
        matcher_success: list[int] = []
        records: list[dict[str, Any]] = []

        symprec = float(self.sanity_cfg.get("symprec", 1e-2))
        pyxtal_tol = float(self.sanity_cfg.get("pyxtal_tol", 1e-2))
        stol = float(self.sanity_cfg.get("stol", 0.5))
        angle_tol = float(self.sanity_cfg.get("angle_tol", 10.0))
        ltol = float(self.sanity_cfg.get("ltol", 0.3))

        for batch in self.loader:
            gt_structure = build_structure_from_sample(
                f=batch.pos,
                l=batch.l[0],
                a=batch.atomic_numbers,
                lattice_transform=self.lattice_transform,
            )

            dataset_sg = int(torch.as_tensor(batch.space_group).reshape(-1)[0].item())
            structure_id = _structure_id_from_batch(batch)
            try:
                result = build_pyxtal_wyckoff_result(
                    gt_structure,
                    symprec=symprec,
                    pyxtal_tol=pyxtal_tol,
                )
                rmse, rmse_status = species_aware_torus_rmse(
                    source_frac_coords=result.expanded_frac_coords,
                    source_atomic_numbers=result.expanded_atomic_numbers,
                    target_frac_coords=np.asarray(gt_structure.frac_coords, dtype=float),
                    target_atomic_numbers=np.asarray(gt_structure.atomic_numbers, dtype=int),
                )
                predicted_structure = _build_structure_from_pyxtal_result(result)
                matched, matcher_rmse, matcher_status = _structure_match_metrics(
                    predicted_structure=predicted_structure,
                    target_structure=gt_structure,
                    stol=stol,
                    angle_tol=angle_tol,
                    ltol=ltol,
                )
                ok = rmse is not None
                successful.append(int(ok))
                sg_matches.append(int(result.space_group == dataset_sg))
                matcher_success.append(int(matched))
                anchor_ratio = float(result.anchor_count / max(result.num_atoms, 1))
                anchor_ratios.append(anchor_ratio)
                if rmse is not None:
                    raw_coord_rmses.append(float(rmse))
                if matcher_rmse is not None:
                    matcher_rmses.append(float(matcher_rmse))
                records.append(
                    {
                        "structure_id": structure_id,
                        "dataset_sg": dataset_sg,
                        "pyxtal_sg": int(result.space_group),
                        "anchor_count": int(result.anchor_count),
                        "num_atoms": int(result.num_atoms),
                        "anchor_ratio": anchor_ratio,
                        "coord_rmse": rmse,
                        "coord_rmse_status": rmse_status,
                        "matcher_match": matched,
                        "matcher_rmse": matcher_rmse,
                        "matcher_status": matcher_status,
                    }
                )
            except Exception as exc:
                successful.append(0)
                sg_matches.append(0)
                matcher_success.append(0)
                records.append(
                    {
                        "structure_id": structure_id,
                        "dataset_sg": dataset_sg,
                        "pyxtal_sg": None,
                        "anchor_count": None,
                        "num_atoms": int(batch.pos.shape[0]),
                        "anchor_ratio": None,
                        "coord_rmse": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        def _mean(values: list[float]) -> float | None:
            return None if not values else float(np.mean(values))

        print(
            f"pyxtal_success rate={format_metric(_mean([float(v) for v in successful]), '.4f')} "
            f"count={sum(successful)}/{len(successful)}",
            flush=True,
        )
        print(
            f"space_group_match dataset_vs_pyxtal={format_metric(_mean([float(v) for v in sg_matches]), '.4f')} "
            f"count={sum(sg_matches)}/{len(sg_matches)}",
            flush=True,
        )
        print(
            f"anchors anchor_ratio_mean={format_metric(_mean(anchor_ratios), '.4f')} "
            f"anchor_ratio_std={format_metric(float(np.std(anchor_ratios)) if anchor_ratios else None, '.4f')}",
            flush=True,
        )
        print(
            f"expanded_vs_gt coord_rmse_mean={format_metric(_mean(raw_coord_rmses), '.6f')} "
            f"coord_rmse_std={format_metric(float(np.std(raw_coord_rmses)) if raw_coord_rmses else None, '.6f')}",
            flush=True,
        )
        print(
            f"structure_match rate={format_metric(_mean([float(v) for v in matcher_success]), '.4f')} "
            f"count={sum(matcher_success)}/{len(matcher_success)}",
            flush=True,
        )
        print(
            f"structure_match rms_mean={format_metric(_mean(matcher_rmses), '.6f')} "
            f"rms_std={format_metric(float(np.std(matcher_rmses)) if matcher_rmses else None, '.6f')}",
            flush=True,
        )

        best_records = sorted(
            [item for item in records if item.get("matcher_rmse") is not None],
            key=lambda item: float(item["matcher_rmse"]),
        )[:5]
        print("top_pyxtal_examples", flush=True)
        for item in best_records:
            print(
                f"structure_id={item['structure_id']} dataset_sg={item['dataset_sg']} "
                f"pyxtal_sg={item['pyxtal_sg']} anchors={item['anchor_count']} num_atoms={item['num_atoms']} "
                f"anchor_ratio={item['anchor_ratio']:.4f} coord_rmse={format_metric(item['coord_rmse'], '.6f')} "
                f"coord_status={item['coord_rmse_status']} matcher_match={int(item['matcher_match'])} "
                f"matcher_rmse={format_metric(item['matcher_rmse'], '.6f')} matcher_status={item['matcher_status']}",
                flush=True,
            )

        failed = [
            item
            for item in records
            if item.get("matcher_rmse") is None or not item.get("matcher_match", False)
        ][:5]
        if failed:
            print("top_pyxtal_failures", flush=True)
            for item in failed:
                print(
                    f"structure_id={item['structure_id']} dataset_sg={item['dataset_sg']} "
                    f"pyxtal_sg={item['pyxtal_sg']} coord_status={item.get('coord_rmse_status', 'na')} "
                    f"matcher_match={int(item.get('matcher_match', False))} "
                    f"matcher_status={item.get('matcher_status', item.get('error', 'unknown'))}",
                    flush=True,
                )


def main() -> None:
    args = parse_args()
    PyXtalWyckoffSanityRunner(config_path=args.config).run()


if __name__ == "__main__":
    main()
