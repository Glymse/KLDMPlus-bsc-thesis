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
from kldmPlus.symmetry import (
    build_pyxtal_wyckoff_result,
    build_symmetry_frame_bridge,
    expand_wyckoff_template_torch,
    extract_wyckoff_templates,
    flatten_site_signature,
    map_standardized_structure_to_vanilla_frame,
    transport_standardized_structure_to_vanilla_frame,
)

try:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Element, Lattice, Structure
except ImportError:  # pragma: no cover
    Element = Lattice = StructureMatcher = Structure = None


TEST_SPLIT = "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyXtal template extraction / torch expansion sanity checks.")
    parser.add_argument("--config", required=True, help="Path to the Wyckoff template sanity YAML file.")
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


def _require_pymatgen() -> None:
    if None in (Element, Lattice, StructureMatcher, Structure):
        raise ImportError("run_wyckoff_template_sanity requires pymatgen structure tools.")


def _build_structure_from_expansion(expansion, lattice_parameters: np.ndarray):
    _require_pymatgen()
    species = [Element.from_Z(int(z)).symbol for z in expansion.atomic_numbers.tolist()]
    return Structure(
        lattice=Lattice.from_parameters(*lattice_parameters.tolist()),
        species=species,
        coords=expansion.frac_coords.detach().cpu().numpy().tolist(),
        coords_are_cartesian=False,
    ).get_sorted_structure()


def _recover_free_vars_from_gt(template, gt_site_entries: list[dict[str, Any]]) -> torch.Tensor:
    if len(gt_site_entries) != len(template.site_templates):
        raise ValueError("Template / GT anchor count mismatch.")

    grouped_entries: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for entry in gt_site_entries:
        key = (int(entry["atomic_number"]), str(entry["label"]))
        grouped_entries.setdefault(key, []).append(entry)
    for value in grouped_entries.values():
        value.sort(key=lambda item: tuple(np.round(item["anchor_frac"], 8).tolist()))

    recovered: list[float] = []
    for template_site in template.site_templates:
        key = (int(template_site.atomic_number), str(template_site.label))
        if key not in grouped_entries or not grouped_entries[key]:
            raise ValueError("Template / GT site signature mismatch.")
        gt_site = grouped_entries[key].pop(0)
        if template_site.dof == 0:
            continue

        basis = np.asarray(template_site.anchor_basis, dtype=float)
        offset = np.asarray(template_site.anchor_offset, dtype=float)
        anchor = np.asarray(gt_site["anchor_frac"], dtype=float)
        solution, *_ = np.linalg.lstsq(basis, anchor - offset, rcond=None)
        recovered.extend(solution.tolist())

    return torch.as_tensor(recovered, dtype=torch.get_default_dtype())


def _matcher_rms(predicted_structure, target_structure, *, stol: float, angle_tol: float, ltol: float) -> tuple[bool, float | None]:
    _require_pymatgen()
    matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
    matched = bool(matcher.fit(target_structure, predicted_structure))
    rms = matcher.get_rms_dist(target_structure, predicted_structure)
    return matched, (None if rms is None else float(rms[0]))


class WyckoffTemplateSanityRunner:
    def __init__(self, config_path: str | Path) -> None:
        from kldmPlus.data import CSPTask, resolve_data_root
        from kldmPlus.data.csp import validate_lattice_configuration

        self.config_path, self.config = load_config(config_path)
        self.experiment_name = str(self.config["experiment_name"])
        self.sanity_cfg = dict(self.config["wyckoff_template_sanity"])

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
            raise ValueError(f"run_wyckoff_template_sanity always uses split={TEST_SPLIT!r}, got {requested_split!r}")

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

        symprec = float(self.sanity_cfg.get("symprec", 1e-2))
        pyxtal_tol = float(self.sanity_cfg.get("pyxtal_tol", 1e-2))
        max_templates = int(self.sanity_cfg.get("max_templates", 64))
        quick = bool(self.sanity_cfg.get("quick", False))
        standardization = str(self.sanity_cfg.get("standardization", "conventional"))
        stol = float(self.sanity_cfg.get("stol", 0.5))
        angle_tol = float(self.sanity_cfg.get("angle_tol", 10.0))
        ltol = float(self.sanity_cfg.get("ltol", 0.3))

        candidate_counts: list[float] = []
        coverage_hits: list[int] = []
        expander_hits: list[int] = []
        expander_rmses: list[float] = []
        records: list[dict[str, Any]] = []

        for batch in self.loader:
            vanilla_structure = build_structure_from_sample(
                f=batch.pos,
                l=batch.l[0],
                a=batch.atomic_numbers,
                lattice_transform=self.lattice_transform,
            )
            bridge = build_symmetry_frame_bridge(
                vanilla_structure=vanilla_structure,
                standardization=standardization,
                symprec=symprec,
                angle_tolerance=angle_tol,
                stol=stol,
                ltol=ltol,
            )
            structure_id = _structure_id_from_batch(batch)
            dataset_sg = int(torch.as_tensor(batch.space_group).reshape(-1)[0].item())
            gt_result = build_pyxtal_wyckoff_result(
                bridge.standardized_structure,
                symprec=symprec,
                pyxtal_tol=pyxtal_tol,
            )
            gt_site_entries = [
                {
                    "atomic_number": int(gt_result.anchor_atomic_numbers[site_idx]),
                    "label": str(gt_result.site_labels[site_idx]),
                    "anchor_frac": np.asarray(gt_result.anchor_frac_coords[site_idx], dtype=float),
                }
                for site_idx in range(int(gt_result.anchor_count))
            ]
            gt_signature = tuple(
                sorted(
                    (int(item["atomic_number"]), str(item["label"]))
                    for item in gt_site_entries
                )
            )

            templates = extract_wyckoff_templates(
                space_group_number=bridge.standardized_space_group,
                atomic_numbers=bridge.standardized_atomic_numbers,
                max_templates=max_templates,
                quick=quick,
            )
            candidate_counts.append(float(len(templates)))

            matched_template = None
            for template in templates:
                if flatten_site_signature(template) == gt_signature:
                    matched_template = template
                    break

            covered = matched_template is not None
            coverage_hits.append(int(covered))

            matcher_ok = False
            matcher_rms = None
            status = "no_matching_template"
            if matched_template is not None:
                free_vars = _recover_free_vars_from_gt(matched_template, gt_site_entries)
                expansion = expand_wyckoff_template_torch(
                    template=matched_template,
                    free_vars=free_vars,
                )
                predicted_structure = _build_structure_from_expansion(
                    expansion=expansion,
                    lattice_parameters=gt_result.lattice_parameters,
                )
                try:
                    predicted_vanilla_like = transport_standardized_structure_to_vanilla_frame(
                        standardized_structure=predicted_structure,
                        bridge=bridge,
                    )
                except Exception:
                    predicted_vanilla_like = map_standardized_structure_to_vanilla_frame(
                        standardized_structure=predicted_structure,
                        vanilla_reference_structure=bridge.vanilla_structure,
                        symprec=symprec,
                        angle_tolerance=angle_tol,
                        stol=stol,
                        ltol=ltol,
                    )
                matcher_ok, matcher_rms = _matcher_rms(
                    predicted_vanilla_like,
                    bridge.vanilla_structure,
                    stol=stol,
                    angle_tol=angle_tol,
                    ltol=ltol,
                )
                status = "ok" if matcher_ok else "matcher_mismatch"
                expander_hits.append(int(matcher_ok))
                if matcher_rms is not None:
                    expander_rmses.append(float(matcher_rms))
            else:
                expander_hits.append(0)

            records.append(
                {
                    "structure_id": structure_id,
                    "dataset_sg": dataset_sg,
                    "gt_sg": int(gt_result.space_group),
                    "standardized_sg": int(bridge.standardized_space_group),
                    "num_templates": int(len(templates)),
                    "covered": covered,
                    "matcher_ok": matcher_ok,
                    "matcher_rms": matcher_rms,
                    "status": status,
                    "gt_signature": gt_signature,
                }
            )

        def _mean(values: list[float]) -> float | None:
            return None if not values else float(np.mean(values))

        print(
            f"templates mean={format_metric(_mean(candidate_counts), '.2f')} "
            f"max={max((int(v) for v in candidate_counts), default=0)}",
            flush=True,
        )
        print(
            f"gt_template_coverage rate={format_metric(_mean([float(v) for v in coverage_hits]), '.4f')} "
            f"count={sum(coverage_hits)}/{len(coverage_hits)}",
            flush=True,
        )
        print(
            f"torch_expander_match rate={format_metric(_mean([float(v) for v in expander_hits]), '.4f')} "
            f"count={sum(expander_hits)}/{len(expander_hits)}",
            flush=True,
        )
        print(
            f"torch_expander_rms mean={format_metric(_mean(expander_rmses), '.6f')} "
            f"std={format_metric(float(np.std(expander_rmses)) if expander_rmses else None, '.6f')}",
            flush=True,
        )

        misses = [item for item in records if not item["covered"]][:5]
        if misses:
            print("top_template_misses", flush=True)
            for item in misses:
                print(
                    f"structure_id={item['structure_id']} dataset_sg={item['dataset_sg']} gt_sg={item['gt_sg']} "
                    f"std_sg={item['standardized_sg']} num_templates={item['num_templates']} status={item['status']}",
                    flush=True,
                )


def main() -> None:
    args = parse_args()
    WyckoffTemplateSanityRunner(config_path=args.config).run()


if __name__ == "__main__":
    main()
