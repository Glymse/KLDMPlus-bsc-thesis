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

from kldmPlus.kldm import _symmop_list_to_tensor_ops
from kldmPlus.sample_evaluation.sample_evaluation import build_structure_from_sample
from kldmPlus.utils.device import get_default_device

try:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:  # pragma: no cover
    SpacegroupAnalyzer = None


TEST_SPLIT = "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run symmetry basis-alignment sanity checks with pymatgen.")
    parser.add_argument("--config", required=True, help="Path to the basis sanity YAML file.")
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


def _require_pymatgen() -> None:
    if SpacegroupAnalyzer is None:
        raise ImportError("run_symmetry_basis_sanity requires pymatgen.")


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


class SymmetryBasisSanityRunner:
    def __init__(self, config_path: str | Path) -> None:
        from kldmPlus.utils.model_loader import build_model

        _require_pymatgen()
        self.config_path, self.config = load_config(config_path)
        self.experiment_name = str(self.config["experiment_name"])
        self.sanity_cfg = dict(self.config["symmetry_basis_sanity"])
        self.device = get_default_device()

        self.loader, self.lattice_transform = self._build_loader()
        self._inject_mattergen_lattice_stats()
        self.model = build_model(config=self.config, device=self.device)
        self.model.eval()

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

        requested_split = str(self.sanity_cfg.get("split", TEST_SPLIT))
        if requested_split != TEST_SPLIT:
            raise ValueError(f"run_symmetry_basis_sanity always uses split={TEST_SPLIT!r}, got {requested_split!r}")

        root = resolve_data_root(dataset_cfg.get("root"))
        dataset_full = task.fit_dataset(root=root, split=TEST_SPLIT, download=True)
        dataset = make_fixed_subset(
            dataset_full,
            subset_size=int(self.sanity_cfg["num_targets"]),
            seed=int(self.sanity_cfg["subset_seed"]),
        )

        loader = DataLoader(
            dataset,
            batch_size=int(self.sanity_cfg.get("batch_size", 1)),
            shuffle=False,
            num_workers=int(dataset_cfg.get("num_workers", 0)),
            pin_memory=bool(dataset_cfg.get("pin_memory", False)),
            collate_fn=dataset_full.collate_fn,
        )

        lattice_transform = task.make_lattice_transform(
            root=root,
            download=True,
            mattergen_limit_var_scaling_constant=model_cfg.get("mattergen_limit_var_scaling_constant"),
        )
        return loader, lattice_transform

    def _inject_mattergen_lattice_stats(self) -> None:
        if getattr(self.lattice_transform, "representation", None) != "mattergen":
            return
        if not hasattr(self.lattice_transform, "stats"):
            return
        c, nu = self.lattice_transform.stats()
        self.config.setdefault("model", {})
        self.config["model"]["mattergen_lattice_c"] = float(c)
        self.config["model"]["mattergen_lattice_nu"] = float(nu)

    def _perturb_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        pos_noise_scale = float(self.sanity_cfg.get("pos_noise_scale", 0.02))
        lattice_noise_scale = float(self.sanity_cfg.get("lattice_noise_scale", 0.02))

        perturbed_pos = batch.pos
        perturbed_l = batch.l

        if pos_noise_scale > 0.0:
            perturbed_pos = self.model.tdm.wrap_positions(batch.pos + pos_noise_scale * torch.randn_like(batch.pos))

        if lattice_noise_scale > 0.0:
            perturbed_l = batch.l + lattice_noise_scale * torch.randn_like(batch.l)

        return perturbed_pos, perturbed_l

    def _standardized_structure(self, structure):
        standardization = str(self.sanity_cfg.get("standardization", "conventional"))
        symprec = float(self.sanity_cfg.get("symprec", 1e-2))
        angle_tolerance = float(self.sanity_cfg.get("angle_tolerance", 5.0))
        analyzer = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)

        if standardization == "conventional":
            standardized = analyzer.get_conventional_standard_structure()
        elif standardization == "primitive":
            standardized = analyzer.get_primitive_standard_structure()
        elif standardization == "refined":
            standardized = analyzer.get_refined_structure()
        else:
            raise ValueError(f"Unknown standardization={standardization!r}")

        return analyzer, standardized

    def _aligned_ops_for_gt_basis(
        self,
        *,
        gt_structure,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        symprec = float(self.sanity_cfg.get("symprec", 1e-2))
        angle_tolerance = float(self.sanity_cfg.get("angle_tolerance", 5.0))
        analyzer_gt, standardized = self._standardized_structure(gt_structure)
        # Important note:
        # SpacegroupAnalyzer(gt_structure).get_symmetry_operations(cartesian=False)
        # already returns the detected symmetry operations in the fractional
        # basis of the GT structure. For the "does basis alignment fix the
        # energy?" sanity check, this is the most direct and stable object to
        # use. We still build the standardized structure for reporting, but we
        # do not force a StructureMatcher supercell alignment here.
        aligned_ops = analyzer_gt.get_symmetry_operations(cartesian=False)
        tensor_ops = _symmop_list_to_tensor_ops(
            aligned_ops,
            device=device,
            dtype=dtype,
        )

        return {
            "dataset_detected_sg": int(analyzer_gt.get_space_group_number()),
            "standardized_detected_sg": int(SpacegroupAnalyzer(
                standardized,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            ).get_space_group_number()),
            "aligned_detected_sg": int(analyzer_gt.get_space_group_number()),
            "aligned_ops": tensor_ops,
            "standardized_num_sites": int(len(standardized)),
            "aligned_num_sites": int(len(gt_structure)),
        }

    def run(self) -> None:
        sample_seed = int(self.sanity_cfg.get("sample_seed", 0))
        set_seed(sample_seed)

        print(f"experiment={self.experiment_name}", flush=True)
        print(
            f"subset split={TEST_SPLIT} num_targets={int(self.sanity_cfg['num_targets'])} "
            f"subset_seed={int(self.sanity_cfg['subset_seed'])} sample_seed={sample_seed}",
            flush=True,
        )

        default_gt_total: list[float] = []
        aligned_gt_total: list[float] = []
        aligned_pert_total: list[float] = []
        aligned_gt_lt_pert: list[int] = []
        dataset_sg_matches_gt_detected: list[int] = []
        records: list[dict[str, Any]] = []

        for batch in self.loader:
            batch = batch.to(self.device)
            gt_structure = build_structure_from_sample(
                f=batch.pos,
                l=batch.l[0],
                a=batch.atomic_numbers,
                lattice_transform=self.lattice_transform,
            )

            aligned_meta = self._aligned_ops_for_gt_basis(
                gt_structure=gt_structure,
                device=batch.pos.device,
                dtype=batch.pos.dtype,
            )

            default_energy = self.model.symmetry_guidance_energy(
                batch=batch,
                f_t=batch.pos,
                l_t=batch.l,
                lattice_transform=self.lattice_transform,
            )
            aligned_energy = self.model.symmetry_guidance_energy(
                batch=batch,
                f_t=batch.pos,
                l_t=batch.l,
                lattice_transform=self.lattice_transform,
                operations_by_graph=[aligned_meta["aligned_ops"]],
            )

            perturbed_pos, perturbed_l = self._perturb_batch(batch)
            aligned_perturbed_energy = self.model.symmetry_guidance_energy(
                batch=batch,
                f_t=perturbed_pos,
                l_t=perturbed_l,
                lattice_transform=self.lattice_transform,
                operations_by_graph=[aligned_meta["aligned_ops"]],
            )

            dataset_sg = int(torch.as_tensor(batch.space_group).reshape(-1)[0].item())
            gt_detected_sg = int(aligned_meta["dataset_detected_sg"])
            default_gt_total_value = float(default_energy["total"].item())
            aligned_gt_total_value = float(aligned_energy["total"].item())
            aligned_pert_total_value = float(aligned_perturbed_energy["total"].item())

            default_gt_total.append(default_gt_total_value)
            aligned_gt_total.append(aligned_gt_total_value)
            aligned_pert_total.append(aligned_pert_total_value)
            aligned_gt_lt_pert.append(int(aligned_gt_total_value < aligned_pert_total_value))
            dataset_sg_matches_gt_detected.append(int(dataset_sg == gt_detected_sg))

            records.append(
                {
                    "structure_id": _structure_id_from_batch(batch),
                    "dataset_sg": dataset_sg,
                    "gt_detected_sg": gt_detected_sg,
                    "standardized_detected_sg": int(aligned_meta["standardized_detected_sg"]),
                    "aligned_detected_sg": int(aligned_meta["aligned_detected_sg"]),
                    "default_gt_total": default_gt_total_value,
                    "aligned_gt_total": aligned_gt_total_value,
                    "aligned_pert_total": aligned_pert_total_value,
                    "standardized_num_sites": int(aligned_meta["standardized_num_sites"]),
                    "aligned_num_sites": int(aligned_meta["aligned_num_sites"]),
                }
            )

        def _mean(values: list[float]) -> float | None:
            return None if not values else float(np.mean(values))

        print(
            f"space_group_match dataset_vs_detected={format_metric(_mean([float(v) for v in dataset_sg_matches_gt_detected]), '.4f')} "
            f"count={sum(dataset_sg_matches_gt_detected)}/{len(dataset_sg_matches_gt_detected)}",
            flush=True,
        )
        print(
            f"default_ground_truth total_mean={format_metric(_mean(default_gt_total), '.6f')}",
            flush=True,
        )
        print(
            f"aligned_ground_truth total_mean={format_metric(_mean(aligned_gt_total), '.6f')}",
            flush=True,
        )
        print(
            f"aligned_perturbed total_mean={format_metric(_mean(aligned_pert_total), '.6f')}",
            flush=True,
        )
        print(
            f"ordering aligned_ground_truth_lt_perturbed={format_metric(_mean([float(v) for v in aligned_gt_lt_pert]), '.4f')} "
            f"count={sum(aligned_gt_lt_pert)}/{len(aligned_gt_lt_pert)}",
            flush=True,
        )

        worst_examples = sorted(records, key=lambda item: item["aligned_gt_total"], reverse=True)[:5]
        print("top_aligned_ground_truth_total_examples", flush=True)
        for item in worst_examples:
            print(
                f"structure_id={item['structure_id']} dataset_sg={item['dataset_sg']} "
                f"gt_detected_sg={item['gt_detected_sg']} std_detected_sg={item['standardized_detected_sg']} "
                f"aligned_detected_sg={item['aligned_detected_sg']} default_gt_total={item['default_gt_total']:.6f} "
                f"aligned_gt_total={item['aligned_gt_total']:.6f} aligned_pert_total={item['aligned_pert_total']:.6f} "
                f"std_sites={item['standardized_num_sites']} aligned_sites={item['aligned_num_sites']}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    SymmetryBasisSanityRunner(config_path=args.config).run()


if __name__ == "__main__":
    main()
