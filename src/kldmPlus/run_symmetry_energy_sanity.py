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

from kldmPlus.utils.device import get_default_device


TEST_SPLIT = "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run symmetry-energy sanity checks on ground-truth structures.")
    parser.add_argument("--config", required=True, help="Path to the sanity YAML file.")
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


class SymmetryEnergySanityRunner:
    def __init__(self, config_path: str | Path) -> None:
        from kldmPlus.utils.model_loader import build_model

        self.config_path, self.config = load_config(config_path)
        self.experiment_name = str(self.config["experiment_name"])
        self.sanity_cfg = dict(self.config["symmetry_sanity"])
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
            raise ValueError(f"run_symmetry_energy_sanity always uses split={TEST_SPLIT!r}, got {requested_split!r}")

        root = resolve_data_root(dataset_cfg.get("root"))
        dataset_full = task.fit_dataset(root=root, split=TEST_SPLIT, download=True)
        dataset = make_fixed_subset(
            dataset_full,
            subset_size=int(self.sanity_cfg["num_targets"]),
            seed=int(self.sanity_cfg["subset_seed"]),
        )

        batch_size = int(self.sanity_cfg.get("batch_size", 1))
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

    def run(self) -> None:
        sample_seed = int(self.sanity_cfg.get("sample_seed", 0))
        set_seed(sample_seed)

        print(f"experiment={self.experiment_name}", flush=True)
        print(
            f"subset split={TEST_SPLIT} num_targets={int(self.sanity_cfg['num_targets'])} "
            f"subset_seed={int(self.sanity_cfg['subset_seed'])} sample_seed={sample_seed}",
            flush=True,
        )
        print(
            f"noise pos_scale={float(self.sanity_cfg.get('pos_noise_scale', 0.02)):.6f} "
            f"lattice_scale={float(self.sanity_cfg.get('lattice_noise_scale', 0.02)):.6f}",
            flush=True,
        )

        gt_coord: list[float] = []
        gt_lattice: list[float] = []
        gt_total: list[float] = []
        pert_coord: list[float] = []
        pert_lattice: list[float] = []
        pert_total: list[float] = []
        gt_lt_pert: list[int] = []
        records: list[dict[str, Any]] = []

        for batch in self.loader:
            batch = batch.to(self.device)
            gt_energy = self.model.symmetry_guidance_energy(
                batch=batch,
                f_t=batch.pos,
                l_t=batch.l,
                lattice_transform=self.lattice_transform,
            )

            perturbed_pos, perturbed_l = self._perturb_batch(batch)
            pert_energy = self.model.symmetry_guidance_energy(
                batch=batch,
                f_t=perturbed_pos,
                l_t=perturbed_l,
                lattice_transform=self.lattice_transform,
            )

            gt_total_value = float(gt_energy["total"].item())
            pert_total_value = float(pert_energy["total"].item())
            gt_coord_value = float(gt_energy["coord"].item())
            gt_lattice_value = float(gt_energy["lattice"].item())
            pert_coord_value = float(pert_energy["coord"].item())
            pert_lattice_value = float(pert_energy["lattice"].item())

            gt_coord.append(gt_coord_value)
            gt_lattice.append(gt_lattice_value)
            gt_total.append(gt_total_value)
            pert_coord.append(pert_coord_value)
            pert_lattice.append(pert_lattice_value)
            pert_total.append(pert_total_value)
            gt_lt_pert.append(int(gt_total_value < pert_total_value))

            structure_id = str(batch.structure_id[0]) if hasattr(batch, "structure_id") else "unknown"
            space_group = int(torch.as_tensor(batch.space_group).reshape(-1)[0].item())
            records.append(
                {
                    "structure_id": structure_id,
                    "space_group": space_group,
                    "gt_total": gt_total_value,
                    "pert_total": pert_total_value,
                    "gt_coord": gt_coord_value,
                    "gt_lattice": gt_lattice_value,
                }
            )

        def _mean(values: list[float]) -> float | None:
            return None if not values else float(np.mean(values))

        def _std(values: list[float]) -> float | None:
            return None if not values else float(np.std(values))

        print(
            f"ground_truth coord_mean={format_metric(_mean(gt_coord), '.6f')} "
            f"coord_std={format_metric(_std(gt_coord), '.6f')} "
            f"lattice_mean={format_metric(_mean(gt_lattice), '.6f')} "
            f"lattice_std={format_metric(_std(gt_lattice), '.6f')} "
            f"total_mean={format_metric(_mean(gt_total), '.6f')} "
            f"total_std={format_metric(_std(gt_total), '.6f')}",
            flush=True,
        )
        print(
            f"perturbed coord_mean={format_metric(_mean(pert_coord), '.6f')} "
            f"coord_std={format_metric(_std(pert_coord), '.6f')} "
            f"lattice_mean={format_metric(_mean(pert_lattice), '.6f')} "
            f"lattice_std={format_metric(_std(pert_lattice), '.6f')} "
            f"total_mean={format_metric(_mean(pert_total), '.6f')} "
            f"total_std={format_metric(_std(pert_total), '.6f')}",
            flush=True,
        )
        print(
            f"ordering ground_truth_lt_perturbed={format_metric(_mean([float(v) for v in gt_lt_pert]), '.4f')} "
            f"count={sum(gt_lt_pert)}/{len(gt_lt_pert)}",
            flush=True,
        )

        worst_examples = sorted(records, key=lambda item: item["gt_total"], reverse=True)[:5]
        print("top_ground_truth_total_examples", flush=True)
        for item in worst_examples:
            print(
                f"structure_id={item['structure_id']} space_group={item['space_group']} "
                f"gt_total={item['gt_total']:.6f} pert_total={item['pert_total']:.6f} "
                f"gt_coord={item['gt_coord']:.6f} gt_lattice={item['gt_lattice']:.6f}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    SymmetryEnergySanityRunner(config_path=args.config).run()


if __name__ == "__main__":
    main()
