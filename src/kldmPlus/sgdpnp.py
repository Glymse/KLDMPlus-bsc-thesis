from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from kldmPlus.data.transform import (
    KLDMContinuousIntervalLattice,
    MatterGenContinuousIntervalLattice,
    lattice_feature_components,
    mattergen_lattice_feature_vector,
)
from kldmPlus.symmetry.k_basis import (
    cell_to_k,
    free_vars_to_k,
    k_to_cell_matrix,
    k_to_free_vars,
    space_group_k_constraint,
)
from kldmPlus.symmetry.pcs_projection import validate_requested_space_group
from kldmPlus.symmetry.template_cache import get_cache_entry, load_template_cache
from kldmPlus.symmetry.template_prior import TemplatePrior, template_prior_score
from kldmPlus.symmetry.template_ranker import load_template_ranker, score_templates
from kldmPlus.symmetry.wyckoff_templates import (
    WyckoffTemplate,
    composition_to_species_counts,
    expand_wyckoff_template_torch,
    extract_wyckoff_templates,
    flatten_site_signature,
    requested_conventional_atomic_numbers,
    sample_random_free_vars,
)

try:
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:  # pragma: no cover
    Lattice = Structure = SpacegroupAnalyzer = None


@dataclass(frozen=True)
class SGDPnPConfig:
    outer_steps: int = 2
    eta_start: float = 0.03
    eta_end: float = 0.015
    final_eta_scale: float = 0.25
    max_templates: int = 256
    template_eval_limit: int = 64
    template_num_sites_min: int | None = None
    template_num_sites_max: int | None = None
    quick_templates: bool = False
    mala_steps: int = 48
    mala_step_size: float = 5.0e-5
    refine_steps: int = 0
    refine_top_k: int = 8
    template_restarts: int = 4
    refine_lr: float = 1.0e-2
    branch_temperature: float = 0.1
    coord_weight: float = 1.0
    lattice_weight: float = 0.05
    steric_weight: float = 10.0
    min_distance: float = 1.0
    volume_weight: float = 0.01
    template_prior_weight: float = 1.0
    template_ranker_path: str | None = None
    template_ranker_weight: float = 0.0
    template_cache_path: str | None = None
    template_cache_required: bool = True
    dds_steps: int = 30
    dds_t_final: float = 5.0e-4
    symprec: float = 1.0e-2
    angle_tolerance: float = 5.0
    initial_pcs: bool = True
    debug: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "SGDPnPConfig":
        if not payload:
            return cls()
        fields = cls.__dataclass_fields__
        values = {key: payload[key] for key in fields if key in payload}
        return cls(**values)


@dataclass
class _Branch:
    template: WyckoffTemplate
    free_vars: torch.Tensor
    lattice_free_vars: torch.Tensor
    frac_coords: torch.Tensor
    atomic_numbers: torch.Tensor
    cell_matrix: torch.Tensor
    energy: float
    min_pair_distance: float
    detected_sg: int | None
    valid: bool
    reason: str
    ranker_score: float = 0.0
    final_energy: float | None = None


_TEMPLATE_CACHE: dict[tuple[int, tuple[int, ...], int, int, int, bool], list[WyckoffTemplate]] = {}
_RANKER_CACHE: dict[tuple[str, str], torch.nn.Module] = {}
_DISK_TEMPLATE_CACHE: dict[str, dict[str, Any]] = {}


def _maybe_load_template_ranker(
    *,
    path: str | None,
    device: torch.device,
) -> torch.nn.Module | None:
    if path is None or str(path).strip() == "":
        return None
    resolved = str(Path(path).expanduser().resolve())
    key = (resolved, str(device))
    cached = _RANKER_CACHE.get(key)
    if cached is None:
        cached = load_template_ranker(resolved, device=device)
        _RANKER_CACHE[key] = cached
        print(f"kldm_dpnp_sg_ranker_load path={resolved}", flush=True)
    return cached


def _maybe_load_disk_template_cache(path: str | None) -> dict[str, Any] | None:
    if path is None or str(path).strip() == "":
        return None
    resolved = str(Path(path).expanduser().resolve())
    cached = _DISK_TEMPLATE_CACHE.get(resolved)
    if cached is None:
        cached = load_template_cache(resolved)
        _DISK_TEMPLATE_CACHE[resolved] = cached
        print(
            f"kldm_dpnp_sg_template_cache_load path={resolved} entries={len(cached.get('entries', {}))}",
            flush=True,
        )
    return cached


def _selection_energy(branch: _Branch) -> float:
    if branch.final_energy is None:
        return float(branch.energy)
    return float(branch.final_energy)


def _set_branch_selection_energy(
    branch: _Branch,
    *,
    cfg: SGDPnPConfig,
    ranker_score: float,
) -> None:
    branch.ranker_score = float(ranker_score)
    branch.final_energy = float(branch.energy) - float(cfg.template_ranker_weight) * float(ranker_score)


def _cached_templates(
    *,
    space_group_number: int,
    atomic_numbers: torch.Tensor,
    max_templates: int,
    num_sites: tuple[int | None, int | None],
    quick: bool,
    template_cache: dict[str, Any] | None = None,
    template_cache_required: bool = True,
) -> list[WyckoffTemplate]:
    atomic_key = tuple(int(v) for v in atomic_numbers.detach().cpu().reshape(-1).tolist())
    entry = get_cache_entry(
        template_cache,
        space_group_number=int(space_group_number),
        atomic_numbers=list(atomic_key),
    )
    if entry is not None:
        templates = list(entry.get("templates", []))
        min_sites, max_sites = num_sites
        if min_sites is not None:
            templates = [template for template in templates if template.total_sites >= int(min_sites)]
        if max_sites is not None:
            templates = [template for template in templates if template.total_sites <= int(max_sites)]
        return templates[: int(max_templates)]
    if template_cache is not None and bool(template_cache_required):
        return []
    key = (
        int(space_group_number),
        atomic_key,
        int(max_templates),
        int(num_sites[0] or -1),
        int(num_sites[1] or -1),
        bool(quick),
    )
    cached = _TEMPLATE_CACHE.get(key)
    if cached is None:
        cached = extract_wyckoff_templates(
            space_group_number=int(space_group_number),
            atomic_numbers=list(atomic_key),
            max_templates=int(max_templates),
            quick=bool(quick),
            num_wp=num_sites,
        )
        _TEMPLATE_CACHE[key] = cached
    return cached


def _lengths_angles_to_cell_matrix(lengths: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    a, b, c = lengths.unbind(dim=-1)
    alpha, beta, gamma = angles.unbind(dim=-1)
    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma).clamp_min(1.0e-8)
    zeros = torch.zeros_like(a)
    row_a = torch.stack([a, zeros, zeros], dim=-1)
    row_b = torch.stack([b * cos_gamma, b * sin_gamma, zeros], dim=-1)
    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz = torch.sqrt((c.square() - cx.square() - cy.square()).clamp_min(1.0e-8))
    row_c = torch.stack([cx, cy, cz], dim=-1)
    return torch.stack([row_a, row_b, row_c], dim=-2)


def _decode_lattice_matrix(
    *,
    l: torch.Tensor,
    num_atoms: int,
    lattice_transform: Any,
) -> torch.Tensor:
    if lattice_transform is not None and hasattr(lattice_transform, "invert_to_matrix"):
        return lattice_transform.invert_to_matrix(l.view(1, -1), num_atoms=num_atoms).reshape(3, 3)
    if lattice_transform is not None:
        lengths, angles = lattice_transform.invert_to_lengths_angles(l.view(1, -1), num_atoms=num_atoms)
    else:
        lengths = torch.exp(l.view(1, -1)[..., :3])
        angles = torch.atan(l.view(1, -1)[..., 3:]) + torch.pi / 2.0
    return _lengths_angles_to_cell_matrix(lengths.squeeze(0), angles.squeeze(0))


def _encode_lattice_features(
    *,
    cell_matrix: torch.Tensor,
    num_atoms: int,
    lattice_transform: Any,
) -> torch.Tensor:
    if isinstance(lattice_transform, MatterGenContinuousIntervalLattice):
        return mattergen_lattice_feature_vector(cell_matrix).view(1, 6)
    if isinstance(lattice_transform, KLDMContinuousIntervalLattice):
        log_lengths, angle_features = lattice_feature_components(cell_matrix, eps=lattice_transform.eps)
        if lattice_transform.standardize and lattice_transform.lengths_loc_scale is not None:
            log_lengths, angle_features = lattice_transform._encode_x0_parts(
                log_lengths=log_lengths,
                angle_features=angle_features,
                num_atoms=int(num_atoms),
            )
            return torch.cat([log_lengths, angle_features], dim=0).view(1, 6)
        features = torch.cat([log_lengths, angle_features], dim=0)
        return lattice_transform.standardize_value(features).view(1, 6)
    log_lengths, angle_features = lattice_feature_components(cell_matrix)
    return torch.cat([log_lengths, angle_features], dim=0).view(1, 6)


def _structure_from_tensors(frac: torch.Tensor, atomic_numbers: torch.Tensor, cell: torch.Tensor):
    if Structure is None or Lattice is None:
        raise ImportError("KLDM-DPnP-SG validation requires pymatgen.")
    return Structure(
        lattice=Lattice(cell.detach().cpu().numpy()),
        species=[int(v) for v in atomic_numbers.detach().cpu().tolist()],
        coords=np.asarray(frac.detach().cpu().numpy(), dtype=float),
        coords_are_cartesian=False,
        to_unit_cell=True,
    )


def _branch_with_primitive_count(
    branch: _Branch,
    *,
    anchor_count: int,
    symprec: float,
    angle_tolerance: float,
) -> _Branch | None:
    if int(branch.frac_coords.shape[0]) == int(anchor_count):
        return branch
    if Structure is None or Lattice is None or SpacegroupAnalyzer is None:
        return None
    try:
        structure = _structure_from_tensors(branch.frac_coords, branch.atomic_numbers, branch.cell_matrix)
        analyzer = SpacegroupAnalyzer(
            structure,
            symprec=float(symprec),
            angle_tolerance=float(angle_tolerance),
        )
        primitive = analyzer.get_primitive_standard_structure(international_monoclinic=False)
    except Exception:
        return None
    if len(primitive) != int(anchor_count):
        return None
    frac = torch.as_tensor(
        np.array(primitive.frac_coords, dtype=float, copy=True),
        device=branch.frac_coords.device,
        dtype=branch.frac_coords.dtype,
    )
    species = torch.as_tensor(
        [int(site.specie.Z) for site in primitive],
        device=branch.atomic_numbers.device,
        dtype=branch.atomic_numbers.dtype,
    )
    cell = torch.as_tensor(
        np.array(primitive.lattice.matrix, dtype=float, copy=True),
        device=branch.cell_matrix.device,
        dtype=branch.cell_matrix.dtype,
    )
    _steric_loss, min_pair = _steric_energy(frac, cell, 0.0)
    return _Branch(
        template=branch.template,
        free_vars=branch.free_vars,
        lattice_free_vars=branch.lattice_free_vars,
        frac_coords=torch.remainder(frac, 1.0),
        atomic_numbers=species,
        cell_matrix=cell,
        energy=branch.energy,
        min_pair_distance=float(min_pair),
        detected_sg=branch.detected_sg,
        valid=branch.valid,
        reason=branch.reason + ";primitive_reduced",
        ranker_score=branch.ranker_score,
        final_energy=branch.final_energy,
    )


def _periodic_distance_sq(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    delta = source.unsqueeze(1) - target.unsqueeze(0)
    delta = delta - torch.round(delta)
    return delta.square().sum(dim=-1)


def _match_cost_matrix(cost_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    detached = cost_matrix.detach()
    if not torch.isfinite(detached).all():
        raise RuntimeError("matrix contains invalid numeric entries")
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # pragma: no cover
        linear_sum_assignment = None

    if linear_sum_assignment is not None:
        row_idx, col_idx = linear_sum_assignment(detached.cpu().numpy())
        return (
            torch.as_tensor(row_idx, device=cost_matrix.device, dtype=torch.long),
            torch.as_tensor(col_idx, device=cost_matrix.device, dtype=torch.long),
        )

    remaining_rows = list(range(detached.shape[0]))
    remaining_cols = list(range(detached.shape[1]))
    chosen_rows: list[int] = []
    chosen_cols: list[int] = []
    detached_cpu = detached.cpu()
    while remaining_rows:
        submatrix = detached_cpu[remaining_rows][:, remaining_cols]
        flat_index = int(torch.argmin(submatrix).item())
        sub_cols = submatrix.shape[1]
        row_pos = flat_index // sub_cols
        col_pos = flat_index % sub_cols
        chosen_rows.append(remaining_rows.pop(row_pos))
        chosen_cols.append(remaining_cols.pop(col_pos))

    row_idx = torch.tensor(chosen_rows, device=cost_matrix.device, dtype=torch.long)
    col_idx = torch.tensor(chosen_cols, device=cost_matrix.device, dtype=torch.long)
    order = torch.argsort(row_idx)
    return row_idx[order], col_idx[order]


def _species_hungarian_torus_loss(
    *,
    candidate_frac: torch.Tensor,
    candidate_species: torch.Tensor,
    anchor_frac: torch.Tensor,
    anchor_species: torch.Tensor,
) -> torch.Tensor:
    if not torch.isfinite(candidate_frac).all() or not torch.isfinite(anchor_frac).all():
        return candidate_frac.new_tensor(float("inf"))
    candidate_values = sorted(int(v) for v in candidate_species.detach().cpu().tolist())
    anchor_values = sorted(int(v) for v in anchor_species.detach().cpu().tolist())
    if candidate_values != anchor_values:
        return candidate_frac.new_tensor(float("inf"))

    loss = candidate_frac.new_zeros(())
    species_count = 0
    for species in torch.unique(candidate_species.detach(), sorted=True).tolist():
        src = candidate_frac[candidate_species == int(species)]
        tgt = anchor_frac[anchor_species == int(species)]
        if src.shape[0] != tgt.shape[0]:
            return candidate_frac.new_tensor(float("inf"))
        if src.numel() == 0:
            continue
        distances = _periodic_distance_sq(src, tgt)
        if not torch.isfinite(distances).all():
            return candidate_frac.new_tensor(float("inf"))
        row_idx, col_idx = _match_cost_matrix(distances)
        delta = src[row_idx] - tgt[col_idx]
        delta = delta - torch.round(delta)
        loss = loss + delta.square().mean()
        species_count += 1
    return loss / max(species_count, 1)


def _centering_symbol(group_symbol: str | None) -> str:
    if not group_symbol:
        return "P"
    stripped = str(group_symbol).strip().upper()
    return stripped[0] if stripped else "P"


def _centering_factor(group_symbol: str | None) -> int:
    symbol = _centering_symbol(group_symbol)
    if symbol in {"A", "B", "C", "I"}:
        return 2
    if symbol == "F":
        return 4
    if symbol == "R":
        return 3
    return 1


def _centering_translations_from_symbol(
    group_symbol: str | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    symbol = _centering_symbol(group_symbol)
    if symbol == "I":
        values = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    elif symbol == "F":
        values = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    elif symbol == "A":
        values = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5]]
    elif symbol == "B":
        values = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]
    elif symbol == "C":
        values = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
    elif symbol == "R":
        values = [[0.0, 0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]]
    else:
        values = [[0.0, 0.0, 0.0]]
    return torch.as_tensor(values, device=device, dtype=dtype)


def _expand_anchor_to_species_multiset(
    *,
    anchor_frac: torch.Tensor,
    anchor_species: torch.Tensor,
    target_species: torch.Tensor,
    group_symbol: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sorted(int(v) for v in anchor_species.detach().cpu().tolist()) == sorted(
        int(v) for v in target_species.detach().cpu().tolist()
    ):
        return anchor_frac, anchor_species

    frac_blocks: list[torch.Tensor] = []
    species_blocks: list[torch.Tensor] = []
    for species in torch.unique(anchor_species.detach(), sorted=True).tolist():
        source = anchor_frac[anchor_species == int(species)]
        source_count = int(source.shape[0])
        target_count = int((target_species == int(species)).sum().item())
        if source_count <= 0 or target_count % source_count != 0:
            return anchor_frac, anchor_species
        factor = target_count // source_count
        translations = _centering_translations_from_symbol(
            group_symbol,
            device=anchor_frac.device,
            dtype=anchor_frac.dtype,
        )
        if int(translations.shape[0]) != int(factor):
            return anchor_frac, anchor_species
        expanded = torch.remainder(source.unsqueeze(0) + translations.unsqueeze(1), 1.0).reshape(-1, 3)
        frac_blocks.append(expanded)
        species_blocks.append(torch.full((expanded.shape[0],), int(species), device=anchor_species.device, dtype=anchor_species.dtype))
    return torch.cat(frac_blocks, dim=0), torch.cat(species_blocks, dim=0)


def _periodic_pair_distances(frac: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    n_atoms = int(frac.shape[0])
    if n_atoms < 2:
        return frac.new_empty((0,))
    i, j = torch.triu_indices(n_atoms, n_atoms, offset=1, device=frac.device)
    delta = frac[i] - frac[j]
    delta = delta - torch.round(delta)
    cart = delta @ cell
    return torch.linalg.norm(cart, dim=-1)


def _steric_energy(frac: torch.Tensor, cell: torch.Tensor, min_distance: float) -> tuple[torch.Tensor, float]:
    distances = _periodic_pair_distances(frac, cell)
    if distances.numel() == 0:
        return frac.new_zeros(()), float("inf")
    min_pair = float(distances.detach().min().cpu().item())
    deficit = torch.nn.functional.softplus(frac.new_tensor(float(min_distance)) - distances)
    return deficit.square().mean(), min_pair


def _energy(
    *,
    theta: torch.Tensor,
    template: WyckoffTemplate,
    free_dim: int,
    constraint: Any,
    anchor_frac: torch.Tensor,
    anchor_species: torch.Tensor,
    anchor_k: torch.Tensor,
    anchor_volume: torch.Tensor,
    eta: float,
    cfg: SGDPnPConfig,
    template_prior_count: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    free_vars = torch.remainder(theta[:free_dim], 1.0)
    lattice_free_vars = theta[free_dim:]
    expansion = expand_wyckoff_template_torch(template=template, free_vars=free_vars)
    k = free_vars_to_k(lattice_free_vars, constraint)
    cell = k_to_cell_matrix(k)
    coord_loss = _species_hungarian_torus_loss(
        candidate_frac=expansion.frac_coords,
        candidate_species=expansion.atomic_numbers,
        anchor_frac=anchor_frac,
        anchor_species=anchor_species,
    )
    anchor_lattice_free_vars = k_to_free_vars(anchor_k, constraint)
    if lattice_free_vars.numel() == 0:
        lattice_loss = lattice_free_vars.new_zeros(())
    else:
        lattice_loss = (lattice_free_vars - anchor_lattice_free_vars).square().mean()
    steric_loss, min_pair = _steric_energy(expansion.frac_coords, cell, cfg.min_distance)
    volume = torch.abs(torch.linalg.det(cell)).clamp_min(1.0e-8)
    volume_loss = torch.log(volume / anchor_volume.clamp_min(1.0e-8)).square()
    prox = (cfg.coord_weight * coord_loss + cfg.lattice_weight * lattice_loss) / (2.0 * max(float(eta), 1.0e-8) ** 2)
    likelihood = cfg.steric_weight * steric_loss + cfg.volume_weight * volume_loss
    prior_bonus = cfg.template_prior_weight * math.log1p(max(int(template_prior_count), 0))
    total = prox + likelihood - expansion.frac_coords.new_tensor(float(prior_bonus))
    return total, expansion.frac_coords, expansion.atomic_numbers, cell, min_pair


def _branch_model_energy(
    *,
    branch: _Branch,
    anchor_frac: torch.Tensor,
    anchor_species: torch.Tensor,
    anchor_cell: torch.Tensor,
    requested_sg: int,
    eta: float,
    cfg: SGDPnPConfig,
    template_prior_count: int = 0,
) -> float:
    constraint = space_group_k_constraint(
        space_group_number=int(requested_sg),
        device=branch.frac_coords.device,
        dtype=branch.frac_coords.dtype,
    )
    branch_k = cell_to_k(branch.cell_matrix)
    anchor_k = cell_to_k(anchor_cell.to(device=branch.cell_matrix.device, dtype=branch.cell_matrix.dtype))
    branch_lattice_free_vars = k_to_free_vars(branch_k, constraint)
    anchor_lattice_free_vars = k_to_free_vars(anchor_k, constraint)
    coord_loss = _species_hungarian_torus_loss(
        candidate_frac=branch.frac_coords,
        candidate_species=branch.atomic_numbers,
        anchor_frac=anchor_frac.to(device=branch.frac_coords.device, dtype=branch.frac_coords.dtype),
        anchor_species=anchor_species.to(device=branch.atomic_numbers.device, dtype=branch.atomic_numbers.dtype),
    )
    if branch_lattice_free_vars.numel() == 0:
        lattice_loss = branch.frac_coords.new_zeros(())
    else:
        lattice_loss = (branch_lattice_free_vars - anchor_lattice_free_vars).square().mean()
    steric_loss, _min_pair = _steric_energy(branch.frac_coords, branch.cell_matrix, cfg.min_distance)
    volume = torch.abs(torch.linalg.det(branch.cell_matrix)).clamp_min(1.0e-8)
    anchor_volume = torch.abs(torch.linalg.det(anchor_cell.to(device=branch.cell_matrix.device, dtype=branch.cell_matrix.dtype))).clamp_min(1.0e-8)
    volume_loss = torch.log(volume / anchor_volume).square()
    prox = (cfg.coord_weight * coord_loss + cfg.lattice_weight * lattice_loss) / (2.0 * max(float(eta), 1.0e-8) ** 2)
    likelihood = cfg.steric_weight * steric_loss + cfg.volume_weight * volume_loss
    prior_bonus = cfg.template_prior_weight * math.log1p(max(int(template_prior_count), 0))
    total = prox + likelihood - branch.frac_coords.new_tensor(float(prior_bonus))
    return float(total.detach().cpu().item())


def _mala_template_branch(
    *,
    template: WyckoffTemplate,
    anchor_frac: torch.Tensor,
    anchor_species: torch.Tensor,
    anchor_cell: torch.Tensor,
    eta: float,
    cfg: SGDPnPConfig,
    template_prior_count: int = 0,
    init_free_vars: torch.Tensor | None = None,
    init_lattice_free_vars: torch.Tensor | None = None,
    mala_steps_override: int | None = None,
    refine_steps_override: int | None = None,
) -> _Branch:
    device = anchor_frac.device
    dtype = anchor_frac.dtype
    constraint = space_group_k_constraint(
        space_group_number=int(template.space_group),
        device=device,
        dtype=dtype,
    )
    free_dim = int(template.total_free_dims)
    if init_free_vars is None:
        free_vars = sample_random_free_vars(template, device=device, dtype=dtype)
    else:
        free_vars = torch.remainder(init_free_vars.to(device=device, dtype=dtype), 1.0)
    anchor_k = cell_to_k(anchor_cell)
    anchor_lattice_free_vars = k_to_free_vars(anchor_k, constraint)
    if init_lattice_free_vars is None:
        lattice_free_vars = anchor_lattice_free_vars
    else:
        lattice_free_vars = init_lattice_free_vars.to(device=device, dtype=dtype)
    theta = torch.cat([free_vars, lattice_free_vars], dim=0).detach()
    step = float(cfg.mala_step_size)
    accept = 0
    anchor_volume = torch.abs(torch.linalg.det(anchor_cell)).clamp_min(1.0e-8)

    def energy_and_grad(value: torch.Tensor):
        value = value.detach().clone().requires_grad_(True)
        energy, frac, species, cell, min_pair = _energy(
            theta=value,
            template=template,
            free_dim=free_dim,
            constraint=constraint,
            anchor_frac=anchor_frac,
            anchor_species=anchor_species,
            anchor_k=anchor_k,
            anchor_volume=anchor_volume,
            eta=eta,
            cfg=cfg,
            template_prior_count=template_prior_count,
        )
        grad, = torch.autograd.grad(energy, value)
        grad = torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        return energy.detach(), grad, frac.detach(), species.detach(), cell.detach(), min_pair

    def proposal_mean(value: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        return value - 0.5 * step * grad

    def theta_residual(dst: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        residual = dst - mean
        if free_dim <= 0:
            return residual
        free_residual = residual[:free_dim]
        free_residual = free_residual - torch.round(free_residual)
        return torch.cat([free_residual, residual[free_dim:]], dim=0)

    def log_q(*, src: torch.Tensor, grad_src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        residual = theta_residual(dst, proposal_mean(src, grad_src))
        return -residual.square().sum() / (2.0 * max(step, 1.0e-12))

    mala_steps = int(cfg.mala_steps) if mala_steps_override is None else int(mala_steps_override)
    refine_steps = (
        int(cfg.refine_steps) if refine_steps_override is None else int(refine_steps_override)
    )

    current_energy, current_grad, frac, species, cell, min_pair = energy_and_grad(theta)
    for _ in range(max(mala_steps, 0)):
        proposal = proposal_mean(theta, current_grad) + math.sqrt(max(step, 1.0e-12)) * torch.randn_like(theta)
        if free_dim > 0:
            proposal = torch.cat([torch.remainder(proposal[:free_dim], 1.0), proposal[free_dim:]], dim=0)
        proposal_energy, proposal_grad, proposal_frac, proposal_species, proposal_cell, proposal_min_pair = energy_and_grad(proposal)
        log_alpha = (
            -proposal_energy
            + current_energy
            + log_q(src=proposal, grad_src=proposal_grad, dst=theta)
            - log_q(src=theta, grad_src=current_grad, dst=proposal)
        )
        if bool(torch.log(torch.rand((), device=device, dtype=dtype)) < log_alpha):
            theta = proposal.detach()
            current_energy = proposal_energy
            current_grad = proposal_grad
            frac = proposal_frac
            species = proposal_species
            cell = proposal_cell
            min_pair = proposal_min_pair
            accept += 1

    if refine_steps > 0:
        refined = theta.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([refined], lr=float(cfg.refine_lr))
        best_theta = theta.detach().clone()
        best_energy = current_energy
        best_frac, best_species, best_cell, best_min_pair = frac, species, cell, min_pair
        for _ in range(refine_steps):
            optimizer.zero_grad(set_to_none=True)
            energy, refined_frac, refined_species, refined_cell, refined_min_pair = _energy(
                theta=refined,
                template=template,
                free_dim=free_dim,
                constraint=constraint,
                anchor_frac=anchor_frac,
                anchor_species=anchor_species,
                anchor_k=anchor_k,
                anchor_volume=anchor_volume,
                eta=eta,
                cfg=cfg,
                template_prior_count=template_prior_count,
            )
            energy.backward()
            optimizer.step()
            with torch.no_grad():
                if free_dim > 0:
                    refined.data[:free_dim] = torch.remainder(refined.data[:free_dim], 1.0)
                if bool(torch.isfinite(energy).item()) and float(energy.detach().cpu().item()) < float(best_energy.cpu().item()):
                    best_theta = refined.detach().clone()
                    best_energy = energy.detach()
                    best_frac = refined_frac.detach()
                    best_species = refined_species.detach()
                    best_cell = refined_cell.detach()
                    best_min_pair = refined_min_pair
        theta = best_theta
        current_energy = best_energy
        frac = best_frac
        species = best_species
        cell = best_cell
        min_pair = best_min_pair

    return _Branch(
        template=template,
        free_vars=torch.remainder(theta[:free_dim].detach(), 1.0),
        lattice_free_vars=theta[free_dim:].detach().clone(),
        frac_coords=frac,
        atomic_numbers=species,
        cell_matrix=cell,
        energy=float(current_energy.cpu().item()),
        min_pair_distance=float(min_pair),
        detected_sg=None,
        valid=False,
        reason=f"mala_accept={accept}/{max(int(cfg.mala_steps), 0)}",
    )


def _sample_branch(branches: list[_Branch], temperature: float) -> _Branch:
    if not branches:
        raise ValueError("Cannot sample an empty branch list.")
    if len(branches) == 1:
        return branches[0]
    if float(temperature) <= 0.0:
        return min(branches, key=_selection_energy)
    energies = torch.tensor([_selection_energy(branch) for branch in branches], dtype=torch.float64)
    energies = torch.nan_to_num(energies, nan=1.0e9, posinf=1.0e9, neginf=-1.0e9)
    logits = -(energies - energies.min()) / max(float(temperature), 1.0e-8)
    probs = torch.softmax(logits, dim=0)
    if not bool(torch.isfinite(probs).all().item()) or float(probs.sum().item()) <= 0.0:
        probs = torch.full_like(probs, 1.0 / float(len(branches)))
    return branches[int(torch.multinomial(probs, 1).item())]


def _rank_templates(
    *,
    templates: list[WyckoffTemplate],
    requested_sg: int,
    template_atomic_numbers: torch.Tensor,
    template_prior: TemplatePrior | None,
    template_ranker: torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> list[tuple[WyckoffTemplate, int, float]]:
    species_order, species_counts = composition_to_species_counts(template_atomic_numbers)
    key = (int(requested_sg), species_order, species_counts)
    ranker_scores = score_templates(
        ranker=template_ranker,
        templates=templates,
        requested_sg=int(requested_sg),
        device=device or template_atomic_numbers.device,
    )
    ranked: list[tuple[WyckoffTemplate, int, float, int]] = []
    for original_rank, template in enumerate(templates, start=1):
        count = template_prior_score(
            prior=template_prior,
            key=key,
            signature=flatten_site_signature(template),
        )
        ranker_score = float(ranker_scores[original_rank - 1]) if original_rank - 1 < len(ranker_scores) else 0.0
        ranked.append((template, int(count), ranker_score, int(original_rank)))
    ranked.sort(
        key=lambda item: (
            -float(item[1]),
            -float(item[2]),
            float(item[0].total_free_dims),
            float(item[0].total_sites),
            float(item[0].total_atoms),
            float(item[3]),
        )
    )
    return [(template, count, ranker_score) for template, count, ranker_score, _rank in ranked]

def _branch_template_signature(branch: _Branch) -> tuple[tuple[int, str, int, int], ...]:
    """Species-labeled Wyckoff signature used for diversity-aware refinement."""
    return tuple(
        sorted(
            (
                int(site.atomic_number),
                str(site.label),
                int(site.multiplicity),
                int(site.dof),
            )
            for site in branch.template.site_templates
        )
    )


def _branch_signature_labels(branch: _Branch) -> list[str]:
    return [f"{site.atomic_number}@{site.label}" for site in branch.template.site_templates]

def _pcs_graph(
    *,
    graph_idx: int,
    anchor_frac: torch.Tensor,
    anchor_l: torch.Tensor,
    anchor_species: torch.Tensor,
    requested_sg: int,
    lattice_transform: Any,
    eta: float,
    cfg: SGDPnPConfig,
    template_prior: TemplatePrior | None = None,
    template_ranker: torch.nn.Module | None = None,
    template_cache: dict[str, Any] | None = None,
) -> _Branch | None:
    anchor_cell = _decode_lattice_matrix(
        l=anchor_l,
        num_atoms=int(anchor_frac.shape[0]),
        lattice_transform=lattice_transform,
    ).to(device=anchor_frac.device, dtype=anchor_frac.dtype)

    template_atomic_numbers = requested_conventional_atomic_numbers(
        anchor_species,
        space_group_number=int(requested_sg),
    ).to(device=anchor_species.device, dtype=torch.long)

    templates = _cached_templates(
        space_group_number=int(requested_sg),
        atomic_numbers=template_atomic_numbers,
        max_templates=int(cfg.max_templates),
        num_sites=(cfg.template_num_sites_min, cfg.template_num_sites_max),
        quick=bool(cfg.quick_templates),
        template_cache=template_cache,
        template_cache_required=bool(cfg.template_cache_required),
    )

    group_symbol = templates[0].group_symbol if templates else "P"
    centering_factor = _centering_factor(group_symbol)

    energy_anchor_frac, energy_anchor_species = _expand_anchor_to_species_multiset(
        anchor_frac=anchor_frac,
        anchor_species=anchor_species,
        target_species=template_atomic_numbers,
        group_symbol=group_symbol,
    )

    energy_anchor_cell = anchor_cell * (float(centering_factor) ** (1.0 / 3.0))

    ranked_templates = _rank_templates(
        templates=templates,
        requested_sg=int(requested_sg),
        template_atomic_numbers=template_atomic_numbers,
        template_prior=template_prior,
        template_ranker=template_ranker,
        device=anchor_frac.device,
    )[: max(1, int(cfg.template_eval_limit))]

    raw_candidates: list[tuple[_Branch, int]] = []
    rejected = 0

    # ------------------------------------------------------------
    # Stage 1:
    # Coarse search over templates/restarts with MALA only.
    # No expensive Adam refinement here.
    # ------------------------------------------------------------
    for template, prior_count, ranker_score in ranked_templates:
        for _ in range(max(1, int(cfg.template_restarts))):
            branch = _mala_template_branch(
                template=template,
                anchor_frac=energy_anchor_frac,
                anchor_species=energy_anchor_species,
                anchor_cell=energy_anchor_cell,
                eta=eta,
                cfg=cfg,
                template_prior_count=prior_count,
                refine_steps_override=0,
            )

            branch = _branch_with_primitive_count(
                branch,
                anchor_count=int(anchor_frac.shape[0]),
                symprec=float(cfg.symprec),
                angle_tolerance=float(cfg.angle_tolerance),
            )

            if branch is None:
                rejected += 1
                continue

            branch.energy = _branch_model_energy(
                branch=branch,
                anchor_frac=anchor_frac,
                anchor_species=anchor_species,
                anchor_cell=anchor_cell,
                requested_sg=int(requested_sg),
                eta=eta,
                cfg=cfg,
                template_prior_count=prior_count,
            )
            _set_branch_selection_energy(
                branch,
                cfg=cfg,
                ranker_score=ranker_score,
            )

            if not math.isfinite(float(branch.energy)):
                rejected += 1
                continue

            if branch.min_pair_distance < float(cfg.min_distance):
                rejected += 1
                continue

            raw_candidates.append((branch, prior_count))

    raw_candidates.sort(key=lambda item: _selection_energy(item[0]))

    if not raw_candidates:
        if cfg.debug:
            print(
                f"kldm_dpnp_sg_pcs graph={graph_idx + 1} status=no_raw_candidates "
                f"templates={len(ranked_templates)}/{len(templates)} rejected={rejected}",
                flush=True,
            )
        return None

    # ------------------------------------------------------------
    # Stage 2:
    # Diversity-aware top-k refinement.
    #
    # Instead of refining raw_candidates[:K], refine the best branch
    # from each unique Wyckoff signature first. This prevents one
    # template family from occupying all refined slots.
    # ------------------------------------------------------------
    candidates: list[_Branch] = []

    if int(cfg.refine_steps) <= 0:
        candidates = [branch for branch, _prior_count in raw_candidates]

    else:
        best_per_signature: dict[
            tuple[tuple[int, str, int, int], ...],
            tuple[_Branch, int],
        ] = {}

        for branch, prior_count in raw_candidates:
            signature = _branch_template_signature(branch)
            current = best_per_signature.get(signature)

            if current is None or _selection_energy(branch) < _selection_energy(current[0]):
                best_per_signature[signature] = (branch, prior_count)

        diverse_candidates = list(best_per_signature.values())



        if cfg.debug and int(graph_idx) == 3:
            print(
                f"kldm_dpnp_sg_debug graph=4 top_unique_signatures "
                f"count={len(diverse_candidates)}",
                flush=True,
            )

            for rank, (branch, prior_count) in enumerate(diverse_candidates[:40], start=1):
                print(
                    f"kldm_dpnp_sg_debug graph=4 unique_rank={rank} "
                    f"energy={branch.energy:.6f} final_energy={_selection_energy(branch):.6f} "
                    f"prior={prior_count} ranker={branch.ranker_score:.6f} "
                    f"min_pair={branch.min_pair_distance:.4f} "
                    f"signature={_branch_signature_labels(branch)}",
                    flush=True,
                )

        diverse_candidates.sort(key=lambda item: _selection_energy(item[0]))



        refine_count = min(
            len(diverse_candidates),
            max(1, int(cfg.refine_top_k)),
        )

        to_refine = diverse_candidates[:refine_count]

        if cfg.debug:
            print(
                f"kldm_dpnp_sg_pcs graph={graph_idx + 1} "
                f"raw_candidates={len(raw_candidates)} "
                f"unique_signatures={len(diverse_candidates)} "
                f"refine_count={len(to_refine)}",
                flush=True,
            )

        for branch, prior_count in to_refine:
            refined = _mala_template_branch(
                template=branch.template,
                anchor_frac=energy_anchor_frac,
                anchor_species=energy_anchor_species,
                anchor_cell=energy_anchor_cell,
                eta=eta,
                cfg=cfg,
                template_prior_count=prior_count,
                init_free_vars=branch.free_vars,
                init_lattice_free_vars=branch.lattice_free_vars,
                mala_steps_override=0,
                refine_steps_override=int(cfg.refine_steps),
            )

            refined = _branch_with_primitive_count(
                refined,
                anchor_count=int(anchor_frac.shape[0]),
                symprec=float(cfg.symprec),
                angle_tolerance=float(cfg.angle_tolerance),
            )

            if refined is None:
                rejected += 1
                continue

            refined.energy = _branch_model_energy(
                branch=refined,
                anchor_frac=anchor_frac,
                anchor_species=anchor_species,
                anchor_cell=anchor_cell,
                requested_sg=int(requested_sg),
                eta=eta,
                cfg=cfg,
                template_prior_count=prior_count,
            )
            _set_branch_selection_energy(
                refined,
                cfg=cfg,
                ranker_score=branch.ranker_score,
            )

            if not math.isfinite(float(refined.energy)):
                rejected += 1
                continue

            if refined.min_pair_distance < float(cfg.min_distance):
                rejected += 1
                continue

            candidates.append(refined)

        # Safety fallback:
        # If every refined branch was rejected, fall back to raw candidates.
        if not candidates:
            candidates = [branch for branch, _prior_count in raw_candidates]

    # ------------------------------------------------------------
    # Stage 3:
    # Order candidates. During debugging, branch_temperature=0 gives MAP.
    # ------------------------------------------------------------
    if float(cfg.branch_temperature) > 0.0:
        ordered: list[_Branch] = []
        remaining = candidates[:]

        while remaining:
            selected = _sample_branch(remaining, cfg.branch_temperature)
            ordered.append(selected)
            remaining.remove(selected)
    else:
        ordered = sorted(candidates, key=_selection_energy)

    # ------------------------------------------------------------
    # Stage 4:
    # Validate exact requested SG and composition.
    # Return the first valid candidate in the ordered list.
    # ------------------------------------------------------------
    for branch in ordered:
        try:
            structure = _structure_from_tensors(
                branch.frac_coords,
                branch.atomic_numbers,
                branch.cell_matrix,
            )

            validation = validate_requested_space_group(
                structure=structure,
                requested_space_group=int(requested_sg),
                expected_atomic_numbers=branch.atomic_numbers,
                symprec=float(cfg.symprec),
                angle_tolerance=float(cfg.angle_tolerance),
            )

            branch.detected_sg = int(validation.detected_space_group)

            if not (validation.composition_match and validation.requested_space_group_match):
                rejected += 1
                continue

        except Exception as exc:
            branch.reason = f"validation_failed:{type(exc).__name__}"
            rejected += 1
            continue

        branch.valid = True

        if cfg.debug:
            print(
                f"kldm_dpnp_sg_pcs graph={graph_idx + 1} status=selected "
                f"templates={len(ranked_templates)}/{len(templates)} "
                f"raw_candidates={len(raw_candidates)} candidates={len(candidates)} "
                f"rejected={rejected} sg={requested_sg} detected_sg={branch.detected_sg} "
                f"energy={branch.energy:.6f} final_energy={_selection_energy(branch):.6f} "
                f"ranker={branch.ranker_score:.6f} "
                f"min_pair_distance={branch.min_pair_distance:.4f} "
                f"signature={_branch_signature_labels(branch)}",
                flush=True,
            )

        return branch

    if cfg.debug:
        print(
            f"kldm_dpnp_sg_pcs graph={graph_idx + 1} status=no_valid_branch "
            f"templates={len(ranked_templates)}/{len(templates)} "
            f"raw_candidates={len(raw_candidates)} candidates={len(candidates)} "
            f"rejected={rejected}",
            flush=True,
        )

    return None


def pcs_sample_batch(
    *,
    batch: Any,
    pos_t: torch.Tensor,
    l_t: torch.Tensor,
    h_t: torch.Tensor,
    lattice_transform: Any,
    eta: float,
    cfg: SGDPnPConfig,
    template_prior: TemplatePrior | None = None,
    template_ranker: torch.nn.Module | None = None,
    template_cache: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ptr = batch.ptr.tolist()
    requested = torch.as_tensor(batch.space_group, device=h_t.device, dtype=torch.long).reshape(-1)
    pos_blocks: list[torch.Tensor] = []
    l_blocks: list[torch.Tensor] = []
    h_blocks: list[torch.Tensor] = []
    success: list[bool] = []
    for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
        branch = _pcs_graph(
            graph_idx=graph_idx,
            anchor_frac=pos_t[start:end],
            anchor_l=l_t[graph_idx],
            anchor_species=h_t[start:end],
            requested_sg=int(requested[graph_idx].item()),
            lattice_transform=lattice_transform,
            eta=float(eta),
            cfg=cfg,
            template_prior=template_prior,
            template_ranker=template_ranker,
            template_cache=template_cache,
        )
        if branch is None:
            pos_blocks.append(pos_t[start:end])
            l_blocks.append(l_t[graph_idx].view(1, -1))
            h_blocks.append(h_t[start:end])
            success.append(False)
            continue
        pos_blocks.append(branch.frac_coords.to(device=pos_t.device, dtype=pos_t.dtype))
        l_blocks.append(
            _encode_lattice_features(
                cell_matrix=branch.cell_matrix.to(device=l_t.device, dtype=l_t.dtype),
                num_atoms=int(branch.frac_coords.shape[0]),
                lattice_transform=lattice_transform,
            )
        )
        h_blocks.append(branch.atomic_numbers.to(device=h_t.device, dtype=h_t.dtype))
        success.append(True)
    return (
        torch.cat(pos_blocks, dim=0),
        torch.cat(l_blocks, dim=0),
        torch.cat(h_blocks, dim=0),
        torch.tensor(success, device=h_t.device, dtype=torch.bool),
    )


def _replace_failed_graphs(
    *,
    batch: Any,
    candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    fallback: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    success: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cand_pos, cand_l, cand_h = candidate
    fb_pos, fb_l, fb_h = fallback
    ptr = batch.ptr.tolist()
    pos_blocks: list[torch.Tensor] = []
    l_blocks: list[torch.Tensor] = []
    h_blocks: list[torch.Tensor] = []
    for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
        if bool(success[graph_idx].item()):
            pos_blocks.append(cand_pos[start:end])
            l_blocks.append(cand_l[graph_idx].view(1, -1))
            h_blocks.append(cand_h[start:end])
        else:
            pos_blocks.append(fb_pos[start:end])
            l_blocks.append(fb_l[graph_idx].view(1, -1))
            h_blocks.append(fb_h[start:end])
    return torch.cat(pos_blocks, dim=0), torch.cat(l_blocks, dim=0), torch.cat(h_blocks, dim=0)


def eta_schedule(cfg: SGDPnPConfig, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if int(cfg.outer_steps) <= 1:
        return torch.tensor([float(cfg.eta_end)], device=device, dtype=dtype)
    return torch.linspace(float(cfg.eta_start), float(cfg.eta_end), int(cfg.outer_steps), device=device, dtype=dtype)


def sample_kldm_dpnp_sg(
    *,
    model: Any,
    n_steps: int,
    batch: Any,
    lattice_transform: Any,
    t_start: float,
    t_final: float,
    config: SGDPnPConfig,
    template_prior: TemplatePrior | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hasattr(batch, "space_group"):
        raise ValueError("KLDM-DPnP-SG requires batch.space_group.")

    started = time.perf_counter()
    pos_t, v_t, l_t, h_t = model.sample_CSP_algorithm3(
        n_steps=n_steps,
        batch=batch,
        t_start=t_start,
        t_final=t_final,
    )
    print(
        f"kldm_dpnp_sg_progress phase=prior graphs={int(batch.num_graphs)} "
        f"elapsed_s={time.perf_counter() - started:.1f}",
        flush=True,
    )
    schedule = eta_schedule(config, device=pos_t.device, dtype=pos_t.dtype)
    template_ranker = _maybe_load_template_ranker(
        path=config.template_ranker_path if float(config.template_ranker_weight) > 0.0 else None,
        device=pos_t.device,
    )
    template_cache = _maybe_load_disk_template_cache(config.template_cache_path)
    pos_current, l_current, h_current = pos_t, l_t, h_t
    last_valid_pcs = (pos_current.clone(), l_current.clone(), h_current.clone())
    last_valid_success = torch.zeros(int(batch.num_graphs), device=pos_t.device, dtype=torch.bool)

    if config.initial_pcs:
        pos_candidate, l_candidate, h_candidate, success = pcs_sample_batch(
            batch=batch,
            pos_t=pos_current,
            l_t=l_current,
            h_t=h_current,
            lattice_transform=lattice_transform,
            eta=float(schedule[0].item()),
            cfg=config,
            template_prior=template_prior,
            template_ranker=template_ranker,
            template_cache=template_cache,
        )
        last_valid_pcs = _replace_failed_graphs(
            batch=batch,
            candidate=(pos_candidate, l_candidate, h_candidate),
            fallback=last_valid_pcs,
            success=success,
        )
        last_valid_success = torch.logical_or(last_valid_success, success)
        pos_current, l_current, h_current = last_valid_pcs
        if config.debug:
            print(
                f"kldm_dpnp_sg_progress phase=initial_pcs success={int(success.sum().item())}/{int(batch.num_graphs)}",
                flush=True,
            )
        v_t = model.tdm.sample_velocity_noise(pos_current, index=batch.batch)

    for step_idx, eta in enumerate(schedule, start=1):
        pcs_started = time.perf_counter()
        pos_candidate, l_candidate, h_candidate, success = pcs_sample_batch(
            batch=batch,
            pos_t=pos_current,
            l_t=l_current,
            h_t=h_current,
            lattice_transform=lattice_transform,
            eta=float(eta.item()),
            cfg=config,
            template_prior=template_prior,
            template_ranker=template_ranker,
            template_cache=template_cache,
        )
        pcs_elapsed = time.perf_counter() - pcs_started
        last_valid_pcs = _replace_failed_graphs(
            batch=batch,
            candidate=(pos_candidate, l_candidate, h_candidate),
            fallback=last_valid_pcs,
            success=success,
        )
        last_valid_success = torch.logical_or(last_valid_success, success)
        pos_half, l_half, h_half = last_valid_pcs
        if int(config.dds_steps) <= 0:
            pos_current, l_current, h_current = pos_half, l_half, h_half
            v_t = model.tdm.sample_velocity_noise(pos_current, index=batch.batch)
            dds_elapsed = 0.0
        else:
            dds_started = time.perf_counter()
            dds_t_start = model._algorithm6_map_eta_to_kldm_time(
                float(eta.item()),
                num_atoms=batch.num_atoms,
                ref_l=l_half,
            )
            dds_t_final = min(float(config.dds_t_final), float(dds_t_start) * 0.5)
            if dds_t_start <= dds_t_final:
                pos_current, l_current, h_current = pos_half, l_half, h_half
                v_t = model.tdm.sample_velocity_noise(pos_current, index=batch.batch)
            else:
                pos_current, v_t, l_current, h_current = model._algorithm6_dds_repair(
                    batch=batch,
                    pos_clean=pos_half,
                    l_clean=l_half,
                    h_clean=h_half,
                    n_steps=int(config.dds_steps),
                    t_start=float(dds_t_start),
                    t_final=float(dds_t_final),
                )
            dds_elapsed = time.perf_counter() - dds_started
        print(
            f"kldm_dpnp_sg_progress phase=outer_step step={step_idx}/{len(schedule)} "
            f"eta={float(eta.item()):.5f} pcs_elapsed_s={pcs_elapsed:.1f} "
            f"dds_elapsed_s={dds_elapsed:.1f} pcs_success={int(success.sum().item())}/{int(batch.num_graphs)}",
            flush=True,
        )

    if int(config.dds_steps) <= 0:
        pos_out, l_out, h_out = last_valid_pcs
        final_success_count = int(last_valid_success.sum().item())
    else:
        final_eta = float(schedule[-1].item()) * float(config.final_eta_scale)
        pos_candidate, l_candidate, h_candidate, success = pcs_sample_batch(
            batch=batch,
            pos_t=pos_current,
            l_t=l_current,
            h_t=h_current,
            lattice_transform=lattice_transform,
            eta=final_eta,
            cfg=config,
            template_prior=template_prior,
            template_ranker=template_ranker,
            template_cache=template_cache,
        )
        pos_out, l_out, h_out = _replace_failed_graphs(
            batch=batch,
            candidate=(pos_candidate, l_candidate, h_candidate),
            fallback=last_valid_pcs,
            success=success,
        )
        final_success_count = int(success.sum().item())
    v_out = model.tdm.sample_velocity_noise(pos_out, index=batch.batch)
    print(
        f"kldm_dpnp_sg_progress phase=done final_pcs_success={final_success_count}/{int(batch.num_graphs)} "
        f"total_elapsed_s={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return pos_out, v_out, l_out, h_out
