from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from kldmPlus.data.transform import (
    KLDMContinuousIntervalLattice,
    lattice_feature_components,
)
from kldmPlus.symmetry.frame_bridge import (
    SymmetryFrameBridge,
    build_symmetry_frame_bridge,
    map_standardized_structure_to_vanilla_frame,
    standardize_structure,
)
from kldmPlus.symmetry.template_prior import TemplatePrior, template_prior_score
from kldmPlus.symmetry.k_basis import (
    KFamilyConstraint,
    cell_to_k,
    free_vars_to_k,
    k_to_cell_matrix,
    k_to_free_vars,
    space_group_k_constraint,
)
from kldmPlus.symmetry.wyckoff_templates import (
    WyckoffTemplate,
    composition_to_species_counts,
    expand_wyckoff_template_torch,
    extract_wyckoff_templates,
    flatten_site_signature,
    requested_composition_key,
    requested_conventional_atomic_numbers,
    sample_random_free_vars,
)

try:
    from pymatgen.core import Element, Lattice, Structure
except ImportError:  # pragma: no cover
    Element = Lattice = Structure = None

try:
    from pyxtal.symmetry import Group
except ImportError:  # pragma: no cover
    Group = None

try:
    import spglib
except ImportError:  # pragma: no cover
    spglib = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


@dataclass
class PCSProjectionResult:
    projected_structure_standardized: Any
    projected_structure_primitive: Any
    projected_structure_vanilla: Any
    template: WyckoffTemplate
    free_vars: torch.Tensor
    lattice_free_vars: torch.Tensor
    objective: float
    template_rank: int
    candidate_count: int
    standardized_space_group: int | None = None
    primitive_space_group: int | None = None


@dataclass
class PCSDiagnostics:
    target_k: torch.Tensor
    final_k: torch.Tensor
    target_cell_from_k: torch.Tensor
    final_cell_from_k: torch.Tensor
    coord_loss: float
    lattice_loss: float
    pairdist_loss: float
    steric_loss: float
    volume_loss: float
    k6_loss: float
    prox_energy: float
    likelihood_energy: float
    total_energy: float


@dataclass
class PCSEnergyResult:
    energy: torch.Tensor
    coord_loss: torch.Tensor
    lattice_loss: torch.Tensor
    pairdist_loss: torch.Tensor
    steric_loss: torch.Tensor
    volume_loss: torch.Tensor
    k6_loss: torch.Tensor
    prox_energy: torch.Tensor
    likelihood_energy: torch.Tensor
    frac_coords: torch.Tensor
    k_projected: torch.Tensor
    projected_cell: torch.Tensor


@dataclass
class PCSTemplateState:
    template: WyckoffTemplate
    constraint: KFamilyConstraint
    bridge: SymmetryFrameBridge
    free_vars: torch.Tensor
    lattice_free_vars: torch.Tensor
    objective: float
    template_rank: int
    candidate_count: int
    ranking_objective: float = float("inf")
    target_centering_symbol: str | None = None
    target_centering_translations: torch.Tensor | None = None
    target_frac: torch.Tensor | None = None
    target_atomic_numbers: torch.Tensor | None = None
    target_cell: torch.Tensor | None = None
    target_k: torch.Tensor | None = None
    fixed_target_assignment: torch.Tensor | None = None
    target_pairdist_hist: torch.Tensor | None = None
    pairdist_bin_centers: torch.Tensor | None = None
    anchor_frac: torch.Tensor | None = None
    anchor_atomic_numbers: torch.Tensor | None = None
    anchor_cell: torch.Tensor | None = None
    anchor_k: torch.Tensor | None = None
    anchor_assignment: torch.Tensor | None = None
    anchor_free_vars: torch.Tensor | None = None
    anchor_lattice_free_vars: torch.Tensor | None = None
    anchor_pairdist_hist: torch.Tensor | None = None
    anchor_pairdist_bin_centers: torch.Tensor | None = None
    anchor_representation_name: str | None = None
    reference_volume: float | None = None
    reference_k6: float | None = None
    projected_reference_volume: float | None = None
    projected_reference_k6: float | None = None
    prior_score: int = 0
    prior_bonus: float = 0.0
    freeze_lattice: bool = False
    target_species_orbit_signature: tuple[tuple[int, str], ...] | None = None
    template_species_orbit_signature: tuple[tuple[int, str], ...] | None = None
    species_orbit_mismatch: int = 0
    orbit_reference_is_oracle: bool = False
    orbit_reference_rank_first: bool = False
    target_representation_name: str | None = None
    branch_frac_coords: torch.Tensor | None = None
    branch_atomic_numbers: torch.Tensor | None = None
    branch_lattice_features: torch.Tensor | None = None
    vanilla_coord_distance: float = float("inf")
    vanilla_lattice_k_distance: float = float("inf")
    asymmetric_unit_distortion: float = float("inf")
    mala_acceptance_rate: float = 0.0
    mala_accept_count: int = 0
    mala_attempted_steps: int = 0
    mala_coord_loss: float = float("inf")
    mala_lattice_loss: float = float("inf")
    mala_pairdist_loss: float = float("inf")
    mala_steric_loss: float = float("inf")
    mala_volume_loss: float = float("inf")
    mala_k6_loss: float = float("inf")
    mala_prox_energy: float = float("inf")
    mala_likelihood_energy: float = float("inf")
    mala_total_energy: float = float("inf")
    soft_physics_failed: bool = False


@dataclass
class PCSValidationResult:
    composition_match: bool
    requested_space_group: int
    detected_space_group: int | None
    requested_space_group_match: bool


def _require_pymatgen() -> None:
    if None in (Element, Lattice, Structure):
        raise ImportError("PCS projection requires pymatgen.")


def _detect_space_group_number(
    *,
    structure,
    symprec: float,
    angle_tolerance: float,
) -> int | None:
    _require_pymatgen()
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PCS projection requires pymatgen symmetry tools.") from exc

    try:
        return int(
            SpacegroupAnalyzer(
                structure,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            ).get_space_group_number()
        )
    except Exception:
        return None


def _torch_atomic_multiset_matches(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape:
        return False
    if left.numel() == 0:
        return True
    left_sorted = torch.sort(left.detach().to(device="cpu", dtype=torch.long)).values
    right_sorted = torch.sort(right.detach().to(device="cpu", dtype=torch.long)).values
    return bool(torch.equal(left_sorted, right_sorted))


def _torus_pairwise_distance_sq(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    delta = source.unsqueeze(1) - target.unsqueeze(0)
    delta = delta - torch.round(delta)
    return delta.square().sum(dim=-1)


def _match_cost_matrix(cost_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    detached = cost_matrix.detach()
    if not torch.isfinite(detached).all():
        raise RuntimeError("matrix contains invalid numeric entries")
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


def _species_matched_torus_loss(
    *,
    source_frac: torch.Tensor,
    source_atomic_numbers: torch.Tensor,
    target_frac: torch.Tensor,
    target_atomic_numbers: torch.Tensor,
) -> torch.Tensor:
    unique_species = sorted(set(int(v) for v in source_atomic_numbers.detach().cpu().tolist()))
    if unique_species != sorted(set(int(v) for v in target_atomic_numbers.detach().cpu().tolist())):
        return source_frac.new_tensor(float("inf"))

    loss = source_frac.new_zeros(())
    for atomic_number in unique_species:
        source_mask = source_atomic_numbers == atomic_number
        target_mask = target_atomic_numbers == atomic_number
        if int(source_mask.sum().item()) != int(target_mask.sum().item()):
            return source_frac.new_tensor(float("inf"))

        src = source_frac[source_mask]
        tgt = target_frac[target_mask]
        if src.numel() == 0:
            continue
        cost_matrix = _torus_pairwise_distance_sq(src, tgt)
        row_idx, col_idx = _match_cost_matrix(cost_matrix)
        delta = src[row_idx] - tgt[col_idx]
        delta = delta - torch.round(delta)
        loss = loss + delta.square().mean()
    return loss


def _species_assignment_indices(
    *,
    source_frac: torch.Tensor,
    source_atomic_numbers: torch.Tensor,
    target_frac: torch.Tensor,
    target_atomic_numbers: torch.Tensor,
) -> torch.Tensor:
    unique_species = sorted(set(int(v) for v in source_atomic_numbers.detach().cpu().tolist()))
    if unique_species != sorted(set(int(v) for v in target_atomic_numbers.detach().cpu().tolist())):
        raise RuntimeError("Could not build a fixed species assignment because the species sets differ.")

    assignment = torch.empty((source_frac.shape[0],), device=source_frac.device, dtype=torch.long)
    for atomic_number in unique_species:
        source_mask = source_atomic_numbers == atomic_number
        target_mask = target_atomic_numbers == atomic_number
        if int(source_mask.sum().item()) != int(target_mask.sum().item()):
            raise RuntimeError("Could not build a fixed species assignment because species counts differ.")
        src = source_frac[source_mask]
        tgt = target_frac[target_mask]
        cost_matrix = _torus_pairwise_distance_sq(src, tgt)
        row_idx, col_idx = _match_cost_matrix(cost_matrix)

        source_indices = torch.nonzero(source_mask, as_tuple=False).squeeze(-1)
        target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)
        assignment[source_indices[row_idx]] = target_indices[col_idx]
    return assignment


def _fixed_assignment_torus_loss(
    *,
    source_frac: torch.Tensor,
    target_frac: torch.Tensor,
    target_assignment: torch.Tensor,
) -> torch.Tensor:
    matched_target = target_frac[target_assignment]
    delta = source_frac - matched_target
    delta = delta - torch.round(delta)
    return delta.square().mean()


def _wrapped_free_var_loss(
    *,
    source_free_vars: torch.Tensor,
    target_free_vars: torch.Tensor,
) -> torch.Tensor:
    if source_free_vars.shape != target_free_vars.shape:
        raise RuntimeError("Reduced-chart free-variable proximal loss requires matching shapes.")
    if source_free_vars.numel() == 0:
        return source_free_vars.new_zeros(())
    delta = source_free_vars - target_free_vars
    delta = delta - torch.round(delta)
    return delta.square().mean()


def _structure_species_orbit_signature_with_source(
    *,
    structure,
    symprec: float,
    angle_tolerance: float,
) -> tuple[tuple[tuple[int, str], ...], str]:
    _require_pymatgen()
    if spglib is not None:
        try:
            dataset = spglib.get_symmetry_dataset(
                (
                    np.asarray(structure.lattice.matrix, dtype=float),
                    np.asarray(structure.frac_coords, dtype=float),
                    np.asarray(structure.atomic_numbers, dtype=int),
                ),
                symprec=float(symprec),
                angle_tolerance=float(angle_tolerance),
            )
            if dataset is not None:
                wyckoffs = getattr(dataset, "wyckoffs", None)
                equivalent_atoms = getattr(dataset, "equivalent_atoms", None)
                if wyckoffs is None and isinstance(dataset, dict):
                    wyckoffs = dataset.get("wyckoffs")
                if equivalent_atoms is None and isinstance(dataset, dict):
                    equivalent_atoms = dataset.get("equivalent_atoms")

                if wyckoffs is not None and equivalent_atoms is not None:
                    atomic_numbers = np.asarray(structure.atomic_numbers, dtype=int)
                    rep_to_count: dict[int, int] = {}
                    rep_to_atomic_number: dict[int, int] = {}
                    rep_to_letter: dict[int, str] = {}
                    for atom_idx, rep_idx in enumerate(np.asarray(equivalent_atoms, dtype=int).tolist()):
                        rep_to_count[rep_idx] = rep_to_count.get(rep_idx, 0) + 1
                        rep_to_atomic_number.setdefault(rep_idx, int(atomic_numbers[atom_idx]))
                        wyckoff_value = wyckoffs[atom_idx]
                        if isinstance(wyckoff_value, (int, np.integer)):
                            letter = chr(ord("a") + int(wyckoff_value))
                        else:
                            letter = str(wyckoff_value)
                        rep_to_letter.setdefault(rep_idx, letter)

                    pairs = tuple(
                        sorted(
                            [
                                (rep_to_atomic_number[rep_idx], f"{rep_to_count[rep_idx]}{rep_to_letter[rep_idx]}")
                                for rep_idx in rep_to_count
                            ],
                            key=lambda item: (item[0], item[1]),
                        )
                    )
                    if pairs:
                        return pairs, "spglib"
        except Exception:
            pass
    try:
        result = build_pyxtal_wyckoff_result(
            structure,
            symprec=symprec,
            pyxtal_tol=max(float(symprec), 1e-3),
        )
        pairs = tuple(
            sorted(
                [
                    (int(z), str(label))
                    for z, label in zip(result.anchor_atomic_numbers.tolist(), result.site_labels)
                ],
                key=lambda item: (item[0], item[1]),
            )
        )
        if pairs:
            return pairs, "pyxtal"
    except Exception:
        pass
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PCS projection requires pymatgen symmetry tools.") from exc

    try:
        symmetrized = SpacegroupAnalyzer(
            structure,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        ).get_symmetrized_structure()
        pairs: list[tuple[int, str]] = []
        for sites, label in zip(symmetrized.equivalent_sites, symmetrized.wyckoff_symbols):
            if not sites:
                continue
            pairs.append((int(sites[0].specie.Z), str(label)))
        return tuple(sorted(pairs, key=lambda item: (item[0], item[1]))), "pymatgen"
    except Exception:
        return (), "none"


def _template_species_orbit_signature(template: WyckoffTemplate) -> tuple[tuple[int, str], ...]:
    pairs = [(int(site.atomic_number), str(site.label)) for site in template.site_templates]
    return tuple(sorted(pairs, key=lambda item: (item[0], item[1])))


def _species_orbit_mismatch_count(
    *,
    template_signature: tuple[tuple[int, str], ...],
    target_signature: tuple[tuple[int, str], ...],
) -> int:
    if not target_signature:
        return 0
    template_counter = Counter(template_signature)
    target_counter = Counter(target_signature)
    mismatch = 0
    for key in set(template_counter) | set(target_counter):
        mismatch += abs(int(template_counter.get(key, 0)) - int(target_counter.get(key, 0)))
    return int(mismatch)


def _periodic_pairwise_distances(
    *,
    frac_coords: torch.Tensor,
    cell_matrix: torch.Tensor,
) -> torch.Tensor:
    num_atoms = int(frac_coords.shape[0])
    if num_atoms < 2:
        return frac_coords.new_zeros((0,))
    delta = frac_coords.unsqueeze(1) - frac_coords.unsqueeze(0)
    delta = delta - torch.round(delta)
    cart_delta = torch.einsum("...d,de->...e", delta, cell_matrix)
    distances = torch.linalg.norm(cart_delta, dim=-1)
    mask = torch.triu(torch.ones((num_atoms, num_atoms), device=frac_coords.device, dtype=torch.bool), diagonal=1)
    return distances[mask]


def _soft_pair_distance_histogram(
    *,
    distances: torch.Tensor,
    bin_centers: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    if distances.numel() == 0:
        return bin_centers.new_zeros(bin_centers.shape)
    bw = max(float(bandwidth), 1e-6)
    weights = torch.exp(-0.5 * ((distances.unsqueeze(-1) - bin_centers.unsqueeze(0)) / bw).square())
    hist = weights.mean(dim=0)
    hist = hist / hist.sum().clamp_min(1e-8)
    return hist


def _pair_distance_histogram_loss_from_distances(
    *,
    source_distances: torch.Tensor,
    target_hist: torch.Tensor,
    bin_centers: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    source_hist = _soft_pair_distance_histogram(
        distances=source_distances,
        bin_centers=bin_centers,
        bandwidth=bandwidth,
    )
    return (source_hist - target_hist).square().mean()


def _pair_distance_histogram_loss(
    *,
    source_frac: torch.Tensor,
    source_cell: torch.Tensor,
    target_frac: torch.Tensor,
    target_cell: torch.Tensor,
    bins: int,
    max_distance: float,
    bandwidth: float,
    target_hist: torch.Tensor | None = None,
    bin_centers: torch.Tensor | None = None,
) -> torch.Tensor:
    if bin_centers is None:
        num_bins = max(int(bins), 2)
        bin_centers = torch.linspace(
            0.0,
            float(max_distance),
            num_bins,
            device=source_frac.device,
            dtype=source_frac.dtype,
        )
    else:
        bin_centers = bin_centers.to(device=source_frac.device, dtype=source_frac.dtype)
    source_distances = _periodic_pairwise_distances(frac_coords=source_frac, cell_matrix=source_cell)
    if target_hist is None:
        target_distances = _periodic_pairwise_distances(frac_coords=target_frac, cell_matrix=target_cell)
        target_hist = _soft_pair_distance_histogram(
            distances=target_distances,
            bin_centers=bin_centers,
            bandwidth=bandwidth,
        )
    else:
        target_hist = target_hist.to(device=source_frac.device, dtype=source_frac.dtype)
    return _pair_distance_histogram_loss_from_distances(
        source_distances=source_distances,
        target_hist=target_hist,
        bin_centers=bin_centers,
        bandwidth=bandwidth,
    )


def _build_target_pairdist_cache(
    *,
    target_frac: torch.Tensor,
    target_cell: torch.Tensor,
    bins: int,
    max_distance: float,
    bandwidth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_bins = max(int(bins), 2)
    bin_centers = torch.linspace(
        0.0,
        float(max_distance),
        num_bins,
        device=target_frac.device,
        dtype=target_frac.dtype,
    )
    target_distances = _periodic_pairwise_distances(frac_coords=target_frac, cell_matrix=target_cell)
    target_hist = _soft_pair_distance_histogram(
        distances=target_distances,
        bin_centers=bin_centers,
        bandwidth=bandwidth,
    )
    return target_hist.detach().clone(), bin_centers.detach().clone()


def _steric_overlap_loss(
    *,
    distances: torch.Tensor,
    min_distance: float,
) -> torch.Tensor:
    if distances.numel() == 0:
        return distances.new_zeros(())
    floor = distances.new_tensor(float(max(min_distance, 0.0)))
    penalties = torch.relu(floor - distances)
    return penalties.square().mean()


def _cell_volume(cell_matrix: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.linalg.det(cell_matrix)).clamp_min(1e-8)


def _volume_ratio_loss(
    *,
    projected_cell: torch.Tensor,
    reference_volume: float | None,
    min_ratio: float,
    max_ratio: float,
) -> torch.Tensor:
    if reference_volume is None or reference_volume <= 0.0:
        return projected_cell.new_zeros(())

    log_ratio = torch.log(
        _cell_volume(projected_cell)
        / projected_cell.new_tensor(float(max(reference_volume, 1e-8)))
    )
    if float(min_ratio) <= 0.0 and float(max_ratio) <= 0.0:
        return log_ratio.square()

    penalty = projected_cell.new_zeros(())
    if float(min_ratio) > 0.0:
        lower = projected_cell.new_tensor(float(np.log(max(float(min_ratio), 1e-8))))
        penalty = penalty + torch.relu(lower - log_ratio).square()
    if float(max_ratio) > 0.0:
        upper = projected_cell.new_tensor(float(np.log(max(float(max_ratio), 1e-8))))
        penalty = penalty + torch.relu(log_ratio - upper).square()
    return penalty


def _k6_reference_loss(
    *,
    k_projected: torch.Tensor,
    reference_k6: float | None,
) -> torch.Tensor:
    if reference_k6 is None:
        return k_projected.new_zeros(())
    ref = k_projected.new_tensor(float(reference_k6))
    return (k_projected[..., -1] - ref).square()


def _encode_cell_to_lattice_features(
    *,
    cell_matrix: torch.Tensor,
    num_atoms: int,
    lattice_transform,
) -> torch.Tensor:
    if not isinstance(lattice_transform, KLDMContinuousIntervalLattice):
        transform = KLDMContinuousIntervalLattice(standardize=False)
    else:
        transform = lattice_transform

    log_lengths, angle_features = lattice_feature_components(cell_matrix, eps=transform.eps)
    if transform.standardize and transform.lengths_loc_scale is not None:
        log_lengths, angle_features = transform._encode_x0_parts(  # noqa: SLF001
            log_lengths=log_lengths,
            angle_features=angle_features,
            num_atoms=int(num_atoms),
        )
        features = torch.cat([log_lengths, angle_features], dim=0)
    else:
        features = torch.cat([log_lengths, angle_features], dim=0)
        features = transform.standardize_value(features)
    return features.view(1, 6)


def _build_structure_from_standardized_projection(
    *,
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell_matrix: torch.Tensor,
):
    _require_pymatgen()
    species = [Element.from_Z(int(z)).symbol for z in atomic_numbers.detach().cpu().tolist()]
    return Structure(
        lattice=Lattice(cell_matrix.detach().cpu().numpy()),
        species=species,
        coords=torch.remainder(frac_coords, 1.0).detach().cpu().numpy().tolist(),
        coords_are_cartesian=False,
    ).get_sorted_structure()


def _build_vanilla_structure(
    *,
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell_matrix: torch.Tensor,
):
    _require_pymatgen()
    species = [Element.from_Z(int(z)).symbol for z in atomic_numbers.detach().cpu().tolist()]
    return Structure(
        lattice=Lattice(cell_matrix.detach().cpu().numpy()),
        species=species,
        coords=torch.remainder(frac_coords, 1.0).detach().cpu().numpy().tolist(),
        coords_are_cartesian=False,
    ).get_sorted_structure()


def _deduplicate_structure_sites(
    *,
    structure,
    expected_atomic_numbers: np.ndarray | None = None,
    tol: float = 1e-3,
):
    _require_pymatgen()
    frac = np.asarray(structure.frac_coords, dtype=float)
    atomic_numbers = np.asarray(structure.atomic_numbers, dtype=int)
    keep_indices: list[int] = []
    seen_by_species: dict[int, list[np.ndarray]] = {}

    for idx, (coord, atomic_number) in enumerate(zip(frac, atomic_numbers)):
        bucket = seen_by_species.setdefault(int(atomic_number), [])
        is_duplicate = False
        for prev in bucket:
            delta = coord - prev
            delta = delta - np.round(delta)
            if float(np.max(np.abs(delta))) <= tol:
                is_duplicate = True
                break
        if not is_duplicate:
            keep_indices.append(idx)
            bucket.append(coord)

    if expected_atomic_numbers is not None:
        expected_atomic_numbers = np.asarray(expected_atomic_numbers, dtype=int)
        keep_atomic_numbers = atomic_numbers[keep_indices]
        if len(keep_indices) != int(expected_atomic_numbers.shape[0]):
            return structure
        if not np.array_equal(np.sort(keep_atomic_numbers), np.sort(expected_atomic_numbers)):
            return structure

    species = [Element.from_Z(int(z)).symbol for z in atomic_numbers[keep_indices].tolist()]
    return Structure(
        lattice=structure.lattice,
        species=species,
        coords=np.mod(frac[keep_indices], 1.0).tolist(),
        coords_are_cartesian=False,
    ).get_sorted_structure()


def _primitive_basis_matrix_for_centering(symbol: str) -> np.ndarray | None:
    centering = (symbol or "P").upper()
    if centering == "F":
        return np.asarray(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            dtype=float,
        )
    if centering == "I":
        return np.asarray(
            [
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
            ],
            dtype=float,
        )
    if centering == "C":
        return np.asarray(
            [
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    if centering == "A":
        return np.asarray(
            [
                [0.0, 0.5, -0.5],
                [0.0, 0.5, 0.5],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
    if centering == "B":
        return np.asarray(
            [
                [0.5, 0.0, -0.5],
                [0.5, 0.0, 0.5],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
    return None


def _structure_to_primitive_centering_basis(
    *,
    structure,
    centering_symbol: str | None,
    expected_atomic_numbers: np.ndarray,
):
    _require_pymatgen()
    basis = _primitive_basis_matrix_for_centering(centering_symbol or "P")
    if basis is None:
        return structure

    old_lattice = np.asarray(structure.lattice.matrix, dtype=float)
    frac = np.asarray(structure.frac_coords, dtype=float)
    atomic_numbers = np.asarray(structure.atomic_numbers, dtype=int)
    new_lattice = basis @ old_lattice
    new_frac = np.mod(frac @ np.linalg.inv(basis), 1.0)
    species = [Element.from_Z(int(z)).symbol for z in atomic_numbers.tolist()]
    transformed = Structure(
        lattice=Lattice(new_lattice),
        species=species,
        coords=new_frac.tolist(),
        coords_are_cartesian=False,
    ).get_sorted_structure()
    return _deduplicate_structure_sites(
        structure=transformed,
        expected_atomic_numbers=expected_atomic_numbers,
    )


def _collapse_centering_equivalent_structure(
    *,
    structure,
    translations: torch.Tensor | None,
    expected_atomic_numbers: np.ndarray,
    tol: float = 1e-3,
):
    _require_pymatgen()
    if translations is None or int(translations.shape[0]) <= 1:
        return structure

    frac = np.asarray(structure.frac_coords, dtype=float)
    atomic_numbers = np.asarray(structure.atomic_numbers, dtype=int)
    expected_atomic_numbers = np.asarray(expected_atomic_numbers, dtype=int)
    if frac.shape[0] <= expected_atomic_numbers.shape[0]:
        return structure

    translations_np = np.asarray(translations.detach().cpu(), dtype=float)
    unused: set[int] = set(range(frac.shape[0]))
    keep_indices: list[int] = []

    def _match_index(source_frac: np.ndarray, atomic_number: int, translation: np.ndarray) -> int | None:
        target_frac = np.mod(source_frac + translation, 1.0)
        best_idx: int | None = None
        best_err = float("inf")
        for candidate_idx in list(unused):
            if int(atomic_numbers[candidate_idx]) != int(atomic_number):
                continue
            delta = frac[candidate_idx] - target_frac
            delta = delta - np.round(delta)
            err = float(np.max(np.abs(delta)))
            if err < min(best_err, tol):
                best_err = err
                best_idx = candidate_idx
        return best_idx

    while unused:
        root_idx = min(unused)
        root_frac = frac[root_idx]
        root_atomic_number = int(atomic_numbers[root_idx])
        orbit_indices: list[int] = []
        for translation in translations_np:
            matched_idx = _match_index(root_frac, root_atomic_number, translation)
            if matched_idx is not None and matched_idx not in orbit_indices:
                orbit_indices.append(matched_idx)

        if len(orbit_indices) == int(translations_np.shape[0]):
            keep_indices.append(root_idx)
            for orbit_idx in orbit_indices:
                unused.discard(orbit_idx)
        else:
            keep_indices.append(root_idx)
            unused.discard(root_idx)

    if len(keep_indices) != int(expected_atomic_numbers.shape[0]):
        return structure

    keep_atomic_numbers = atomic_numbers[keep_indices]
    if not np.array_equal(np.sort(keep_atomic_numbers), np.sort(expected_atomic_numbers)):
        return structure

    species = [Element.from_Z(int(z)).symbol for z in keep_atomic_numbers.tolist()]
    return Structure(
        lattice=structure.lattice,
        species=species,
        coords=np.mod(frac[keep_indices], 1.0).tolist(),
        coords_are_cartesian=False,
    ).get_sorted_structure()


def _standardized_target_tensors(
    bridge: SymmetryFrameBridge,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _standardized_target_tensors_from_structure(
        structure=bridge.standardized_structure,
        device=device,
        dtype=dtype,
    )


def _standardized_target_tensors_from_structure(
    structure,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    standardized_frac = torch.tensor(
        np.asarray(structure.frac_coords, dtype=float).copy(),
        device=device,
        dtype=dtype,
    )
    standardized_atomic_numbers = torch.as_tensor(
        np.asarray(structure.atomic_numbers, dtype=int),
        device=device,
        dtype=torch.long,
    )
    standardized_cell = torch.tensor(
        np.asarray(structure.lattice.matrix, dtype=float).copy(),
        device=device,
        dtype=dtype,
    )
    current_k = cell_to_k(standardized_cell, eps=1e-8)
    return standardized_frac, standardized_atomic_numbers, standardized_cell, current_k


def _requested_centering_symbol(space_group_number: int) -> str:
    if Group is None:
        return "P"
    try:
        group = Group(int(space_group_number))
        symbol = str(group.symbol).strip()
        return symbol[0].upper() if symbol else "P"
    except Exception:
        return "P"


def _centering_translations(symbol: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    centering = symbol.upper()
    if centering == "P":
        translations = [[0.0, 0.0, 0.0]]
    elif centering == "I":
        translations = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    elif centering == "F":
        translations = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    elif centering == "A":
        translations = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5]]
    elif centering == "B":
        translations = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]
    elif centering == "C":
        translations = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
    elif centering == "R":
        translations = [[0.0, 0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]]
    else:
        translations = [[0.0, 0.0, 0.0]]
    return torch.tensor(translations, device=device, dtype=dtype)


def _expand_target_by_translations(
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    translations: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if translations is None or translations.shape[0] <= 1:
        return frac_coords, atomic_numbers

    expanded_coords = torch.cat(
        [
            torch.remainder(frac_coords + translation.view(1, 3), 1.0)
            for translation in translations
        ],
        dim=0,
    )
    expanded_atomic_numbers = atomic_numbers.repeat(int(translations.shape[0]))
    return expanded_coords, expanded_atomic_numbers


def _raw_target_in_requested_conventional_frame(
    *,
    frac_coords: torch.Tensor,
    cell_matrix: torch.Tensor,
    centering_symbol: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    basis = _primitive_basis_matrix_for_centering(centering_symbol)
    if basis is None or str(centering_symbol).upper() == "P":
        return frac_coords, cell_matrix

    basis_t = torch.as_tensor(basis, device=frac_coords.device, dtype=frac_coords.dtype)
    conventional_transform = torch.linalg.inv(basis_t)
    conventional_cell = conventional_transform @ cell_matrix
    conventional_frac = torch.remainder(frac_coords @ basis_t, 1.0)
    return conventional_frac, conventional_cell


def _candidate_composition_variants(
    *,
    raw_atomic_numbers: torch.Tensor,
    standardized_atomic_numbers: torch.Tensor,
    target_atomic_numbers_for_fit: torch.Tensor,
    requested_sg: int,
) -> list[tuple[torch.Tensor, tuple[int, tuple[int, ...], tuple[int, ...]]]]:
    variants: list[torch.Tensor] = [
        raw_atomic_numbers.detach().clone(),
        requested_conventional_atomic_numbers(
            raw_atomic_numbers,
            space_group_number=requested_sg,
        ).detach().clone(),
        standardized_atomic_numbers.detach().clone(),
        target_atomic_numbers_for_fit.detach().clone(),
    ]
    unique: list[tuple[torch.Tensor, tuple[int, tuple[int, ...], tuple[int, ...]]]] = []
    seen_keys: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()
    for variant in variants:
        species_order, species_counts = composition_to_species_counts(variant)
        key = (int(requested_sg), species_order, species_counts)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append((variant.to(dtype=torch.long), key))
    return unique


def _target_representations(
    *,
    raw_requested_frac: torch.Tensor,
    raw_requested_atomic_numbers: torch.Tensor,
    raw_requested_cell: torch.Tensor,
    raw_requested_frac_for_fit: torch.Tensor,
    raw_requested_atomic_numbers_for_fit: torch.Tensor,
    standardized_frac: torch.Tensor,
    standardized_atomic_numbers: torch.Tensor,
    standardized_cell: torch.Tensor,
    target_frac_for_fit: torch.Tensor,
    target_atomic_numbers_for_fit: torch.Tensor,
    include_requested_frame: bool = True,
) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
    variants: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    if include_requested_frame:
        variants.extend(
            [
                ("raw_requested_expanded", raw_requested_frac_for_fit, raw_requested_atomic_numbers_for_fit, raw_requested_cell),
                ("raw_requested", raw_requested_frac, raw_requested_atomic_numbers, raw_requested_cell),
            ]
        )
        variants.extend(
            [
                ("expanded", target_frac_for_fit, target_atomic_numbers_for_fit, standardized_cell),
                ("standardized", standardized_frac, standardized_atomic_numbers, standardized_cell),
            ]
        )
    else:
        use_expanded = int(target_atomic_numbers_for_fit.shape[0]) != int(standardized_atomic_numbers.shape[0])
        if use_expanded:
            # DiffCSP++/PyXtal templates for centered groups live in the requested
            # conventional cell. Expanding atoms inside the primitive standardized
            # cell creates an artificial density increase and later materializes
            # as a smaller primitive cell.
            variants.append(("raw_requested_expanded", raw_requested_frac_for_fit, raw_requested_atomic_numbers_for_fit, raw_requested_cell))
        else:
            variants.append(("standardized", standardized_frac, standardized_atomic_numbers, standardized_cell))
    unique: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    seen: set[tuple[tuple[int, ...], int, tuple[float, ...]]] = set()
    for name, frac, atomic_numbers, cell in variants:
        key = (
            tuple(int(v) for v in torch.sort(atomic_numbers.detach().to(dtype=torch.long)).values.cpu().tolist()),
            int(atomic_numbers.shape[0]),
            tuple(round(float(v), 8) for v in cell.detach().reshape(-1).cpu().tolist()),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, frac, atomic_numbers, cell))
    return unique


def _target_representation_from_name(
    *,
    target_name: str,
    raw_requested_frac: torch.Tensor,
    raw_requested_atomic_numbers: torch.Tensor,
    raw_requested_cell: torch.Tensor,
    standardized_frac: torch.Tensor,
    standardized_atomic_numbers: torch.Tensor,
    standardized_cell: torch.Tensor,
    centering_translations: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if target_name == "raw_requested_expanded":
        anchor_frac, anchor_atomic_numbers = _expand_target_by_translations(
            raw_requested_frac,
            raw_requested_atomic_numbers,
            centering_translations,
        )
        anchor_cell = raw_requested_cell
    elif target_name == "raw_requested":
        anchor_frac = raw_requested_frac
        anchor_atomic_numbers = raw_requested_atomic_numbers
        anchor_cell = raw_requested_cell
    elif target_name == "expanded":
        anchor_frac, anchor_atomic_numbers = _expand_target_by_translations(
            standardized_frac,
            standardized_atomic_numbers,
            centering_translations,
        )
        anchor_cell = standardized_cell
    elif target_name == "standardized":
        anchor_frac = standardized_frac
        anchor_atomic_numbers = standardized_atomic_numbers
        anchor_cell = standardized_cell
    else:
        raise ValueError(f"Unknown target representation {target_name!r}.")
    anchor_k = cell_to_k(anchor_cell, eps=1e-8)
    return anchor_frac, anchor_atomic_numbers, anchor_cell, anchor_k


def refresh_pcs_state_anchor(
    *,
    state: PCSTemplateState,
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell_matrix: torch.Tensor,
    pairdist_weight: float,
    pairdist_bins: int,
    pairdist_max_distance: float,
    pairdist_bandwidth: float,
    coord_weight: float = 1.0,
    lattice_weight: float = 1.0,
    optimization_steps: int = 32,
    learning_rate: float = 2e-2,
) -> PCSTemplateState:
    _require_pymatgen()
    device = frac_coords.device
    dtype = frac_coords.dtype

    current_structure = _build_vanilla_structure(
        frac_coords=frac_coords,
        atomic_numbers=atomic_numbers,
        cell_matrix=cell_matrix,
    )
    _analyzer, standardized_current = standardize_structure(
        current_structure,
        standardization=state.bridge.standardization,
        symprec=state.bridge.symprec,
        angle_tolerance=state.bridge.angle_tolerance,
    )
    del _analyzer
    standardized_frac, standardized_atomic_numbers, standardized_cell, _standardized_k = (
        _standardized_target_tensors_from_structure(
            standardized_current,
            device=device,
            dtype=dtype,
        )
    )
    del _standardized_k

    centering_symbol = state.target_centering_symbol or _requested_centering_symbol(int(state.constraint.space_group))
    centering_translations = state.target_centering_translations
    if centering_translations is not None:
        centering_translations = centering_translations.to(device=device, dtype=dtype)
    raw_requested_frac, raw_requested_cell = _raw_target_in_requested_conventional_frame(
        frac_coords=torch.remainder(frac_coords, 1.0),
        cell_matrix=cell_matrix,
        centering_symbol=centering_symbol,
    )
    target_name = state.target_representation_name or "standardized"
    anchor_frac, anchor_atomic_numbers, anchor_cell, anchor_k = _target_representation_from_name(
        target_name=target_name,
        raw_requested_frac=raw_requested_frac,
        raw_requested_atomic_numbers=atomic_numbers.to(device=device, dtype=torch.long),
        raw_requested_cell=raw_requested_cell,
        standardized_frac=standardized_frac,
        standardized_atomic_numbers=standardized_atomic_numbers,
        standardized_cell=standardized_cell,
        centering_translations=centering_translations,
    )

    # Centered conventional templates can have more atoms than the vanilla KLDM graph.
    # Example: SG 227 F-centered conventional template has 24 atoms, while the KLDM
    # graph may carry the primitive/composition cell with 6 atoms.
    #
    # In that case, refreshing the proximal anchor from the current vanilla graph
    # creates an incompatible assignment. Keep the original expanded anchor instead.
    expected_template_atoms = int(state.template.total_atoms)
    refreshed_anchor_atoms = int(anchor_atomic_numbers.shape[0])

    if refreshed_anchor_atoms != expected_template_atoms:
        can_keep_existing_anchor = (
            state.anchor_frac is not None
            and state.anchor_atomic_numbers is not None
            and state.anchor_cell is not None
            and state.anchor_k is not None
            and state.anchor_assignment is not None
            and int(state.anchor_atomic_numbers.shape[0]) == expected_template_atoms
            and int(state.anchor_assignment.numel()) == expected_template_atoms
        )

        if can_keep_existing_anchor:
            return state

        raise RuntimeError(
            "PCS refreshed anchor atom count does not match the template atom count: "
            f"target_repr={target_name!r}, "
            f"template_atoms={expected_template_atoms}, "
            f"refreshed_anchor_atoms={refreshed_anchor_atoms}, "
            f"template_species={_format_species_counts(expand_wyckoff_template_torch(template=state.template, free_vars=state.free_vars).atomic_numbers)}, "
            f"refreshed_anchor_species={_format_species_counts(anchor_atomic_numbers)}."
        )

    anchor_free_vars, anchor_lattice_free_vars, _anchor_fit_objective = _optimize_template_fit(
        template=state.template,
        constraint=state.constraint,
        target_frac=anchor_frac,
        target_atomic_numbers=anchor_atomic_numbers,
        target_k=anchor_k,
        optimization_steps=max(int(optimization_steps), 1),
        learning_rate=float(learning_rate),
        coord_weight=float(coord_weight),
        lattice_weight=float(lattice_weight),
        pairdist_weight=float(pairdist_weight),
        pairdist_bins=int(pairdist_bins),
        pairdist_max_distance=float(pairdist_max_distance),
        pairdist_bandwidth=float(pairdist_bandwidth),
        steric_weight=0.0,
        steric_min_distance=0.0,
        volume_weight=0.0,
        volume_ratio_min=0.0,
        volume_ratio_max=0.0,
        k6_weight=0.0,
        freeze_lattice_free_vars=bool(state.freeze_lattice),
        init_free_vars=state.free_vars,
        init_lattice_free=state.lattice_free_vars,
    )

    expansion = expand_wyckoff_template_torch(
        template=state.template,
        free_vars=anchor_free_vars.to(device=device, dtype=dtype),
    )

    try:
        anchor_assignment = _species_assignment_indices(
            source_frac=expansion.frac_coords,
            source_atomic_numbers=expansion.atomic_numbers,
            target_frac=anchor_frac,
            target_atomic_numbers=anchor_atomic_numbers,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not refresh PCS anchor assignment. "
            f"target_repr={state.target_representation_name!r}, "
            f"anchor_repr={state.anchor_representation_name!r}, "
            f"template_atoms={int(expansion.frac_coords.shape[0])}, "
            f"anchor_atoms={int(anchor_frac.shape[0])}, "
            f"template_species={_format_species_counts(expansion.atomic_numbers)}, "
            f"anchor_species={_format_species_counts(anchor_atomic_numbers)}. "
            "Refusing to reuse a stale assignment because it can corrupt "
            "expanded/conventional-vs-primitive indexing."
        ) from exc

    if anchor_assignment.ndim != 1:
        raise RuntimeError(
            "PCS anchor assignment must be a vector, "
            f"got shape={tuple(anchor_assignment.shape)}."
        )

    if int(anchor_assignment.numel()) != int(expansion.frac_coords.shape[0]):
        raise RuntimeError(
            "PCS anchor assignment has wrong length: "
            f"assignment_len={int(anchor_assignment.numel())}, "
            f"template_atoms={int(expansion.frac_coords.shape[0])}, "
            f"anchor_atoms={int(anchor_frac.shape[0])}."
        )

    if anchor_assignment.numel() > 0 and int(anchor_assignment.max().item()) >= int(anchor_frac.shape[0]):
        raise RuntimeError(
            "PCS anchor assignment indexes outside the anchor target: "
            f"max_assignment={int(anchor_assignment.max().item())}, "
            f"anchor_atoms={int(anchor_frac.shape[0])}, "
            f"template_atoms={int(expansion.frac_coords.shape[0])}."
        )

    anchor_pairdist_hist = None
    anchor_pairdist_bin_centers = None
    if float(pairdist_weight) > 0.0:
        anchor_pairdist_hist, anchor_pairdist_bin_centers = _build_target_pairdist_cache(
            target_frac=anchor_frac,
            target_cell=anchor_cell,
            bins=pairdist_bins,
            max_distance=pairdist_max_distance,
            bandwidth=pairdist_bandwidth,
        )

    return replace(
        state,
        anchor_frac=anchor_frac.detach().clone(),
        anchor_atomic_numbers=anchor_atomic_numbers.detach().clone(),
        anchor_cell=anchor_cell.detach().clone(),
        anchor_k=anchor_k.detach().clone(),
        anchor_assignment=anchor_assignment.detach().clone(),
        anchor_free_vars=anchor_free_vars.detach().clone(),
        anchor_lattice_free_vars=anchor_lattice_free_vars.detach().clone(),
        anchor_pairdist_hist=anchor_pairdist_hist.detach().clone() if anchor_pairdist_hist is not None else None,
        anchor_pairdist_bin_centers=(
            anchor_pairdist_bin_centers.detach().clone() if anchor_pairdist_bin_centers is not None else None
        ),
        anchor_representation_name=str(target_name),
    )


def _format_species_counts(atomic_numbers: torch.Tensor) -> str:
    species_order, species_counts = composition_to_species_counts(atomic_numbers)
    return "{" + ", ".join(f"{z}:{count}" for z, count in zip(species_order, species_counts)) + "}"


def _format_template_summary(
    template: WyckoffTemplate,
    *,
    composition_key: tuple[int, tuple[int, ...], tuple[int, ...]],
    prior_score: int,
) -> str:
    labels = [site.label for site in template.site_templates]
    signature = flatten_site_signature(template)
    return (
        f"atoms={template.total_atoms} "
        f"species_counts={{{', '.join(f'{z}:{count}' for z, count in zip(template.species_order, template.species_counts))}}} "
        f"labels={labels} "
        f"signature={signature} "
        f"prior={prior_score} "
        f"composition_key={composition_key}"
    )


def _template_energy_from_state(
    *,
    template: WyckoffTemplate,
    constraint: KFamilyConstraint,
    theta: torch.Tensor,
    free_dim: int,
    target_frac: torch.Tensor,
    target_atomic_numbers: torch.Tensor,
    target_cell: torch.Tensor,
    target_k: torch.Tensor,
    coord_weight: float,
    lattice_weight: float,
    pairdist_weight: float,
    pairdist_bins: int,
    pairdist_max_distance: float,
    pairdist_bandwidth: float,
    steric_weight: float,
    steric_min_distance: float,
    volume_weight: float,
    volume_ratio_min: float,
    volume_ratio_max: float,
    k6_weight: float,
    target_assignment: torch.Tensor | None = None,
    reference_volume: float | None = None,
    reference_k6: float | None = None,
    prior_bonus: float = 0.0,
    eta: float | None = None,
    target_pairdist_hist: torch.Tensor | None = None,
    pairdist_bin_centers: torch.Tensor | None = None,
    anchor_free_vars: torch.Tensor | None = None,
    anchor_lattice_free_vars: torch.Tensor | None = None,
) -> PCSEnergyResult:
    free_vars = theta[:free_dim]
    lattice_free = theta[free_dim:]
    expansion = expand_wyckoff_template_torch(
        template=template,
        free_vars=free_vars,
    )
    if anchor_free_vars is not None:
        coord_loss = _wrapped_free_var_loss(
            source_free_vars=free_vars,
            target_free_vars=anchor_free_vars.to(device=free_vars.device, dtype=free_vars.dtype),
        )
    elif target_assignment is None:
        coord_loss = _species_matched_torus_loss(
            source_frac=expansion.frac_coords,
            source_atomic_numbers=expansion.atomic_numbers,
            target_frac=target_frac,
            target_atomic_numbers=target_atomic_numbers,
        )
    else:
        coord_loss = _fixed_assignment_torus_loss(
            source_frac=expansion.frac_coords,
            target_frac=target_frac,
            target_assignment=target_assignment,
        )
    if not torch.isfinite(coord_loss):
        inf = target_frac.new_tensor(float("inf"))
        k_projected = free_vars_to_k(lattice_free, constraint)
        projected_cell = k_to_cell_matrix(k_projected)
        zero = target_frac.new_zeros(())
        return PCSEnergyResult(
            energy=inf,
            coord_loss=coord_loss,
            lattice_loss=inf,
            pairdist_loss=inf,
            steric_loss=inf,
            volume_loss=inf,
            k6_loss=inf,
            prox_energy=inf,
            likelihood_energy=zero,
            frac_coords=expansion.frac_coords,
            k_projected=k_projected,
            projected_cell=projected_cell,
        )

    k_projected = free_vars_to_k(lattice_free, constraint)
    projected_cell = k_to_cell_matrix(k_projected)
    if anchor_lattice_free_vars is not None:
        anchor_lattice_free_vars = anchor_lattice_free_vars.to(device=lattice_free.device, dtype=lattice_free.dtype)
        if lattice_free.shape != anchor_lattice_free_vars.shape:
            raise RuntimeError("Reduced-chart lattice proximal loss requires matching shapes.")
        lattice_loss = (
            lattice_free - anchor_lattice_free_vars
        ).square().mean() if lattice_free.numel() > 0 else lattice_free.new_zeros(())
    else:
        lattice_loss = (k_projected - target_k).square().mean()
    need_pair_geometry = float(pairdist_weight) > 0.0 or float(steric_weight) > 0.0
    source_distances = (
        _periodic_pairwise_distances(frac_coords=expansion.frac_coords, cell_matrix=projected_cell)
        if need_pair_geometry
        else None
    )
    pairdist_loss = (
        _pair_distance_histogram_loss_from_distances(
            source_distances=source_distances,
            target_hist=target_pairdist_hist.to(device=target_frac.device, dtype=target_frac.dtype)
            if target_pairdist_hist is not None
            else _soft_pair_distance_histogram(
                distances=_periodic_pairwise_distances(frac_coords=target_frac, cell_matrix=target_cell),
                bin_centers=pairdist_bin_centers
                if pairdist_bin_centers is not None
                else torch.linspace(
                    0.0,
                    float(pairdist_max_distance),
                    max(int(pairdist_bins), 2),
                    device=target_frac.device,
                    dtype=target_frac.dtype,
                ),
                bandwidth=pairdist_bandwidth,
            ),
            bin_centers=pairdist_bin_centers.to(device=target_frac.device, dtype=target_frac.dtype)
            if pairdist_bin_centers is not None
            else torch.linspace(
                0.0,
                float(pairdist_max_distance),
                max(int(pairdist_bins), 2),
                device=target_frac.device,
                dtype=target_frac.dtype,
            ),
            bandwidth=pairdist_bandwidth,
        )
        if float(pairdist_weight) > 0.0 and source_distances is not None
        else target_frac.new_zeros(())
    )
    steric_loss = (
        _steric_overlap_loss(
            distances=source_distances,
            min_distance=steric_min_distance,
        )
        if float(steric_weight) > 0.0 and source_distances is not None
        else target_frac.new_zeros(())
    )
    volume_loss = (
        _volume_ratio_loss(
            projected_cell=projected_cell,
            reference_volume=reference_volume,
            min_ratio=volume_ratio_min,
            max_ratio=volume_ratio_max,
        )
        if float(volume_weight) > 0.0
        else target_frac.new_zeros(())
    )
    k6_loss = (
        _k6_reference_loss(
            k_projected=k_projected,
            reference_k6=reference_k6,
        )
        if float(k6_weight) > 0.0
        else target_frac.new_zeros(())
    )
    prox_energy = (
        float(coord_weight) * coord_loss
        + float(lattice_weight) * lattice_loss
        + float(pairdist_weight) * pairdist_loss
    )
    likelihood_energy = (
        + float(steric_weight) * steric_loss
        + float(volume_weight) * volume_loss
        + float(k6_weight) * k6_loss
    )
    if eta is not None:
        eta_sq = max(float(eta) ** 2, 1e-8)
        energy = prox_energy / (2.0 * eta_sq) + likelihood_energy
    else:
        energy = prox_energy + likelihood_energy
    energy = energy - target_frac.new_tensor(float(prior_bonus))
    return PCSEnergyResult(
        energy=energy,
        coord_loss=coord_loss,
        lattice_loss=lattice_loss,
        pairdist_loss=pairdist_loss,
        steric_loss=steric_loss,
        volume_loss=volume_loss,
        k6_loss=k6_loss,
        prox_energy=prox_energy,
        likelihood_energy=likelihood_energy,
        frac_coords=expansion.frac_coords,
        k_projected=k_projected,
        projected_cell=projected_cell,
    )


def _pcs_state_rank_key(state: PCSTemplateState) -> tuple[float, ...]:
    return (algorithm6_candidate_score(state), float(state.template_rank))


def algorithm6_candidate_score(candidate: Any) -> float:
    """Single score used for selecting algorithm6 PCS candidates (lower is better)."""
    orbit_mismatch = getattr(candidate, "orbit_mismatch", None)
    if orbit_mismatch is None:
        orbit_mismatch = getattr(candidate, "species_orbit_mismatch", 0)
    hard_penalty = 0.0
    if int(orbit_mismatch) > 0:
        hard_penalty += 1.0e6 * float(orbit_mismatch)
    if bool(getattr(candidate, "soft_physics_failed", False)):
        hard_penalty += 1.0e5
    ranking_objective = getattr(candidate, "ranking_objective", float("inf"))
    return hard_penalty + float(ranking_objective)


def pcs_state_diagnostics(
    *,
    state: PCSTemplateState,
    coord_weight: float,
    lattice_weight: float,
    pairdist_weight: float,
    pairdist_bins: int,
    pairdist_max_distance: float,
    pairdist_bandwidth: float,
    steric_weight: float,
    steric_min_distance: float,
    volume_weight: float = 0.0,
    volume_ratio_min: float = 0.0,
    volume_ratio_max: float = 0.0,
    k6_weight: float = 0.0,
    eta: float | None = None,
) -> PCSDiagnostics:
    target_frac = state.anchor_frac if state.anchor_frac is not None else state.target_frac
    target_atomic_numbers = state.anchor_atomic_numbers if state.anchor_atomic_numbers is not None else state.target_atomic_numbers
    target_k = state.anchor_k if state.anchor_k is not None else state.target_k
    target_cell = state.anchor_cell if state.anchor_cell is not None else state.target_cell
    target_assignment = state.anchor_assignment if state.anchor_assignment is not None else state.fixed_target_assignment
    target_pairdist_hist = state.anchor_pairdist_hist if state.anchor_pairdist_hist is not None else state.target_pairdist_hist
    pairdist_bin_centers = (
        state.anchor_pairdist_bin_centers
        if state.anchor_pairdist_bin_centers is not None
        else state.pairdist_bin_centers
    )
    if target_frac is None or target_atomic_numbers is None or target_k is None or target_cell is None:
        raise RuntimeError("PCS diagnostics require an anchor target state.")
    if target_assignment is None:
        raise RuntimeError("PCS diagnostics require a fixed target assignment.")

    dtype = state.free_vars.dtype
    device = state.free_vars.device
    free_dim = state.template.total_free_dims
    theta = torch.cat(
        [
            state.free_vars.to(device=device, dtype=dtype).reshape(-1),
            state.lattice_free_vars.to(device=device, dtype=dtype).reshape(-1),
        ],
        dim=0,
    )
    energy_result = _template_energy_from_state(
        template=state.template,
        constraint=state.constraint,
        theta=theta,
        free_dim=free_dim,
        target_frac=target_frac.to(device=device, dtype=dtype),
        target_atomic_numbers=target_atomic_numbers.to(device=device, dtype=torch.long),
        target_cell=target_cell.to(device=device, dtype=dtype),
        target_k=target_k.to(device=device, dtype=dtype),
        coord_weight=coord_weight,
        lattice_weight=lattice_weight,
        pairdist_weight=pairdist_weight,
        pairdist_bins=pairdist_bins,
        pairdist_max_distance=pairdist_max_distance,
        pairdist_bandwidth=pairdist_bandwidth,
        steric_weight=steric_weight,
        steric_min_distance=steric_min_distance,
        volume_weight=volume_weight,
        volume_ratio_min=volume_ratio_min,
        volume_ratio_max=volume_ratio_max,
        k6_weight=k6_weight,
        target_assignment=target_assignment.to(device=device, dtype=torch.long),
        reference_volume=state.reference_volume,
        reference_k6=state.reference_k6,
        prior_bonus=state.prior_bonus,
        eta=eta,
        target_pairdist_hist=target_pairdist_hist,
        pairdist_bin_centers=pairdist_bin_centers,
    )
    target_k = target_k.to(device=device, dtype=dtype)
    return PCSDiagnostics(
        target_k=target_k.detach().clone(),
        final_k=energy_result.k_projected.detach().clone(),
        target_cell_from_k=k_to_cell_matrix(target_k).detach().clone(),
        final_cell_from_k=energy_result.projected_cell.detach().clone(),
        coord_loss=float(energy_result.coord_loss.detach().item()),
        lattice_loss=float(energy_result.lattice_loss.detach().item()),
        pairdist_loss=float(energy_result.pairdist_loss.detach().item()),
        steric_loss=float(energy_result.steric_loss.detach().item()),
        volume_loss=float(energy_result.volume_loss.detach().item()),
        k6_loss=float(energy_result.k6_loss.detach().item()),
        prox_energy=float(energy_result.prox_energy.detach().item()),
        likelihood_energy=float(energy_result.likelihood_energy.detach().item()),
        total_energy=float(energy_result.energy.detach().item()),
    )


def pcs_projected_objective(
    *,
    state: PCSTemplateState,
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell_matrix: torch.Tensor,
    coord_weight: float,
    lattice_weight: float,
    pairdist_weight: float,
    pairdist_bins: int,
    pairdist_max_distance: float,
    pairdist_bandwidth: float,
    steric_weight: float,
    steric_min_distance: float,
    volume_weight: float = 0.0,
    volume_ratio_min: float = 0.0,
    volume_ratio_max: float = 0.0,
    k6_weight: float = 0.0,
) -> float:
    """Score a materialized/projected structure against the state's anchor target.

    This is intentionally separate from `pcs_state_diagnostics()`: once a branch
    has been materialized back into vanilla coordinates/lattice tensors, we want
    to rank the *actual rebuilt structure* rather than the pre-materialization
    latent template state.
    """
    target_frac = state.anchor_frac if state.anchor_frac is not None else state.target_frac
    target_atomic_numbers = state.anchor_atomic_numbers if state.anchor_atomic_numbers is not None else state.target_atomic_numbers
    target_k = state.anchor_k if state.anchor_k is not None else state.target_k
    target_cell = state.anchor_cell if state.anchor_cell is not None else state.target_cell
    target_assignment = state.anchor_assignment if state.anchor_assignment is not None else state.fixed_target_assignment
    target_pairdist_hist = state.anchor_pairdist_hist if state.anchor_pairdist_hist is not None else state.target_pairdist_hist
    pairdist_bin_centers = (
        state.anchor_pairdist_bin_centers
        if state.anchor_pairdist_bin_centers is not None
        else state.pairdist_bin_centers
    )
    if target_frac is None or target_atomic_numbers is None or target_k is None or target_cell is None:
        raise RuntimeError("Projected PCS objective requires an anchor target state.")

    dtype = frac_coords.dtype
    device = frac_coords.device
    frac_coords = frac_coords.to(device=device, dtype=dtype)
    atomic_numbers = atomic_numbers.to(device=device, dtype=torch.long)
    cell_matrix = cell_matrix.to(device=device, dtype=dtype)
    target_frac = target_frac.to(device=device, dtype=dtype)
    target_atomic_numbers = target_atomic_numbers.to(device=device, dtype=torch.long)
    target_k = target_k.to(device=device, dtype=dtype)
    target_cell = target_cell.to(device=device, dtype=dtype)

    # Some anchor targets live in an expanded conventional representation
    # (for example centered lattices), while the materialized branch lives in
    # the vanilla / primitive-sized representation. In those cases, projected
    # reranking is not directly comparable and we should fall back to the
    # branch's existing latent-state ranking objective.
    same_atom_count = int(frac_coords.shape[0]) == int(target_frac.shape[0])
    same_species_multiset = (
        same_atom_count
        and bool(torch.equal(torch.sort(atomic_numbers).values, torch.sort(target_atomic_numbers).values))
    )
    if not same_species_multiset:
        return float(state.ranking_objective)

    if target_assignment is None:
        coord_loss = _species_matched_torus_loss(
            source_frac=frac_coords,
            source_atomic_numbers=atomic_numbers,
            target_frac=target_frac,
            target_atomic_numbers=target_atomic_numbers,
        )
    else:
        coord_loss = _fixed_assignment_torus_loss(
            source_frac=frac_coords,
            target_frac=target_frac,
            target_assignment=target_assignment.to(device=device, dtype=torch.long),
        )
    if not torch.isfinite(coord_loss):
        return float("inf")

    projected_k = cell_to_k(cell_matrix, eps=1e-8)
    lattice_loss = (projected_k - target_k).square().mean()
    need_pair_geometry = float(pairdist_weight) > 0.0 or float(steric_weight) > 0.0
    source_distances = (
        _periodic_pairwise_distances(frac_coords=frac_coords, cell_matrix=cell_matrix)
        if need_pair_geometry
        else None
    )
    pairdist_loss = (
        _pair_distance_histogram_loss_from_distances(
            source_distances=source_distances,
            target_hist=target_pairdist_hist.to(device=device, dtype=dtype)
            if target_pairdist_hist is not None
            else _soft_pair_distance_histogram(
                distances=_periodic_pairwise_distances(frac_coords=target_frac, cell_matrix=target_cell),
                bin_centers=pairdist_bin_centers
                if pairdist_bin_centers is not None
                else torch.linspace(
                    0.0,
                    float(pairdist_max_distance),
                    max(int(pairdist_bins), 2),
                    device=device,
                    dtype=dtype,
                ),
                bandwidth=pairdist_bandwidth,
            ),
            bin_centers=pairdist_bin_centers.to(device=device, dtype=dtype)
            if pairdist_bin_centers is not None
            else torch.linspace(
                0.0,
                float(pairdist_max_distance),
                max(int(pairdist_bins), 2),
                device=device,
                dtype=dtype,
            ),
            bandwidth=pairdist_bandwidth,
        )
        if float(pairdist_weight) > 0.0 and source_distances is not None
        else target_frac.new_zeros(())
    )
    steric_loss = (
        _steric_overlap_loss(
            distances=source_distances,
            min_distance=steric_min_distance,
        )
        if float(steric_weight) > 0.0 and source_distances is not None
        else target_frac.new_zeros(())
    )
    volume_loss = (
        _volume_ratio_loss(
            projected_cell=cell_matrix,
            reference_volume=(
                state.projected_reference_volume
                if state.projected_reference_volume is not None
                else state.reference_volume
            ),
            min_ratio=volume_ratio_min,
            max_ratio=volume_ratio_max,
        )
        if float(volume_weight) > 0.0
        else target_frac.new_zeros(())
    )
    k6_loss = (
        _k6_reference_loss(
            k_projected=projected_k,
            reference_k6=(
                state.projected_reference_k6
                if state.projected_reference_k6 is not None
                else state.reference_k6
            ),
        )
        if float(k6_weight) > 0.0
        else target_frac.new_zeros(())
    )
    prox_energy = (
        float(coord_weight) * coord_loss
        + float(lattice_weight) * lattice_loss
        + float(pairdist_weight) * pairdist_loss
    )
    likelihood_energy = (
        + float(steric_weight) * steric_loss
        + float(volume_weight) * volume_loss
        + float(k6_weight) * k6_loss
    )
    energy = prox_energy + likelihood_energy
    energy = energy - target_frac.new_tensor(float(state.prior_bonus))
    return float(energy.detach().item())


def _optimize_template_fit(
    *,
    template: WyckoffTemplate,
    constraint: KFamilyConstraint,
    target_frac: torch.Tensor,
    target_atomic_numbers: torch.Tensor,
    target_k: torch.Tensor,
    optimization_steps: int,
    learning_rate: float,
    coord_weight: float,
    lattice_weight: float,
    pairdist_weight: float,
    pairdist_bins: int,
    pairdist_max_distance: float,
    pairdist_bandwidth: float,
    steric_weight: float,
    steric_min_distance: float,
    volume_weight: float,
    volume_ratio_min: float,
    volume_ratio_max: float,
    k6_weight: float,
    freeze_lattice_free_vars: bool,
    init_free_vars: torch.Tensor | None = None,
    init_lattice_free: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    device = target_frac.device
    dtype = target_frac.dtype
    free_dim = template.total_free_dims
    target_cell = k_to_cell_matrix(target_k)
    reference_volume = float(_cell_volume(target_cell).detach().item())
    reference_k6 = float(target_k[..., -1].detach().item()) if target_k.numel() > 0 else None
    target_pairdist_hist = None
    pairdist_bin_centers = None
    if float(pairdist_weight) > 0.0:
        target_pairdist_hist, pairdist_bin_centers = _build_target_pairdist_cache(
            target_frac=target_frac,
            target_cell=target_cell,
            bins=pairdist_bins,
            max_distance=pairdist_max_distance,
            bandwidth=pairdist_bandwidth,
        )
    best_theta: torch.Tensor | None = None
    best_energy = float("inf")
    n_restarts = 6 if init_free_vars is None else 1

    for restart_idx in range(int(n_restarts)):
        if init_free_vars is None:
            free_vars = sample_random_free_vars(template, device=device, dtype=dtype).clone().detach()
        else:
            free_vars = init_free_vars.to(device=device, dtype=dtype).clone().detach()
        if init_lattice_free is None:
            lattice_free = k_to_free_vars(target_k, constraint).clone().detach()
        else:
            lattice_free = init_lattice_free.to(device=device, dtype=dtype).clone().detach()
        fixed_lattice_free = lattice_free.clone().detach()

        if restart_idx > 0 and init_lattice_free is None and lattice_free.numel() > 0:
            lattice_free = lattice_free + 1.0e-3 * torch.randn_like(lattice_free)

        theta = torch.cat([free_vars.reshape(-1), lattice_free.reshape(-1)], dim=0).requires_grad_(True)
        lr_scale = 1.0 if restart_idx < 2 else (0.2 if restart_idx < 4 else 0.05)
        optimizer = torch.optim.Adam([theta], lr=float(learning_rate) * lr_scale)
        last_finite_theta = theta.detach().clone()

        # Important:
        # The initial lattice_free is k_to_free_vars(target_k, constraint) when
        # init_lattice_free is None. That is often already the best lattice.
        # If we only save the post-optimization theta, the first Adam steps can
        # collapse the lattice volume and we lose the good initial state.
        with torch.no_grad():
            initial_energy_result = _template_energy_from_state(
                template=template,
                constraint=constraint,
                theta=last_finite_theta,
                free_dim=free_dim,
                target_frac=target_frac,
                target_atomic_numbers=target_atomic_numbers,
                target_cell=target_cell,
                target_k=target_k,
                coord_weight=coord_weight,
                lattice_weight=lattice_weight,
                pairdist_weight=pairdist_weight,
                pairdist_bins=pairdist_bins,
                pairdist_max_distance=pairdist_max_distance,
                pairdist_bandwidth=pairdist_bandwidth,
                steric_weight=steric_weight,
                steric_min_distance=steric_min_distance,
                volume_weight=volume_weight,
                volume_ratio_min=volume_ratio_min,
                volume_ratio_max=volume_ratio_max,
                k6_weight=k6_weight,
                reference_volume=reference_volume,
                reference_k6=reference_k6,
                target_pairdist_hist=target_pairdist_hist,
                pairdist_bin_centers=pairdist_bin_centers,
            )
            initial_energy = initial_energy_result.energy
            if torch.isfinite(initial_energy) and float(initial_energy.item()) < best_energy:
                best_energy = float(initial_energy.item())
                best_theta = last_finite_theta.clone()

        for _step_idx in range(int(optimization_steps)):
            optimizer.zero_grad(set_to_none=True)
            energy_result = _template_energy_from_state(
                template=template,
                constraint=constraint,
                theta=theta,
                free_dim=free_dim,
                target_frac=target_frac,
                target_atomic_numbers=target_atomic_numbers,
                target_cell=target_cell,
                target_k=target_k,
                coord_weight=coord_weight,
                lattice_weight=lattice_weight,
                pairdist_weight=pairdist_weight,
                pairdist_bins=pairdist_bins,
                pairdist_max_distance=pairdist_max_distance,
                pairdist_bandwidth=pairdist_bandwidth,
                steric_weight=steric_weight,
                steric_min_distance=steric_min_distance,
                volume_weight=volume_weight,
                volume_ratio_min=volume_ratio_min,
                volume_ratio_max=volume_ratio_max,
                k6_weight=k6_weight,
                reference_volume=reference_volume,
                reference_k6=reference_k6,
                target_pairdist_hist=target_pairdist_hist,
                pairdist_bin_centers=pairdist_bin_centers,
            )
            energy = energy_result.energy
            if not torch.isfinite(energy):
                break
            energy.backward()
            if freeze_lattice_free_vars and theta.grad is not None and theta.shape[0] > free_dim:
                theta.grad[free_dim:] = 0.0
            torch.nn.utils.clip_grad_norm_([theta], max_norm=10.0)
            optimizer.step()
            with torch.no_grad():
                theta.data = torch.nan_to_num(theta.data, nan=0.0, posinf=0.0, neginf=0.0)
                theta.data = theta.data.clamp_(-8.0, 8.0)
                if freeze_lattice_free_vars and theta.shape[0] > free_dim:
                    theta.data[free_dim:] = fixed_lattice_free
            last_finite_theta = theta.detach().clone()
            with torch.no_grad():
                step_energy_result = _template_energy_from_state(
                    template=template,
                    constraint=constraint,
                    theta=last_finite_theta,
                    free_dim=free_dim,
                    target_frac=target_frac,
                    target_atomic_numbers=target_atomic_numbers,
                    target_cell=target_cell,
                    target_k=target_k,
                    coord_weight=coord_weight,
                    lattice_weight=lattice_weight,
                    pairdist_weight=pairdist_weight,
                    pairdist_bins=pairdist_bins,
                    pairdist_max_distance=pairdist_max_distance,
                    pairdist_bandwidth=pairdist_bandwidth,
                    steric_weight=steric_weight,
                    steric_min_distance=steric_min_distance,
                    volume_weight=volume_weight,
                    volume_ratio_min=volume_ratio_min,
                    volume_ratio_max=volume_ratio_max,
                    k6_weight=k6_weight,
                    reference_volume=reference_volume,
                    reference_k6=reference_k6,
                    target_pairdist_hist=target_pairdist_hist,
                    pairdist_bin_centers=pairdist_bin_centers,
                )
                step_energy = step_energy_result.energy
                if torch.isfinite(step_energy) and float(step_energy.item()) < best_energy:
                    best_energy = float(step_energy.item())
                    best_theta = last_finite_theta.clone()

        with torch.no_grad():
            energy_result = _template_energy_from_state(
                template=template,
                constraint=constraint,
                theta=last_finite_theta,
                free_dim=free_dim,
                target_frac=target_frac,
                target_atomic_numbers=target_atomic_numbers,
                target_cell=target_cell,
                target_k=target_k,
                coord_weight=coord_weight,
                lattice_weight=lattice_weight,
                pairdist_weight=pairdist_weight,
                pairdist_bins=pairdist_bins,
                pairdist_max_distance=pairdist_max_distance,
                pairdist_bandwidth=pairdist_bandwidth,
                steric_weight=steric_weight,
                steric_min_distance=steric_min_distance,
                volume_weight=volume_weight,
                volume_ratio_min=volume_ratio_min,
                volume_ratio_max=volume_ratio_max,
                k6_weight=k6_weight,
                reference_volume=reference_volume,
                reference_k6=reference_k6,
                target_pairdist_hist=target_pairdist_hist,
                pairdist_bin_centers=pairdist_bin_centers,
            )
            energy = energy_result.energy
            if torch.isfinite(energy) and float(energy.item()) < best_energy:
                best_energy = float(energy.item())
                best_theta = last_finite_theta.clone()

    if best_theta is None:
        empty_free = torch.empty((free_dim,), device=device, dtype=dtype)
        empty_lattice = torch.empty((len(constraint.free_indices),), device=device, dtype=dtype)
        return empty_free, empty_lattice, float("inf")

    return (
        best_theta[:free_dim].clone(),
        best_theta[free_dim:].clone(),
        best_energy,
    )


def select_requested_template_state(
    *,
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell_matrix: torch.Tensor,
    space_group_number: int,
    standardization: str = "conventional",
    symprec: float = 1e-2,
    angle_tolerance: float = 5.0,
    max_templates: int = 256,
    template_eval_limit: int = 32,
    optimization_steps: int = 150,
    learning_rate: float = 5e-2,
    coord_weight: float = 1.0,
    lattice_weight: float = 0.25,
    pairdist_weight: float = 0.0,
    pairdist_bins: int = 32,
    pairdist_max_distance: float = 8.0,
    pairdist_bandwidth: float = 0.25,
    steric_weight: float = 0.0,
    steric_min_distance: float = 0.8,
    volume_weight: float = 0.0,
    volume_ratio_min: float = 0.0,
    volume_ratio_max: float = 0.0,
    k6_weight: float = 0.0,
    freeze_lattice_free_vars: bool = False,
    quick_templates: bool = False,
    template_prior: TemplatePrior | None = None,
    template_prior_weight: float = 1.0,
    debug_template_candidates: bool = False,
    debug_label: str | None = None,
    oracle_reference_structure=None,
    oracle_fit_structure=None,
) -> PCSTemplateState:
    return select_requested_template_states(
        frac_coords=frac_coords,
        atomic_numbers=atomic_numbers,
        cell_matrix=cell_matrix,
        space_group_number=space_group_number,
        standardization=standardization,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        max_templates=max_templates,
        template_eval_limit=template_eval_limit,
        optimization_steps=optimization_steps,
        learning_rate=learning_rate,
        coord_weight=coord_weight,
        lattice_weight=lattice_weight,
        pairdist_weight=pairdist_weight,
        pairdist_bins=pairdist_bins,
        pairdist_max_distance=pairdist_max_distance,
        pairdist_bandwidth=pairdist_bandwidth,
        steric_weight=steric_weight,
        steric_min_distance=steric_min_distance,
        volume_weight=volume_weight,
        volume_ratio_min=volume_ratio_min,
        volume_ratio_max=volume_ratio_max,
        k6_weight=k6_weight,
        freeze_lattice_free_vars=freeze_lattice_free_vars,
        quick_templates=quick_templates,
        top_k=1,
        template_prior=template_prior,
        template_prior_weight=template_prior_weight,
        debug_template_candidates=debug_template_candidates,
        debug_label=debug_label,
        oracle_reference_structure=oracle_reference_structure,
        oracle_fit_structure=oracle_fit_structure,
    )[0]


def initialize_constrained_template_states(
    *,
    reference_frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell_matrix: torch.Tensor,
    space_group_number: int,
    standardization: str = "conventional",
    symprec: float = 1e-2,
    angle_tolerance: float = 5.0,
    max_templates: int = 256,
    template_eval_limit: int = 32,
    quick_templates: bool = False,
    top_k: int = 1,
    template_prior: TemplatePrior | None = None,
    template_prior_weight: float = 1.0,
    debug_template_candidates: bool = False,
    debug_label: str | None = None,
    freeze_lattice_free_vars: bool = False,
    oracle_reference_structure=None,
    oracle_fit_structure=None,
) -> list[PCSTemplateState]:
    """Initialize DiffCSP++ chart branches without fitting an unconstrained sample.

    This is the CSP++/DPnP initialization path: choose discrete Wyckoff templates
    from `(composition, space group)`, sample their continuous chart variables,
    and use the KLDM sample only as a frame/scale reference for the later DDS
    prior move.
    """
    _require_pymatgen()
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    device = reference_frac_coords.device
    dtype = reference_frac_coords.dtype
    requested_sg = int(space_group_number)

    vanilla_structure = _build_vanilla_structure(
        frac_coords=reference_frac_coords,
        atomic_numbers=atomic_numbers,
        cell_matrix=cell_matrix,
    )
    bridge_source = "sample"
    try:
        bridge = build_symmetry_frame_bridge(
            vanilla_structure=vanilla_structure,
            standardization=standardization,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
    except Exception as sample_bridge_exc:
        oracle_bridge_structure = oracle_fit_structure or oracle_reference_structure
        if oracle_bridge_structure is None:
            raise
        try:
            bridge = build_symmetry_frame_bridge(
                vanilla_structure=oracle_bridge_structure,
                standardization=standardization,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            )
        except Exception as oracle_bridge_exc:
            raise RuntimeError(
                "symmetry_frame_bridge_failed "
                f"requested_sg={requested_sg} atoms={int(atomic_numbers.numel())} "
                f"sample_error={type(sample_bridge_exc).__name__}:{sample_bridge_exc} "
                f"oracle_error={type(oracle_bridge_exc).__name__}:{oracle_bridge_exc}"
            ) from oracle_bridge_exc
        bridge_source = "oracle"

    target_centering = _requested_centering_symbol(requested_sg)
    target_centering_translations = _centering_translations(target_centering, device=device, dtype=dtype)
    conventional_atomic_numbers = requested_conventional_atomic_numbers(
        atomic_numbers,
        space_group_number=requested_sg,
    ).to(device=device, dtype=torch.long)
    use_centering_expansion = int(conventional_atomic_numbers.shape[0]) != int(atomic_numbers.shape[0])

    raw_requested_frac, raw_requested_cell = _raw_target_in_requested_conventional_frame(
        frac_coords=torch.remainder(reference_frac_coords, 1.0),
        cell_matrix=cell_matrix,
        centering_symbol=target_centering,
    )
    del raw_requested_frac
    chart_cell = raw_requested_cell if use_centering_expansion else cell_matrix

    constraint = space_group_k_constraint(
        space_group_number=requested_sg,
        device=device,
        dtype=dtype,
    )
    chart_k_seed = cell_to_k(chart_cell, eps=1e-8)
    lattice_free = k_to_free_vars(chart_k_seed, constraint).detach().clone()
    chart_k = free_vars_to_k(lattice_free, constraint).detach().clone()
    chart_cell_projected = k_to_cell_matrix(chart_k).detach().clone()

    projected_reference_k = cell_to_k(cell_matrix, eps=1e-8).detach().clone()
    projected_reference_volume = float(_cell_volume(cell_matrix).detach().item())
    projected_reference_k6 = (
        float(projected_reference_k[..., -1].detach().item())
        if projected_reference_k.numel() > 0
        else None
    )
    chart_reference_volume = float(_cell_volume(chart_cell_projected).detach().item())
    chart_reference_k6 = float(chart_k[..., -1].detach().item()) if chart_k.numel() > 0 else None

    species_order, species_counts = composition_to_species_counts(conventional_atomic_numbers)
    composition_key = (requested_sg, species_order, species_counts)
    templates = extract_wyckoff_templates(
        space_group_number=requested_sg,
        atomic_numbers=conventional_atomic_numbers,
        max_templates=max_templates,
        quick=quick_templates,
    )
    if not templates:
        raise RuntimeError(
            "CSP++ constrained initialization could not enumerate Wyckoff templates "
            f"for requested space group {requested_sg} and composition "
            f"{_format_species_counts(conventional_atomic_numbers)}."
        )

    ranked_templates = sorted(
        enumerate(templates, start=1),
        key=lambda item: (
            -template_prior_score(
                prior=template_prior,
                key=composition_key,
                signature=flatten_site_signature(item[1]),
            ),
            item[1].total_free_dims,
            item[1].total_sites,
            item[1].total_atoms,
            item[0],
        ),
    )[: max(1, min(int(template_eval_limit), len(templates)))]

    if debug_template_candidates:
        debug_parts = [f"requested_sg={requested_sg}"]
        if debug_label:
            debug_parts.insert(0, debug_label)
        debug_parts.extend(
            [
                "target_signature_source=csppp_constrained",
                "fit_target_source=none",
                f"bridge_source={bridge_source}",
                f"raw_requested_volume={float(_cell_volume(raw_requested_cell).detach().item()):.6f}",
                f"projected_reference_volume={projected_reference_volume:.6f}",
                f"chart_volume={chart_reference_volume:.6f}",
                "target_signature=na",
            ]
        )
        print("algorithm6_template_pool " + " ".join(debug_parts), flush=True)

    states: list[PCSTemplateState] = []
    for pool_idx, (template_rank, template) in enumerate(ranked_templates, start=1):
        prior_score = template_prior_score(
            prior=template_prior,
            key=composition_key,
            signature=flatten_site_signature(template),
        )
        prior_bonus = float(template_prior_weight) * float(np.log1p(max(prior_score, 0)))
        free_vars = sample_random_free_vars(template, device=device, dtype=dtype)
        expansion = expand_wyckoff_template_torch(template=template, free_vars=free_vars)
        if int(expansion.atomic_numbers.shape[0]) != int(conventional_atomic_numbers.shape[0]):
            continue
        if not _torch_atomic_multiset_matches(expansion.atomic_numbers, conventional_atomic_numbers):
            continue

        fixed_assignment = torch.arange(
            int(expansion.frac_coords.shape[0]),
            device=device,
            dtype=torch.long,
        )
        target_name = "raw_requested_expanded" if use_centering_expansion else "standardized"
        target_pairdist_hist = None
        pairdist_bin_centers = None
        template_species_orbit_signature = _template_species_orbit_signature(template)
        state = PCSTemplateState(
            template=template,
            constraint=constraint,
            bridge=bridge,
            free_vars=free_vars.detach().clone(),
            lattice_free_vars=lattice_free.detach().clone(),
            objective=0.0,
            ranking_objective=float(-prior_bonus),
            template_rank=int(template_rank),
            candidate_count=int(len(templates)),
            target_centering_symbol=target_centering if use_centering_expansion else None,
            target_centering_translations=target_centering_translations if use_centering_expansion else None,
            target_frac=expansion.frac_coords.detach().clone(),
            target_atomic_numbers=expansion.atomic_numbers.detach().clone(),
            target_cell=chart_cell_projected.detach().clone(),
            target_k=chart_k.detach().clone(),
            fixed_target_assignment=fixed_assignment.detach().clone(),
            target_pairdist_hist=target_pairdist_hist,
            pairdist_bin_centers=pairdist_bin_centers,
            anchor_frac=expansion.frac_coords.detach().clone(),
            anchor_atomic_numbers=expansion.atomic_numbers.detach().clone(),
            anchor_cell=chart_cell_projected.detach().clone(),
            anchor_k=chart_k.detach().clone(),
            anchor_assignment=fixed_assignment.detach().clone(),
            anchor_free_vars=free_vars.detach().clone(),
            anchor_lattice_free_vars=lattice_free.detach().clone(),
            anchor_pairdist_hist=None,
            anchor_pairdist_bin_centers=None,
            anchor_representation_name=target_name,
            reference_volume=chart_reference_volume,
            reference_k6=chart_reference_k6,
            projected_reference_volume=(
                projected_reference_volume if use_centering_expansion else chart_reference_volume
            ),
            projected_reference_k6=(
                projected_reference_k6 if use_centering_expansion else chart_reference_k6
            ),
            prior_score=int(prior_score),
            prior_bonus=prior_bonus,
            freeze_lattice=bool(freeze_lattice_free_vars),
            target_species_orbit_signature=(),
            template_species_orbit_signature=template_species_orbit_signature,
            species_orbit_mismatch=0,
            orbit_reference_is_oracle=False,
            orbit_reference_rank_first=False,
            target_representation_name=target_name,
        )
        states.append(state)

        if debug_template_candidates:
            signature_labels = [f"{Element.from_Z(int(z)).symbol}@{label}" for z, label in template_species_orbit_signature]
            print(
                f"algorithm6_template_pool_item {debug_label or 'graph=?'} "
                f"idx={pool_idx} template_rank={int(template_rank)} "
                f"orbit_mismatch=0 prior_score={int(prior_score)} "
                f"total_sites={int(template.total_sites)} total_atoms={int(template.total_atoms)} "
                f"signature={signature_labels}",
                flush=True,
            )
            print(
                f"algorithm6_template_candidate {debug_label or 'graph=?'} "
                f"idx={pool_idx} template_rank={int(template_rank)} orbit_mismatch=0 "
                f"target_repr={target_name} objective=0.000000 "
                f"objective_minus_bonus={float(-prior_bonus):.6f} "
                f"prior_score={int(prior_score)} signature={signature_labels}",
                flush=True,
            )

    if not states:
        raise RuntimeError(
            "CSP++ constrained initialization enumerated templates, but none matched the "
            f"requested conventional composition {_format_species_counts(conventional_atomic_numbers)}."
        )

    states.sort(key=_pcs_state_rank_key)
    return states[: int(top_k)]


def select_requested_template_states(
    *,
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell_matrix: torch.Tensor,
    space_group_number: int,
    standardization: str = "conventional",
    symprec: float = 1e-2,
    angle_tolerance: float = 5.0,
    max_templates: int = 256,
    template_eval_limit: int = 32,
    optimization_steps: int = 150,
    learning_rate: float = 5e-2,
    coord_weight: float = 1.0,
    lattice_weight: float = 0.25,
    pairdist_weight: float = 0.0,
    pairdist_bins: int = 32,
    pairdist_max_distance: float = 8.0,
    pairdist_bandwidth: float = 0.25,
    steric_weight: float = 0.0,
    steric_min_distance: float = 0.8,
    volume_weight: float = 0.0,
    volume_ratio_min: float = 0.0,
    volume_ratio_max: float = 0.0,
    k6_weight: float = 0.0,
    freeze_lattice_free_vars: bool = False,
    quick_templates: bool = False,
    top_k: int = 1,
    template_prior: TemplatePrior | None = None,
    template_prior_weight: float = 1.0,
    debug_template_candidates: bool = False,
    debug_label: str | None = None,
    oracle_reference_structure=None,
    oracle_fit_structure=None,
) -> list[PCSTemplateState]:
    _require_pymatgen()
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    device = frac_coords.device
    dtype = frac_coords.dtype

    vanilla_structure = _build_vanilla_structure(
        frac_coords=frac_coords,
        atomic_numbers=atomic_numbers,
        cell_matrix=cell_matrix,
    )
    bridge_source = "sample"
    try:
        bridge = build_symmetry_frame_bridge(
            vanilla_structure=vanilla_structure,
            standardization=standardization,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
    except Exception as sample_bridge_exc:
        oracle_bridge_structure = oracle_fit_structure or oracle_reference_structure
        if oracle_bridge_structure is None:
            raise
        try:
            bridge = build_symmetry_frame_bridge(
                vanilla_structure=oracle_bridge_structure,
                standardization=standardization,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            )
        except Exception as oracle_bridge_exc:
            raise RuntimeError(
                "symmetry_frame_bridge_failed "
                f"requested_sg={int(space_group_number)} atoms={int(atomic_numbers.numel())} "
                f"sample_error={type(sample_bridge_exc).__name__}:{sample_bridge_exc} "
                f"oracle_error={type(oracle_bridge_exc).__name__}:{oracle_bridge_exc}"
            ) from oracle_bridge_exc
        bridge_source = "oracle"
    standardized_frac, standardized_atomic_numbers, _standardized_cell, current_k = _standardized_target_tensors(
        bridge,
        device=device,
        dtype=dtype,
    )
    fit_standardized_frac = standardized_frac
    fit_standardized_atomic_numbers = standardized_atomic_numbers
    fit_target_source = "bridge"
    if oracle_fit_structure is not None:
        try:
            _oracle_fit_analyzer, oracle_fit_standardized = standardize_structure(
                oracle_fit_structure,
                standardization=standardization,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            )
            del _oracle_fit_analyzer
            fit_standardized_frac, fit_standardized_atomic_numbers, _fit_standardized_cell, _fit_current_k = (
                _standardized_target_tensors_from_structure(
                    oracle_fit_standardized,
                    device=device,
                    dtype=dtype,
                )
            )
            del _fit_standardized_cell, _fit_current_k
            fit_target_source = "oracle"
        except Exception:
            fit_target_source = "bridge"
    requested_sg = int(space_group_number)
    use_nonoracle_orbit_target = bool(oracle_reference_structure is not None)

    target_centering = _requested_centering_symbol(requested_sg)
    target_centering_translations = _centering_translations(target_centering, device=device, dtype=dtype)
    use_centering_expansion = (
        requested_conventional_atomic_numbers(
            atomic_numbers,
            space_group_number=requested_sg,
        ).shape[0]
        != fit_standardized_atomic_numbers.shape[0]
    )
    raw_requested_frac, _raw_requested_cell = _raw_target_in_requested_conventional_frame(
        frac_coords=frac_coords,
        cell_matrix=cell_matrix,
        centering_symbol=target_centering,
    )
    target_frac_for_fit, target_atomic_numbers_for_fit = _expand_target_by_translations(
        fit_standardized_frac,
        fit_standardized_atomic_numbers,
        target_centering_translations if use_centering_expansion else None,
    )
    raw_requested_frac_for_fit, raw_requested_atomic_numbers_for_fit = _expand_target_by_translations(
        raw_requested_frac,
        atomic_numbers,
        target_centering_translations if use_centering_expansion else None,
    )
    if use_nonoracle_orbit_target:
        raw_requested_structure_for_signature = _build_structure_from_standardized_projection(
            frac_coords=raw_requested_frac_for_fit,
            atomic_numbers=raw_requested_atomic_numbers_for_fit,
            cell_matrix=_raw_requested_cell,
        )
        target_species_orbit_signature, target_signature_backend = _structure_species_orbit_signature_with_source(
            structure=raw_requested_structure_for_signature,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        orbit_reference_signature = target_species_orbit_signature
        orbit_reference_source = f"bridge_{target_signature_backend}"
        orbit_reference_rank_first = bool(
            target_signature_backend in {"spglib", "pyxtal"} and target_species_orbit_signature
        )
    else:
        target_species_orbit_signature = ()
        target_signature_backend = "disabled_nonoracle"
        orbit_reference_signature = ()
        orbit_reference_source = "disabled_nonoracle"
        orbit_reference_rank_first = False
    if oracle_reference_structure is not None:
        try:
            _oracle_analyzer, oracle_standardized = standardize_structure(
                oracle_reference_structure,
                standardization=standardization,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            )
            del _oracle_analyzer
            orbit_reference_signature, oracle_signature_backend = _structure_species_orbit_signature_with_source(
                structure=oracle_standardized,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            )
            orbit_reference_source = f"oracle_{oracle_signature_backend}"
            orbit_reference_rank_first = bool(orbit_reference_signature)
        except Exception:
            orbit_reference_signature = target_species_orbit_signature
            orbit_reference_source = f"bridge_{target_signature_backend}"
            orbit_reference_rank_first = bool(
                target_signature_backend in {"spglib", "pyxtal"} and target_species_orbit_signature
            )
    target_representations = _target_representations(
        raw_requested_frac=raw_requested_frac,
        raw_requested_atomic_numbers=atomic_numbers,
        raw_requested_cell=_raw_requested_cell,
        raw_requested_frac_for_fit=raw_requested_frac_for_fit,
        raw_requested_atomic_numbers_for_fit=raw_requested_atomic_numbers_for_fit,
        standardized_frac=fit_standardized_frac,
        standardized_atomic_numbers=fit_standardized_atomic_numbers,
        standardized_cell=_standardized_cell,
        target_frac_for_fit=target_frac_for_fit,
        target_atomic_numbers_for_fit=target_atomic_numbers_for_fit,
        include_requested_frame=bool(oracle_reference_structure is not None or oracle_fit_structure is not None),
    )
    if debug_template_candidates:
        for target_name, _target_frac, target_atomic_numbers, target_cell in target_representations:
            target_k = cell_to_k(target_cell, eps=1e-8).detach().clone()
            target_k6 = float(target_k[..., -1].detach().item()) if target_k.numel() > 0 else float("nan")
            print(
                f"algorithm6_target_representation {debug_label or 'graph=?'} "
                f"name={target_name} num_atoms={int(target_atomic_numbers.shape[0])} "
                f"volume={float(_cell_volume(target_cell).detach().item()):.6f} "
                f"k6={target_k6:.6f}",
                flush=True,
            )

    composition_variants = _candidate_composition_variants(
        raw_atomic_numbers=atomic_numbers,
        standardized_atomic_numbers=fit_standardized_atomic_numbers,
        target_atomic_numbers_for_fit=target_atomic_numbers_for_fit,
        requested_sg=requested_sg,
    )
    if not use_nonoracle_orbit_target:
        canonical_target_atomic_numbers = target_representations[0][2].detach().clone().to(dtype=torch.long)
        species_order, species_counts = composition_to_species_counts(canonical_target_atomic_numbers)
        composition_variants = [
            (
                canonical_target_atomic_numbers,
                (int(requested_sg), species_order, species_counts),
            )
        ]


    ranked_templates: list[tuple[WyckoffTemplate, int, tuple[int, tuple[int, ...], tuple[int, ...]]]] = []
    seen_template_signatures: set[tuple[tuple[int, str], ...]] = set()
    total_templates_found = 0
    for variant_atomic_numbers, composition_key in composition_variants:
        templates = extract_wyckoff_templates(
            space_group_number=requested_sg,
            atomic_numbers=variant_atomic_numbers,
            max_templates=max_templates,
            quick=quick_templates,
        )
        total_templates_found += len(templates)
        if not templates:
            continue
        local_ranked = sorted(
            templates,
            key=lambda template: (
                -template_prior_score(
                    prior=template_prior,
                    key=composition_key,
                    signature=flatten_site_signature(template),
                ),
                template.total_free_dims,
                template.total_sites,
                template.total_atoms,
            ),
        )
        for local_rank, template in enumerate(local_ranked, start=1):
            signature = flatten_site_signature(template)
            if signature in seen_template_signatures:
                continue
            seen_template_signatures.add(signature)
            ranked_templates.append((template, local_rank, composition_key))

    if not ranked_templates:
        raise RuntimeError(
            "PCS template enumeration could not find any PyXtal templates for the requested "
            f"space group {requested_sg} in the standardized frame."
        )

    def _pre_eval_template_rank_key(
        entry: tuple[WyckoffTemplate, int, tuple[int, tuple[int, ...], tuple[int, ...]]],
    ) -> tuple[float, ...]:
        template, template_rank, composition_key = entry

        prior_score = template_prior_score(
            prior=template_prior,
            key=composition_key,
            signature=flatten_site_signature(template),
        )

        orbit_mismatch = _species_orbit_mismatch_count(
            template_signature=_template_species_orbit_signature(template),
            target_signature=orbit_reference_signature,
        )

        # Important:
        # In oracle/debug mode, rank by the true orbit mismatch BEFORE truncating
        # by template_eval_limit. Otherwise the correct target template can be
        # thrown away before it is ever evaluated.
        if bool(orbit_reference_rank_first) or bool(str(orbit_reference_source).startswith("oracle")):
            return (
                float(orbit_mismatch),
                -float(prior_score),
                float(template.total_free_dims),
                float(template.total_sites),
                float(template.total_atoms),
                float(template_rank),
            )

        # Non-oracle mode:
        # Keep the dataset prior first, but use orbit mismatch as a tie-breaker.
        return (
            -float(prior_score),
            float(orbit_mismatch),
            float(template.total_free_dims),
            float(template.total_sites),
            float(template.total_atoms),
            float(template_rank),
        )

    ranked_templates = sorted(
        ranked_templates,
        key=_pre_eval_template_rank_key,
    )[: max(1, min(int(template_eval_limit), len(ranked_templates)))]

    if debug_template_candidates:
        debug_parts = [f"requested_sg={requested_sg}"]
        if debug_label:
            debug_parts.insert(0, debug_label)
        debug_parts.append(
            f"target_signature_source={orbit_reference_source}"
        )
        debug_parts.append(f"fit_target_source={fit_target_source}")
        debug_parts.append(f"init_pairdist_weight={float(pairdist_weight):.6f}")
        debug_parts.append(f"volume_weight={float(volume_weight):.6f}")
        debug_parts.append(f"k6_weight={float(k6_weight):.6f}")
        debug_parts.append(
            f"raw_requested_volume={float(_cell_volume(_raw_requested_cell).detach().item()):.6f}"
        )
        debug_parts.append(
            f"standardized_volume={float(_cell_volume(_standardized_cell).detach().item()):.6f}"
        )
        debug_parts.append(
            "target_signature="
            + (
                "na"
                if not orbit_reference_signature
                else str([f"{Element.from_Z(int(z)).symbol}@{label}" for z, label in orbit_reference_signature])
            )
        )
        print("algorithm6_template_pool " + " ".join(debug_parts), flush=True)
        for pool_idx, (template, template_rank, composition_key) in enumerate(ranked_templates, start=1):
            template_signature = _template_species_orbit_signature(template)
            mismatch = _species_orbit_mismatch_count(
                template_signature=template_signature,
                target_signature=orbit_reference_signature,
            )
            prior_score = template_prior_score(
                prior=template_prior,
                key=composition_key,
                signature=flatten_site_signature(template),
            )
            signature_labels = [f"{Element.from_Z(int(z)).symbol}@{label}" for z, label in template_signature]
            print(
                f"algorithm6_template_pool_item {debug_label or 'graph=?'} "
                f"idx={pool_idx} template_rank={int(template_rank)} "
                f"orbit_mismatch={int(mismatch)} prior_score={int(prior_score)} "
                f"total_sites={int(template.total_sites)} total_atoms={int(template.total_atoms)} "
                f"signature={signature_labels}",
                flush=True,
            )

    constraint = space_group_k_constraint(
        space_group_number=requested_sg,
        device=device,
        dtype=dtype,
    )
    candidate_states: list[PCSTemplateState] = []
    reject_reasons = {
        "nonfinite_objective": 0,
        "atom_count_mismatch": 0,
        "atomic_multiset_mismatch": 0,
        "assignment_failed": 0,
    }

    for template, template_rank, composition_key in ranked_templates:
        best_state: PCSTemplateState | None = None
        template_species_orbit_signature = _template_species_orbit_signature(template)
        species_orbit_mismatch = _species_orbit_mismatch_count(
            template_signature=template_species_orbit_signature,
            target_signature=orbit_reference_signature,
        )
        if not use_nonoracle_orbit_target:
            species_orbit_mismatch = 0
        for target_name, candidate_target_frac, candidate_target_atomic_numbers, candidate_target_cell in target_representations:
            free_vars, lattice_free, objective = _optimize_template_fit(
                template=template,
                constraint=constraint,
                target_frac=candidate_target_frac,
                target_atomic_numbers=candidate_target_atomic_numbers,
                target_k=cell_to_k(candidate_target_cell, eps=1e-8),
                optimization_steps=optimization_steps,
                learning_rate=learning_rate,
                coord_weight=coord_weight,
                lattice_weight=lattice_weight,
                pairdist_weight=pairdist_weight,
                pairdist_bins=pairdist_bins,
                pairdist_max_distance=pairdist_max_distance,
                pairdist_bandwidth=pairdist_bandwidth,
                steric_weight=steric_weight,
                steric_min_distance=steric_min_distance,
                volume_weight=volume_weight,
                volume_ratio_min=volume_ratio_min,
                volume_ratio_max=volume_ratio_max,
                k6_weight=k6_weight,
                freeze_lattice_free_vars=freeze_lattice_free_vars,
            )
            if not np.isfinite(objective) and float(pairdist_weight) > 0.0:
                free_vars, lattice_free, objective = _optimize_template_fit(
                    template=template,
                    constraint=constraint,
                    target_frac=candidate_target_frac,
                    target_atomic_numbers=candidate_target_atomic_numbers,
                    target_k=cell_to_k(candidate_target_cell, eps=1e-8),
                    optimization_steps=optimization_steps,
                    learning_rate=learning_rate,
                    coord_weight=coord_weight,
                    lattice_weight=lattice_weight,
                    pairdist_weight=0.0,
                    pairdist_bins=pairdist_bins,
                    pairdist_max_distance=pairdist_max_distance,
                    pairdist_bandwidth=pairdist_bandwidth,
                    steric_weight=steric_weight,
                    steric_min_distance=steric_min_distance,
                    volume_weight=volume_weight,
                    volume_ratio_min=volume_ratio_min,
                    volume_ratio_max=volume_ratio_max,
                    k6_weight=k6_weight,
                    freeze_lattice_free_vars=freeze_lattice_free_vars,
                )
            if not np.isfinite(objective):
                reject_reasons["nonfinite_objective"] += 1
                continue
            expansion = expand_wyckoff_template_torch(template=template, free_vars=free_vars)
            if expansion.frac_coords.shape[0] != candidate_target_frac.shape[0]:
                reject_reasons["atom_count_mismatch"] += 1
                continue
            if not _torch_atomic_multiset_matches(expansion.atomic_numbers, candidate_target_atomic_numbers):
                reject_reasons["atomic_multiset_mismatch"] += 1
                continue
            try:
                fixed_target_assignment = _species_assignment_indices(
                    source_frac=expansion.frac_coords,
                    source_atomic_numbers=expansion.atomic_numbers,
                    target_frac=candidate_target_frac,
                    target_atomic_numbers=candidate_target_atomic_numbers,
                )
            except Exception:
                reject_reasons["assignment_failed"] += 1
                continue
            prior_score = template_prior_score(
                prior=template_prior,
                key=composition_key,
                signature=flatten_site_signature(template),
            )
            prior_bonus = float(template_prior_weight) * float(np.log1p(max(prior_score, 0)))
            target_cell = candidate_target_cell.detach().clone()
            target_k = cell_to_k(target_cell, eps=1e-8).detach().clone()
            reference_volume = float(_cell_volume(target_cell).detach().item())
            reference_k6 = float(target_k[..., -1].detach().item()) if target_k.numel() > 0 else None
            projected_reference_k = cell_to_k(cell_matrix, eps=1e-8).detach().clone()
            projected_reference_volume = float(_cell_volume(cell_matrix).detach().item())
            projected_reference_k6 = (
                float(projected_reference_k[..., -1].detach().item())
                if projected_reference_k.numel() > 0
                else None
            )
            target_pairdist_hist = None
            pairdist_bin_centers = None
            if float(pairdist_weight) > 0.0:
                target_pairdist_hist, pairdist_bin_centers = _build_target_pairdist_cache(
                    target_frac=candidate_target_frac,
                    target_cell=target_cell,
                    bins=pairdist_bins,
                    max_distance=pairdist_max_distance,
                    bandwidth=pairdist_bandwidth,
                )
            is_centering_expanded_target = (
                use_centering_expansion
                and target_name in {"expanded", "raw_requested_expanded"}
            )
            state_candidate = PCSTemplateState(
                template=template,
                constraint=constraint,
                bridge=bridge,
                free_vars=free_vars,
                lattice_free_vars=lattice_free,
                objective=objective,
                ranking_objective=float(objective - prior_bonus),
                template_rank=int(template_rank),
                candidate_count=int(total_templates_found),
                target_centering_symbol=(
                    target_centering
                    if is_centering_expanded_target else None
                ),
                target_centering_translations=(
                    target_centering_translations
                    if is_centering_expanded_target else None
                ),
                target_frac=candidate_target_frac.detach().clone(),
                target_atomic_numbers=candidate_target_atomic_numbers.detach().clone(),
                target_cell=target_cell,
                target_k=target_k,
                fixed_target_assignment=fixed_target_assignment.detach().clone(),
                target_pairdist_hist=target_pairdist_hist,
                pairdist_bin_centers=pairdist_bin_centers,
                anchor_frac=candidate_target_frac.detach().clone(),
                anchor_atomic_numbers=candidate_target_atomic_numbers.detach().clone(),
                anchor_cell=target_cell.detach().clone(),
                anchor_k=target_k.detach().clone(),
                anchor_assignment=fixed_target_assignment.detach().clone(),
                anchor_free_vars=free_vars.detach().clone(),
                anchor_lattice_free_vars=lattice_free.detach().clone(),
                anchor_pairdist_hist=target_pairdist_hist.detach().clone() if target_pairdist_hist is not None else None,
                anchor_pairdist_bin_centers=(
                    pairdist_bin_centers.detach().clone() if pairdist_bin_centers is not None else None
                ),
                anchor_representation_name=str(target_name),
                reference_volume=reference_volume,
                reference_k6=reference_k6,
                projected_reference_volume=(
                    projected_reference_volume if is_centering_expanded_target else reference_volume
                ),
                projected_reference_k6=(
                    projected_reference_k6 if is_centering_expanded_target else reference_k6
                ),
                prior_score=int(prior_score),
                prior_bonus=prior_bonus,
                freeze_lattice=bool(freeze_lattice_free_vars),
                target_species_orbit_signature=orbit_reference_signature,
                template_species_orbit_signature=template_species_orbit_signature,
                species_orbit_mismatch=int(species_orbit_mismatch),
                orbit_reference_is_oracle=bool(str(orbit_reference_source).startswith("oracle")),
                orbit_reference_rank_first=bool(orbit_reference_rank_first),
                target_representation_name=str(target_name),
            )
            candidate_key = _pcs_state_rank_key(state_candidate)
            best_key = None if best_state is None else _pcs_state_rank_key(best_state)
            if best_state is None or candidate_key < best_key:
                best_state = state_candidate
        if best_state is not None:
            candidate_states.append(best_state)

    if not candidate_states:
        top_candidate_lines: list[str] = []
        for template, _template_rank, composition_key in ranked_templates[: min(12, len(ranked_templates))]:
            prior_score = template_prior_score(
                prior=template_prior,
                key=composition_key,
                signature=flatten_site_signature(template),
            )
            top_candidate_lines.append(
                _format_template_summary(
                    template,
                    composition_key=composition_key,
                    prior_score=prior_score,
                )
            )
        variant_lines = [
            f"{idx + 1}. key={composition_key} counts={_format_species_counts(variant_atomic_numbers)}"
            for idx, (variant_atomic_numbers, composition_key) in enumerate(composition_variants)
        ]
        target_lines = [
            f"{name}: counts={_format_species_counts(target_atomic_numbers)} num_atoms={int(target_atomic_numbers.shape[0])} "
            f"volume={float(_cell_volume(target_cell).detach().item()):.4f}"
            for name, _target_frac, target_atomic_numbers, target_cell in target_representations
        ]
        raise RuntimeError(
            f"PCS template selection found templates for requested space group {requested_sg}, "
            "but none preserved the standardized composition and atom count.\n"
            f"raw_counts={_format_species_counts(atomic_numbers)}\n"
            f"standardized_counts={_format_species_counts(standardized_atomic_numbers)}\n"
            f"expanded_target_counts={_format_species_counts(target_atomic_numbers_for_fit)}\n"
            f"reject_reason_counts={reject_reasons}\n"
            f"composition_variants=\n  " + "\n  ".join(variant_lines) + "\n"
            f"target_representations=\n  " + "\n  ".join(target_lines) + "\n"
            f"top_candidate_templates=\n  " + ("\n  ".join(top_candidate_lines) if top_candidate_lines else "<none>")
        )

    candidate_states.sort(key=_pcs_state_rank_key)
    if debug_template_candidates:
        for candidate_idx, state in enumerate(candidate_states[: min(12, len(candidate_states))], start=1):
            signature_labels = (
                "na"
                if state.template_species_orbit_signature is None
                else [f"{Element.from_Z(int(z)).symbol}@{label}" for z, label in state.template_species_orbit_signature]
            )
            theta = torch.cat(
                [
                    state.free_vars.reshape(-1),
                    state.lattice_free_vars.reshape(-1),
                ],
                dim=0,
            )
            energy_result = _template_energy_from_state(
                template=state.template,
                constraint=state.constraint,
                theta=theta,
                free_dim=state.template.total_free_dims,
                target_frac=state.target_frac,
                target_atomic_numbers=state.target_atomic_numbers,
                target_cell=state.target_cell,
                target_k=state.target_k,
                coord_weight=coord_weight,
                lattice_weight=lattice_weight,
                pairdist_weight=pairdist_weight,
                pairdist_bins=pairdist_bins,
                pairdist_max_distance=pairdist_max_distance,
                pairdist_bandwidth=pairdist_bandwidth,
                steric_weight=steric_weight,
                steric_min_distance=steric_min_distance,
                volume_weight=volume_weight,
                volume_ratio_min=volume_ratio_min,
                volume_ratio_max=volume_ratio_max,
                k6_weight=k6_weight,
                target_assignment=state.fixed_target_assignment,
                reference_volume=state.reference_volume,
                reference_k6=state.reference_k6,
                prior_bonus=0.0,
                eta=None,
                target_pairdist_hist=state.target_pairdist_hist,
                pairdist_bin_centers=state.pairdist_bin_centers,
            )
            print(
                f"algorithm6_template_candidate {debug_label or 'graph=?'} "
                f"idx={candidate_idx} template_rank={int(state.template_rank)} "
                f"orbit_mismatch={int(state.species_orbit_mismatch)} "
                f"target_repr={state.target_representation_name or 'na'} "
                f"objective={float(state.objective):.6f} "
                f"objective_minus_bonus={float(state.objective - state.prior_bonus):.6f} "
                f"prior_score={int(state.prior_score)} "
                f"signature={signature_labels}",
                flush=True,
            )
            print(
                f"algorithm6_template_candidate_terms {debug_label or 'graph=?'} "
                f"idx={candidate_idx} coord_loss={float(energy_result.coord_loss.item()):.6f} "
                f"lattice_loss={float(energy_result.lattice_loss.item()):.6f} "
                f"pairdist_loss={float(energy_result.pairdist_loss.item()):.6f} "
                f"steric_loss={float(energy_result.steric_loss.item()):.6f} "
                f"volume_loss={float(energy_result.volume_loss.item()):.6f} "
                f"k6_loss={float(energy_result.k6_loss.item()):.6f} "
                f"target_volume={float(_cell_volume(state.target_cell).detach().item()):.6f} "
                f"target_k6={float(state.target_k[..., -1].detach().item()) if state.target_k is not None else float('nan'):.6f} "
                f"prox_energy={float(energy_result.prox_energy.item()):.6f} "
                f"likelihood_energy={float(energy_result.likelihood_energy.item()):.6f} "
                f"energy={float(energy_result.energy.item()):.6f}",
                flush=True,
            )
    return candidate_states[: int(top_k)]


def sample_pcs_step_mala(
    *,
    state: PCSTemplateState,
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell_matrix: torch.Tensor,
    eta: float,
    mala_steps: int = 8,
    mala_step_size: float = 5e-2,
    coord_weight: float = 1.0,
    lattice_weight: float = 0.25,
    pairdist_weight: float = 0.0,
    pairdist_bins: int = 32,
    pairdist_max_distance: float = 8.0,
    pairdist_bandwidth: float = 0.25,
    steric_weight: float = 0.0,
    steric_min_distance: float = 0.8,
    volume_weight: float = 0.0,
    volume_ratio_min: float = 0.0,
    volume_ratio_max: float = 0.0,
    k6_weight: float = 0.0,
    freeze_lattice_free_vars: bool = False,
) -> PCSTemplateState:
    _require_pymatgen()
    if mala_steps < 1:
        raise ValueError("mala_steps must be >= 1.")
    if mala_step_size <= 0.0:
        raise ValueError("mala_step_size must be positive.")

    device = frac_coords.device
    dtype = frac_coords.dtype

    state = refresh_pcs_state_anchor(
        state=state,
        frac_coords=frac_coords,
        atomic_numbers=atomic_numbers,
        cell_matrix=cell_matrix,
        pairdist_weight=pairdist_weight,
        pairdist_bins=pairdist_bins,
        pairdist_max_distance=pairdist_max_distance,
        pairdist_bandwidth=pairdist_bandwidth,
        coord_weight=coord_weight,
        lattice_weight=lattice_weight,
    )
    if state.anchor_frac is None or state.anchor_atomic_numbers is None or state.anchor_k is None or state.anchor_cell is None:
        raise RuntimeError("PCS chain state is missing its refreshed proximal anchor.")
    if state.anchor_assignment is None:
        raise RuntimeError("PCS chain state is missing its refreshed target assignment.")

    target_frac = state.anchor_frac.to(device=device, dtype=dtype)
    target_atomic_numbers = state.anchor_atomic_numbers.to(device=device, dtype=torch.long)
    target_cell = state.anchor_cell.to(device=device, dtype=dtype)
    target_k = state.anchor_k.to(device=device, dtype=dtype)
    fixed_target_assignment = state.anchor_assignment.to(device=device, dtype=torch.long)

    free_dim = state.template.total_free_dims
    theta = torch.cat(
        [
            state.free_vars.to(device=device, dtype=dtype).reshape(-1),
            state.lattice_free_vars.to(device=device, dtype=dtype).reshape(-1),
        ],
        dim=0,
    )
    theta_anchor = theta.detach().clone()
    step_size = torch.as_tensor(float(mala_step_size), device=device, dtype=dtype)
    grad_clip_norm = theta.new_tensor(10.0)
    freeze_lattice = bool(freeze_lattice_free_vars or state.freeze_lattice)

    def energy_and_grad(theta_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta_var = theta_value.detach().clone().requires_grad_(True)
        energy_result = _template_energy_from_state(
            template=state.template,
            constraint=state.constraint,
            theta=theta_var,
            free_dim=free_dim,
            target_frac=target_frac,
            target_atomic_numbers=target_atomic_numbers,
            target_cell=target_cell,
            target_k=target_k,
            coord_weight=coord_weight,
            lattice_weight=lattice_weight,
            pairdist_weight=pairdist_weight,
            pairdist_bins=pairdist_bins,
            pairdist_max_distance=pairdist_max_distance,
            pairdist_bandwidth=pairdist_bandwidth,
            steric_weight=steric_weight,
            steric_min_distance=steric_min_distance,
            volume_weight=volume_weight,
            volume_ratio_min=volume_ratio_min,
            volume_ratio_max=volume_ratio_max,
            k6_weight=k6_weight,
            target_assignment=fixed_target_assignment,
            reference_volume=state.reference_volume,
            reference_k6=state.reference_k6,
            prior_bonus=state.prior_bonus,
            eta=eta,
            target_pairdist_hist=state.anchor_pairdist_hist,
            pairdist_bin_centers=state.anchor_pairdist_bin_centers,
            anchor_free_vars=state.anchor_free_vars,
            anchor_lattice_free_vars=state.anchor_lattice_free_vars,
        )
        energy = energy_result.energy
        if not torch.isfinite(energy):
            raise RuntimeError("PCS MALA energy became non-finite.")
        grad, = torch.autograd.grad(energy, theta_var)
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        if freeze_lattice and grad.shape[0] > free_dim:
            grad[free_dim:] = 0.0
        grad_norm = torch.linalg.vector_norm(grad)
        if torch.isfinite(grad_norm) and grad_norm > grad_clip_norm:
            grad = grad * (grad_clip_norm / grad_norm.clamp_min(1e-12))
        return energy.detach(), grad.detach()

    gamma = theta.new_tensor(float(max(eta, 1.0e-4)))
    gamma_sq = gamma.square().clamp_min(1.0e-8)
    coef_anchor = 1.0 - torch.exp(-step_size / gamma_sq)
    coef_grad = gamma_sq * coef_anchor
    coef_noise = gamma * torch.sqrt((1.0 - torch.exp(-2.0 * step_size / gamma_sq)).clamp_min(1.0e-8))

    def proposal_mean(src: torch.Tensor, grad_src: torch.Tensor) -> torch.Tensor:
        return (1.0 - coef_anchor) * src + coef_anchor * theta_anchor - coef_grad * grad_src

    def log_q(src: torch.Tensor, grad_src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        mean = proposal_mean(src, grad_src)
        residual = dst - mean
        return -residual.square().sum() / (2.0 * coef_noise.square().clamp_min(1.0e-8))

    current_energy, current_grad = energy_and_grad(theta)
    accept_count = 0
    for _step_idx in range(int(mala_steps)):
        noise = torch.randn_like(theta)
        if freeze_lattice and noise.shape[0] > free_dim:
            noise[free_dim:] = 0.0
        proposal = proposal_mean(theta, current_grad) + coef_noise * noise
        if freeze_lattice and proposal.shape[0] > free_dim:
            proposal[free_dim:] = theta_anchor[free_dim:]
        try:
            proposal_energy, proposal_grad = energy_and_grad(proposal)
        except RuntimeError as exc:
            if "non-finite" in str(exc).lower():
                continue
            raise

        log_alpha = (
            -proposal_energy
            + current_energy
            + log_q(proposal, proposal_grad, theta)
            - log_q(theta, current_grad, proposal)
        )
        accept = bool(torch.log(torch.rand((), device=device, dtype=dtype)).item() < float(log_alpha.item()))
        if accept:
            theta = proposal
            current_energy = proposal_energy
            current_grad = proposal_grad
            accept_count += 1

    final_free_vars = theta[:free_dim]
    final_lattice_free_vars = theta[free_dim:]
    final_expansion = expand_wyckoff_template_torch(
        template=state.template,
        free_vars=final_free_vars,
    )
    final_energy_result = _template_energy_from_state(
        template=state.template,
        constraint=state.constraint,
        theta=theta.detach(),
        free_dim=free_dim,
        target_frac=target_frac,
        target_atomic_numbers=target_atomic_numbers,
        target_cell=target_cell,
        target_k=target_k,
        coord_weight=coord_weight,
        lattice_weight=lattice_weight,
        pairdist_weight=pairdist_weight,
        pairdist_bins=pairdist_bins,
        pairdist_max_distance=pairdist_max_distance,
        pairdist_bandwidth=pairdist_bandwidth,
        steric_weight=steric_weight,
        steric_min_distance=steric_min_distance,
        volume_weight=volume_weight,
        volume_ratio_min=volume_ratio_min,
        volume_ratio_max=volume_ratio_max,
        k6_weight=k6_weight,
        target_assignment=fixed_target_assignment,
        reference_volume=state.reference_volume,
        reference_k6=state.reference_k6,
        prior_bonus=state.prior_bonus,
        eta=eta,
        target_pairdist_hist=state.anchor_pairdist_hist,
        pairdist_bin_centers=state.anchor_pairdist_bin_centers,
        anchor_free_vars=state.anchor_free_vars,
        anchor_lattice_free_vars=state.anchor_lattice_free_vars,
    )
    vanilla_coord_distance = float(
        _fixed_assignment_torus_loss(
            source_frac=final_expansion.frac_coords,
            target_frac=target_frac,
            target_assignment=fixed_target_assignment,
        ).detach().item()
    )
    vanilla_lattice_k_distance = float((final_energy_result.k_projected - target_k).square().mean().detach().item())
    asymmetric_unit_distortion = float(final_free_vars.square().mean().detach().item()) if final_free_vars.numel() > 0 else 0.0

    return replace(
        state,
        free_vars=final_free_vars.detach().clone(),
        lattice_free_vars=final_lattice_free_vars.detach().clone(),
        objective=float(final_energy_result.energy.item()),
        ranking_objective=float(final_energy_result.energy.item()),
        vanilla_coord_distance=vanilla_coord_distance,
        vanilla_lattice_k_distance=vanilla_lattice_k_distance,
        asymmetric_unit_distortion=asymmetric_unit_distortion,
        mala_acceptance_rate=float(accept_count) / float(max(int(mala_steps), 1)),
        mala_accept_count=int(accept_count),
        mala_attempted_steps=int(mala_steps),
        mala_coord_loss=float(final_energy_result.coord_loss.detach().item()),
        mala_lattice_loss=float(final_energy_result.lattice_loss.detach().item()),
        mala_pairdist_loss=float(final_energy_result.pairdist_loss.detach().item()),
        mala_steric_loss=float(final_energy_result.steric_loss.detach().item()),
        mala_volume_loss=float(final_energy_result.volume_loss.detach().item()),
        mala_k6_loss=float(final_energy_result.k6_loss.detach().item()),
        mala_prox_energy=float(final_energy_result.prox_energy.detach().item()),
        mala_likelihood_energy=float(final_energy_result.likelihood_energy.detach().item()),
        mala_total_energy=float(final_energy_result.energy.detach().item()),
    )


def materialize_pcs_state(
    *,
    state: PCSTemplateState,
    vanilla_reference_structure,
) -> PCSProjectionResult:
    del vanilla_reference_structure
    expansion = expand_wyckoff_template_torch(
        template=state.template,
        free_vars=state.free_vars,
    )
    cell_projected = k_to_cell_matrix(free_vars_to_k(state.lattice_free_vars, state.constraint))
    projected_standardized = _build_structure_from_standardized_projection(
        frac_coords=expansion.frac_coords,
        atomic_numbers=expansion.atomic_numbers,
        cell_matrix=cell_projected,
    )
    standardized_space_group = _detect_space_group_number(
        structure=projected_standardized,
        symprec=state.bridge.symprec,
        angle_tolerance=state.bridge.angle_tolerance,
    )
    try:
        _analyzer, reduced_standardized = standardize_structure(
            projected_standardized,
            standardization="primitive",
            symprec=state.bridge.symprec,
            angle_tolerance=state.bridge.angle_tolerance,
        )
    except Exception:
        reduced_standardized = projected_standardized
    primitive_space_group = _detect_space_group_number(
        structure=reduced_standardized,
        symprec=state.bridge.symprec,
        angle_tolerance=state.bridge.angle_tolerance,
    )
    collapsed_standardized = _collapse_centering_equivalent_structure(
        structure=projected_standardized,
        translations=state.target_centering_translations,
        expected_atomic_numbers=state.bridge.vanilla_atomic_numbers,
    )
    primitive_centered = _structure_to_primitive_centering_basis(
        structure=projected_standardized,
        centering_symbol=state.target_centering_symbol,
        expected_atomic_numbers=state.bridge.vanilla_atomic_numbers,
    )
    projected_vanilla = map_standardized_structure_to_vanilla_frame(
        standardized_structure=collapsed_standardized,
        vanilla_reference_structure=state.bridge.vanilla_structure,
        symprec=state.bridge.symprec,
        angle_tolerance=state.bridge.angle_tolerance,
    )
    mapped_space_group = _detect_space_group_number(
        structure=projected_vanilla,
        symprec=state.bridge.symprec,
        angle_tolerance=state.bridge.angle_tolerance,
    )
    if (
        state.target_centering_symbol not in {None, "", "P"}
        and primitive_centered is not None
        and mapped_space_group != int(state.constraint.space_group)
    ):
        projected_vanilla = primitive_centered
    return PCSProjectionResult(
        projected_structure_standardized=projected_standardized,
        projected_structure_primitive=reduced_standardized,
        projected_structure_vanilla=projected_vanilla,
        template=state.template,
        free_vars=state.free_vars.detach().clone(),
        lattice_free_vars=state.lattice_free_vars.detach().clone(),
        objective=float(state.objective),
        template_rank=int(state.template_rank),
        candidate_count=int(state.candidate_count),
        standardized_space_group=standardized_space_group,
        primitive_space_group=primitive_space_group,
    )


def validate_requested_space_group(
    *,
    structure,
    requested_space_group: int,
    expected_atomic_numbers: torch.Tensor,
    symprec: float = 1e-2,
    angle_tolerance: float = 5.0,
) -> PCSValidationResult:
    _require_pymatgen()
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PCS validation requires pymatgen symmetry tools.") from exc

    detected_sg: int | None
    try:
        detected_sg = int(
            SpacegroupAnalyzer(
                structure,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            ).get_space_group_number()
        )
    except Exception:
        detected_sg = None

    predicted_atomic_numbers = torch.as_tensor(np.asarray(structure.atomic_numbers, dtype=int), dtype=torch.long)
    composition_match = _torch_atomic_multiset_matches(predicted_atomic_numbers, expected_atomic_numbers)
    requested_match = detected_sg == int(requested_space_group)
    return PCSValidationResult(
        composition_match=composition_match,
        requested_space_group=int(requested_space_group),
        detected_space_group=detected_sg,
        requested_space_group_match=requested_match,
    )


def project_vanilla_sample_to_spacegroup_manifold(
    *,
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell_matrix: torch.Tensor,
    space_group_number: int,
    lattice_transform,
    standardization: str = "conventional",
    symprec: float = 1e-2,
    angle_tolerance: float = 5.0,
    max_templates: int = 256,
    template_eval_limit: int = 32,
    optimization_steps: int = 150,
    learning_rate: float = 5e-2,
    coord_weight: float = 1.0,
    lattice_weight: float = 0.25,
    quick_templates: bool = False,
) -> PCSProjectionResult:
    del lattice_transform
    state = select_requested_template_state(
        frac_coords=frac_coords,
        atomic_numbers=atomic_numbers,
        cell_matrix=cell_matrix,
        space_group_number=space_group_number,
        standardization=standardization,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        max_templates=max_templates,
        template_eval_limit=template_eval_limit,
        optimization_steps=optimization_steps,
        learning_rate=learning_rate,
        coord_weight=coord_weight,
        lattice_weight=lattice_weight,
        quick_templates=quick_templates,
    )
    return materialize_pcs_state(
        state=state,
        vanilla_reference_structure=state.bridge.vanilla_structure,
    )


def vanilla_structure_to_model_tensors(
    *,
    structure,
    lattice_transform,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frac_coords = torch.as_tensor(np.asarray(structure.frac_coords, dtype=float), device=device, dtype=dtype)
    atomic_numbers = torch.as_tensor(np.asarray(structure.atomic_numbers, dtype=int), device=device, dtype=torch.long)
    cell_matrix = torch.tensor(
        np.asarray(structure.lattice.matrix, dtype=float).copy(),
        device=device,
        dtype=dtype,
    )
    l_features = _encode_cell_to_lattice_features(
        cell_matrix=cell_matrix,
        num_atoms=int(len(structure)),
        lattice_transform=lattice_transform,
    ).to(device=device, dtype=dtype)
    return frac_coords, l_features, atomic_numbers
