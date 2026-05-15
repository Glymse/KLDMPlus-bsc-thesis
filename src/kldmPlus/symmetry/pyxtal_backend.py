from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from pymatgen.core import Element
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:  # pragma: no cover
    Element = None
    SpacegroupAnalyzer = None

try:
    from pyxtal import pyxtal
except ImportError:  # pragma: no cover
    pyxtal = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


@dataclass
class PyXtalWyckoffResult:
    space_group: int
    lattice_parameters: np.ndarray
    anchor_frac_coords: np.ndarray
    anchor_atomic_numbers: np.ndarray
    site_labels: tuple[str, ...]
    site_multiplicities: np.ndarray
    site_dofs: np.ndarray
    site_free_coordinate_masks: np.ndarray
    expanded_frac_coords: np.ndarray
    expanded_atomic_numbers: np.ndarray
    anchor_index: np.ndarray
    affine_ops: np.ndarray
    anchor_count: int
    num_atoms: int


def _require_pyxtal_dependencies() -> None:
    missing: list[str] = []
    if Element is None or SpacegroupAnalyzer is None:
        missing.append("pymatgen")
    if pyxtal is None:
        missing.append("pyxtal")
    if missing:
        raise ImportError(
            "PyXtal Wyckoff utilities require: " + ", ".join(missing) + ".",
        )


def _atomic_number(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        return int(Element(value).Z)
    if hasattr(value, "Z"):
        return int(value.Z)
    return int(Element(str(value)).Z)


def _free_coordinate_mask_from_wp(wp: Any) -> np.ndarray:
    if hasattr(wp, "get_frozen_axis"):
        frozen_axes = list(wp.get_frozen_axis() or [])
        free_mask = np.ones(3, dtype=bool)
        free_mask[frozen_axes] = False
        return free_mask

    anchor_rotation = np.asarray(wp.ops[0].rotation_matrix, dtype=float)
    column_norms = np.linalg.norm(anchor_rotation, axis=0)
    return column_norms > 1e-8


def build_pyxtal_wyckoff_result(
    structure,
    *,
    symprec: float = 1e-2,
    pyxtal_tol: float = 1e-2,
) -> PyXtalWyckoffResult:
    """Extract a PyXtal Wyckoff decomposition from a pymatgen Structure.

    The returned anchors/orbit operators mirror the representation DiffCSP uses:
    a smaller set of anchor sites plus affine ops that expand them to all atoms.
    """
    _require_pyxtal_dependencies()

    refined = SpacegroupAnalyzer(structure, symprec=symprec).get_refined_structure()
    crystal = pyxtal()
    try:
        crystal.from_seed(refined, tol=pyxtal_tol)
    except Exception:
        crystal.from_seed(refined, tol=max(pyxtal_tol * 0.01, 1e-4))

    anchor_frac_coords: list[np.ndarray] = []
    anchor_atomic_numbers: list[int] = []
    site_labels: list[str] = []
    site_multiplicities: list[int] = []
    site_dofs: list[int] = []
    site_free_coordinate_masks: list[np.ndarray] = []
    expanded_frac_coords: list[np.ndarray] = []
    expanded_atomic_numbers: list[int] = []
    anchor_index: list[int] = []
    affine_ops: list[np.ndarray] = []

    for site_idx, site in enumerate(crystal.atom_sites):
        specie_z = _atomic_number(site.specie)
        anchor_frac = np.asarray(site.position, dtype=float) % 1.0
        wp = site.wp
        free_mask = _free_coordinate_mask_from_wp(wp)
        anchor_frac_coords.append(anchor_frac)
        anchor_atomic_numbers.append(specie_z)
        site_labels.append(str(wp.get_label()))
        site_multiplicities.append(int(wp.multiplicity))
        site_dofs.append(int(wp.get_dof()))
        site_free_coordinate_masks.append(free_mask)

        for op in wp:
            affine = np.asarray(op.affine_matrix, dtype=float)
            expanded = np.asarray(op.operate(site.position), dtype=float) % 1.0
            expanded_frac_coords.append(expanded)
            expanded_atomic_numbers.append(specie_z)
            anchor_index.append(site_idx)
            affine_ops.append(affine)

    return PyXtalWyckoffResult(
        space_group=int(crystal.group.number),
        lattice_parameters=np.asarray(crystal.lattice.get_para(degree=True), dtype=float),
        anchor_frac_coords=np.asarray(anchor_frac_coords, dtype=float),
        anchor_atomic_numbers=np.asarray(anchor_atomic_numbers, dtype=int),
        site_labels=tuple(site_labels),
        site_multiplicities=np.asarray(site_multiplicities, dtype=int),
        site_dofs=np.asarray(site_dofs, dtype=int),
        site_free_coordinate_masks=np.asarray(site_free_coordinate_masks, dtype=bool),
        expanded_frac_coords=np.asarray(expanded_frac_coords, dtype=float),
        expanded_atomic_numbers=np.asarray(expanded_atomic_numbers, dtype=int),
        anchor_index=np.asarray(anchor_index, dtype=int),
        affine_ops=np.asarray(affine_ops, dtype=float),
        anchor_count=len(anchor_frac_coords),
        num_atoms=len(expanded_frac_coords),
    )


def _torus_pairwise_distance_sq(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = source[:, None, :] - target[None, :, :]
    delta = delta - np.round(delta)
    return np.sum(delta * delta, axis=-1)


def _match_cost_matrix(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if linear_sum_assignment is not None:
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        return np.asarray(row_idx, dtype=int), np.asarray(col_idx, dtype=int)

    remaining_rows = list(range(cost_matrix.shape[0]))
    remaining_cols = list(range(cost_matrix.shape[1]))
    chosen_rows: list[int] = []
    chosen_cols: list[int] = []

    while remaining_rows:
        submatrix = cost_matrix[np.ix_(remaining_rows, remaining_cols)]
        flat_index = int(np.argmin(submatrix))
        n_cols = submatrix.shape[1]
        row_pos = flat_index // n_cols
        col_pos = flat_index % n_cols
        chosen_rows.append(remaining_rows.pop(row_pos))
        chosen_cols.append(remaining_cols.pop(col_pos))

    order = np.argsort(np.asarray(chosen_rows))
    return np.asarray(chosen_rows, dtype=int)[order], np.asarray(chosen_cols, dtype=int)[order]


def species_aware_torus_rmse(
    *,
    source_frac_coords: np.ndarray,
    source_atomic_numbers: np.ndarray,
    target_frac_coords: np.ndarray,
    target_atomic_numbers: np.ndarray,
) -> tuple[float | None, str | None]:
    """Match by species and return the torus RMSE plus an explicit status."""
    if len(source_frac_coords) != len(target_frac_coords):
        return None, "num_atoms_mismatch"

    total_sq = 0.0
    total_count = 0
    species_values = sorted(set(int(v) for v in source_atomic_numbers.tolist()))
    if species_values != sorted(set(int(v) for v in target_atomic_numbers.tolist())):
        return None, "species_mismatch"

    for atomic_number in species_values:
        source_mask = source_atomic_numbers == atomic_number
        target_mask = target_atomic_numbers == atomic_number
        if int(np.sum(source_mask)) != int(np.sum(target_mask)):
            return None, "species_count_mismatch"

        src = source_frac_coords[source_mask]
        tgt = target_frac_coords[target_mask]
        if len(src) == 0:
            continue

        cost_matrix = _torus_pairwise_distance_sq(src, tgt)
        row_idx, col_idx = _match_cost_matrix(cost_matrix)
        deltas = src[row_idx] - tgt[col_idx]
        deltas = deltas - np.round(deltas)
        total_sq += float(np.sum(deltas * deltas))
        total_count += int(deltas.size)

    if total_count == 0:
        return None, "empty_matching"
    return float(np.sqrt(total_sq / total_count)), "ok"
