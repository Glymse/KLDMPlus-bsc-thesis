from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None

try:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Element, Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:  # pragma: no cover
    Element = Lattice = SpacegroupAnalyzer = Structure = StructureMatcher = None


@dataclass
class SymmetryFrameBridge:
    vanilla_structure: Any
    standardized_structure: Any
    standardized_to_vanilla_structure: Any
    detected_space_group: int
    standardized_space_group: int
    standardization: str
    symprec: float
    angle_tolerance: float
    standardized_to_vanilla_linear: np.ndarray | None = None
    standardized_to_vanilla_tau: np.ndarray | None = None
    standardized_to_vanilla_assignment: np.ndarray | None = None
    standardized_to_vanilla_rmse: float | None = None
    standardized_to_vanilla_method: str | None = None

    @property
    def vanilla_atomic_numbers(self) -> np.ndarray:
        return np.asarray(self.vanilla_structure.atomic_numbers, dtype=int)

    @property
    def standardized_atomic_numbers(self) -> np.ndarray:
        return np.asarray(self.standardized_structure.atomic_numbers, dtype=int)


def _require_pymatgen() -> None:
    if None in (Element, Lattice, SpacegroupAnalyzer, Structure, StructureMatcher):
        raise ImportError("Symmetry frame bridge requires pymatgen.")


def _wrap01_numpy(value: np.ndarray) -> np.ndarray:
    return np.remainder(np.asarray(value, dtype=float), 1.0)


def _torus_delta_numpy(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.asarray(source, dtype=float) - np.asarray(target, dtype=float)
    return delta - np.round(delta)


def _torus_pairwise_distance_sq(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = _torus_delta_numpy(source[:, None, :], target[None, :, :])
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


def _species_assignment_with_tau(
    *,
    source_frac: np.ndarray,
    source_atomic_numbers: np.ndarray,
    target_frac: np.ndarray,
    target_atomic_numbers: np.ndarray,
    tau: np.ndarray,
) -> tuple[np.ndarray, float]:
    if sorted(source_atomic_numbers.tolist()) != sorted(target_atomic_numbers.tolist()):
        raise ValueError("Species multiset mismatch in semantic frame transport.")

    assignment = np.empty(source_frac.shape[0], dtype=int)
    moved_source = _wrap01_numpy(source_frac + np.asarray(tau, dtype=float).reshape(1, 3))
    total_sq = 0.0
    total_dims = 0

    for atomic_number in sorted(set(int(v) for v in source_atomic_numbers.tolist())):
        src_idx = np.where(source_atomic_numbers == atomic_number)[0]
        dst_idx = np.where(target_atomic_numbers == atomic_number)[0]
        if src_idx.size != dst_idx.size:
            raise ValueError(f"Species count mismatch for Z={atomic_number}.")
        src = moved_source[src_idx]
        dst = target_frac[dst_idx]
        cost = _torus_pairwise_distance_sq(src, dst)
        rows, cols = _match_cost_matrix(cost)
        assignment[src_idx[rows]] = dst_idx[cols]
        delta = _torus_delta_numpy(src[rows], dst[cols])
        total_sq += float(np.sum(delta * delta))
        total_dims += int(delta.size)

    rmse = 0.0 if total_dims == 0 else float(np.sqrt(total_sq / total_dims))
    return assignment, rmse


def _estimate_semantic_transport(
    *,
    standardized_structure,
    vanilla_structure,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    standardized_frac = _wrap01_numpy(np.asarray(standardized_structure.frac_coords, dtype=float))
    vanilla_frac = _wrap01_numpy(np.asarray(vanilla_structure.frac_coords, dtype=float))
    standardized_atomic_numbers = np.asarray(standardized_structure.atomic_numbers, dtype=int).reshape(-1)
    vanilla_atomic_numbers = np.asarray(vanilla_structure.atomic_numbers, dtype=int).reshape(-1)
    if standardized_frac.shape != vanilla_frac.shape:
        raise ValueError(
            "Standardized and vanilla structures must contain the same number of atoms "
            f"for semantic transport, got {standardized_frac.shape} vs {vanilla_frac.shape}."
        )

    lattice_standardized = np.asarray(standardized_structure.lattice.matrix, dtype=float)
    lattice_vanilla = np.asarray(vanilla_structure.lattice.matrix, dtype=float)
    linear = lattice_standardized @ np.linalg.inv(lattice_vanilla)
    source_in_vanilla_fractional = _wrap01_numpy(standardized_frac @ linear)

    species_values = sorted(set(int(v) for v in standardized_atomic_numbers.tolist()))
    tau_candidates: list[np.ndarray] = []
    for atomic_number in species_values:
        src_idx = np.where(standardized_atomic_numbers == atomic_number)[0]
        dst_idx = np.where(vanilla_atomic_numbers == atomic_number)[0]
        if src_idx.size == 0:
            continue
        for src_atom in src_idx:
            diffs = _torus_delta_numpy(vanilla_frac[dst_idx], source_in_vanilla_fractional[src_atom])
            for diff in diffs:
                tau_candidates.append(np.asarray(diff, dtype=float).reshape(3))

    if not tau_candidates:
        tau_candidates = [np.zeros(3, dtype=float)]

    best_tau = np.zeros(3, dtype=float)
    best_assignment: np.ndarray | None = None
    best_rmse = float("inf")
    for tau in tau_candidates:
        assignment, rmse = _species_assignment_with_tau(
            source_frac=source_in_vanilla_fractional,
            source_atomic_numbers=standardized_atomic_numbers,
            target_frac=vanilla_frac,
            target_atomic_numbers=vanilla_atomic_numbers,
            tau=tau,
        )
        if rmse < best_rmse:
            best_tau = tau.copy()
            best_assignment = assignment.copy()
            best_rmse = float(rmse)

    if best_assignment is None:
        raise RuntimeError("Could not estimate standardized-to-vanilla semantic transport.")
    return linear, best_tau, best_assignment, best_rmse


def estimate_semantic_transport_for_reference_order(
    *,
    standardized_reference_frac_coords: np.ndarray,
    standardized_reference_atomic_numbers: np.ndarray,
    bridge: SymmetryFrameBridge,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate a semantic transport for a specific standardized atom ordering.

    This is the right fit when a downstream algorithm uses a standardized
    expansion order coming from PyXtal/template logic rather than the raw atom
    order of `bridge.standardized_structure`.
    """
    if bridge.standardized_to_vanilla_linear is None:
        raise ValueError("Bridge does not contain a semantic standardized-to-vanilla linear map.")

    source_frac = _wrap01_numpy(np.asarray(standardized_reference_frac_coords, dtype=float))
    source_atomic_numbers = np.asarray(standardized_reference_atomic_numbers, dtype=int).reshape(-1)
    target_frac = _wrap01_numpy(np.asarray(bridge.vanilla_structure.frac_coords, dtype=float))
    target_atomic_numbers = np.asarray(bridge.vanilla_structure.atomic_numbers, dtype=int).reshape(-1)
    linear = np.asarray(bridge.standardized_to_vanilla_linear, dtype=float)
    source_in_vanilla_fractional = _wrap01_numpy(source_frac @ linear)

    species_values = sorted(set(int(v) for v in source_atomic_numbers.tolist()))
    tau_candidates: list[np.ndarray] = []
    for atomic_number in species_values:
        src_idx = np.where(source_atomic_numbers == atomic_number)[0]
        dst_idx = np.where(target_atomic_numbers == atomic_number)[0]
        if src_idx.size == 0:
            continue
        for src_atom in src_idx:
            diffs = _torus_delta_numpy(target_frac[dst_idx], source_in_vanilla_fractional[src_atom])
            for diff in diffs:
                tau_candidates.append(np.asarray(diff, dtype=float).reshape(3))

    if not tau_candidates:
        tau_candidates = [np.zeros(3, dtype=float)]

    best_tau = np.zeros(3, dtype=float)
    best_assignment: np.ndarray | None = None
    best_rmse = float("inf")
    for tau in tau_candidates:
        assignment, rmse = _species_assignment_with_tau(
            source_frac=source_in_vanilla_fractional,
            source_atomic_numbers=source_atomic_numbers,
            target_frac=target_frac,
            target_atomic_numbers=target_atomic_numbers,
            tau=tau,
        )
        if rmse < best_rmse:
            best_tau = tau.copy()
            best_assignment = assignment.copy()
            best_rmse = float(rmse)

    if best_assignment is None:
        raise RuntimeError("Could not estimate semantic transport for the requested standardized reference order.")
    return best_tau, best_assignment, best_rmse


def transport_standardized_frac_to_vanilla_frame_with_tau(
    *,
    standardized_frac_coords: np.ndarray,
    bridge: SymmetryFrameBridge,
    tau: np.ndarray,
) -> np.ndarray:
    if bridge.standardized_to_vanilla_linear is None:
        raise ValueError("Bridge does not contain a semantic standardized-to-vanilla linear map.")
    frac = _wrap01_numpy(np.asarray(standardized_frac_coords, dtype=float))
    linear = np.asarray(bridge.standardized_to_vanilla_linear, dtype=float)
    return _wrap01_numpy(frac @ linear + np.asarray(tau, dtype=float).reshape(1, 3))


def transport_vanilla_frac_to_standardized_frame_with_tau(
    *,
    vanilla_frac_coords: np.ndarray,
    bridge: SymmetryFrameBridge,
    tau: np.ndarray,
) -> np.ndarray:
    if bridge.standardized_to_vanilla_linear is None:
        raise ValueError("Bridge does not contain a semantic standardized-to-vanilla linear map.")
    frac = _wrap01_numpy(np.asarray(vanilla_frac_coords, dtype=float))
    linear = np.asarray(bridge.standardized_to_vanilla_linear, dtype=float)
    inverse_linear = np.linalg.inv(linear)
    centered = _wrap01_numpy(frac - np.asarray(tau, dtype=float).reshape(1, 3))
    return _wrap01_numpy(centered @ inverse_linear)


def standardize_structure(
    structure,
    *,
    standardization: str = "conventional",
    symprec: float = 1e-2,
    angle_tolerance: float = 5.0,
):
    _require_pymatgen()
    analyzer = SpacegroupAnalyzer(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    if standardization == "conventional":
        standardized = analyzer.get_conventional_standard_structure()
    elif standardization == "primitive":
        standardized = analyzer.get_primitive_standard_structure()
    elif standardization == "refined":
        standardized = analyzer.get_refined_structure()
    else:
        raise ValueError(f"Unknown standardization={standardization!r}")
    return analyzer, standardized


def build_symmetry_frame_bridge(
    *,
    vanilla_structure,
    standardization: str = "conventional",
    symprec: float = 1e-2,
    angle_tolerance: float = 5.0,
    stol: float = 0.5,
    ltol: float = 0.3,
):
    """Create a sampling-time bridge between KLDM's vanilla frame and a standardized symmetry frame.

    The standardized structure is used for PyXtal/Wyckoff reasoning only. The
    KLDM pipeline can stay untouched in its original frame, while projection or
    template logic works on the standardized copy.
    """
    _require_pymatgen()
    analyzer, standardized = standardize_structure(
        vanilla_structure,
        standardization=standardization,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    standardized_analyzer = SpacegroupAnalyzer(
        standardized,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    matcher = StructureMatcher(stol=stol, angle_tol=angle_tolerance, ltol=ltol)
    try:
        standardized_to_vanilla = matcher.get_s2_like_s1(vanilla_structure, standardized)
    except Exception:
        standardized_to_vanilla = standardized

    semantic_linear = None
    semantic_tau = None
    semantic_assignment = None
    semantic_rmse = None
    semantic_method = None
    try:
        semantic_linear, semantic_tau, semantic_assignment, semantic_rmse = _estimate_semantic_transport(
            standardized_structure=standardized,
            vanilla_structure=vanilla_structure,
        )
        semantic_method = "affine_semantic_transport"
    except Exception:
        semantic_method = "structure_matcher_fallback"

    return SymmetryFrameBridge(
        vanilla_structure=vanilla_structure,
        standardized_structure=standardized,
        standardized_to_vanilla_structure=standardized_to_vanilla,
        detected_space_group=int(analyzer.get_space_group_number()),
        standardized_space_group=int(standardized_analyzer.get_space_group_number()),
        standardization=str(standardization),
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
        standardized_to_vanilla_linear=semantic_linear,
        standardized_to_vanilla_tau=semantic_tau,
        standardized_to_vanilla_assignment=semantic_assignment,
        standardized_to_vanilla_rmse=semantic_rmse,
        standardized_to_vanilla_method=semantic_method,
    )


def transport_standardized_frac_to_vanilla_frame(
    *,
    standardized_frac_coords: np.ndarray,
    bridge: SymmetryFrameBridge,
) -> np.ndarray:
    """Transport standardized fractional coordinates into vanilla fractional space.

    This transport is chart-preserving: it applies a fixed affine fractional map
    inferred once from the bridge's standardized/vanilla GT pair instead of
    re-running a structure matcher on each predicted structure.
    """
    if bridge.standardized_to_vanilla_linear is None or bridge.standardized_to_vanilla_tau is None:
        raise ValueError("Bridge does not contain a semantic standardized-to-vanilla transport.")
    frac = _wrap01_numpy(np.asarray(standardized_frac_coords, dtype=float))
    linear = np.asarray(bridge.standardized_to_vanilla_linear, dtype=float)
    tau = np.asarray(bridge.standardized_to_vanilla_tau, dtype=float).reshape(1, 3)
    return _wrap01_numpy(frac @ linear + tau)


def transport_vanilla_frac_to_standardized_frame(
    *,
    vanilla_frac_coords: np.ndarray,
    bridge: SymmetryFrameBridge,
) -> np.ndarray:
    if bridge.standardized_to_vanilla_linear is None or bridge.standardized_to_vanilla_tau is None:
        raise ValueError("Bridge does not contain a semantic standardized-to-vanilla transport.")
    return transport_vanilla_frac_to_standardized_frame_with_tau(
        vanilla_frac_coords=vanilla_frac_coords,
        bridge=bridge,
        tau=np.asarray(bridge.standardized_to_vanilla_tau, dtype=float),
    )


def transport_standardized_structure_to_vanilla_frame(
    *,
    standardized_structure,
    bridge: SymmetryFrameBridge,
):
    """Build a vanilla-lattice structure using the bridge's semantic transport."""
    _require_pymatgen()
    transported_frac = transport_standardized_frac_to_vanilla_frame(
        standardized_frac_coords=np.asarray(standardized_structure.frac_coords, dtype=float),
        bridge=bridge,
    )
    return Structure(
        lattice=Lattice(np.asarray(bridge.vanilla_structure.lattice.matrix, dtype=float)),
        species=list(standardized_structure.species),
        coords=transported_frac,
        coords_are_cartesian=False,
    )


def transport_standardized_tangent_block_to_vanilla_frame(
    *,
    tangent_block: np.ndarray,
    bridge: SymmetryFrameBridge,
) -> np.ndarray:
    """Transport a stacked standardized tangent block into vanilla fractional coordinates.

    `tangent_block` is expected to have shape `[3 * multiplicity, dof]`, matching the
    local orbit Jacobian convention used by the symmetry/projector code.
    """
    if bridge.standardized_to_vanilla_linear is None:
        raise ValueError("Bridge does not contain a semantic standardized-to-vanilla transport.")
    block = np.asarray(tangent_block, dtype=float)
    if block.ndim != 2 or block.shape[0] % 3 != 0:
        raise ValueError(f"Expected tangent_block with shape [3*m, dof], got {block.shape}.")
    linear = np.asarray(bridge.standardized_to_vanilla_linear, dtype=float)
    transformed = block.copy()
    for start in range(0, block.shape[0], 3):
        transformed[start : start + 3, :] = linear.T @ block[start : start + 3, :]
    return transformed


def map_standardized_structure_to_vanilla_frame(
    *,
    standardized_structure,
    vanilla_reference_structure,
    symprec: float = 1e-2,
    angle_tolerance: float = 5.0,
    stol: float = 0.5,
    ltol: float = 0.3,
):
    """Express a standardized-frame structure in a vanilla-KLDM-like frame when possible."""
    _require_pymatgen()
    matcher = StructureMatcher(stol=stol, angle_tol=angle_tolerance, ltol=ltol)
    try:
        return matcher.get_s2_like_s1(vanilla_reference_structure, standardized_structure)
    except Exception:
        analyzer, fallback = standardize_structure(
            vanilla_reference_structure,
            standardization="conventional",
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        del analyzer
        try:
            return matcher.get_s2_like_s1(vanilla_reference_structure, fallback)
        except Exception:
            return standardized_structure


def structure_to_tensors(structure) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_pymatgen()
    frac = torch.as_tensor(np.asarray(structure.frac_coords, dtype=float), dtype=torch.get_default_dtype())
    lattice = torch.as_tensor(np.asarray(structure.lattice.matrix, dtype=float), dtype=torch.get_default_dtype())
    atomic_numbers = torch.as_tensor(np.asarray(structure.atomic_numbers, dtype=int), dtype=torch.long)
    return frac, lattice, atomic_numbers
