from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from kldmPlus.data.transform import (
    ContinuousIntervalLattice,
    DEFAULT_ATOMIC_VOCAB,
    KLDMContinuousIntervalLattice,
)

try:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Element, Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:  # pragma: no cover
    Element = Lattice = Structure = StructureMatcher = SpacegroupAnalyzer = None


TensorLike = torch.Tensor | list[float] | list[int] | list[list[float]]


@dataclass
class CSPReconstructionResult:
    valid: bool
    match: bool
    rmse: float | None
    predicted_structure: Any
    target_structure: Any
    formula: str | None = None
    num_atoms: int | None = None
    composition_match: bool | None = None
    requested_space_group: int | None = None
    detected_space_group: int | None = None
    requested_space_group_match: bool | None = None
    validity_reason: str | None = None
    min_pair_distance: float | None = None
    volume: float | None = None
    max_lattice_length: float | None = None
    frac_rmse: float | None = None
    frac_rmse_status: str | None = None
    lattice_lengths_mae: float | None = None
    lattice_angles_mae: float | None = None
    lattice_lengths_rmse: float | None = None
    lattice_angles_rmse: float | None = None
    volume_rel_error: float | None = None
    matcher_diagnostics: Any | None = None


@dataclass
class SpeciesMatchDiagnostics:
    atomic_number: int
    symbol: str
    count: int
    rmse: float | None
    mean_distance: float | None
    max_distance: float | None
    mean_torus_shift: tuple[float, float, float] | None = None
    max_shift_deviation: float | None = None
    predicted_orbits: list[str] = field(default_factory=list)
    target_orbits: list[str] = field(default_factory=list)


@dataclass
class MatchFailureDiagnostics:
    diagnosis: str
    predicted_standardized_structure: Any | None = None
    target_standardized_structure: Any | None = None
    predicted_primitive_structure: Any | None = None
    target_primitive_structure: Any | None = None
    standardized_predicted_space_group: int | None = None
    standardized_target_space_group: int | None = None
    primitive_predicted_space_group: int | None = None
    primitive_target_space_group: int | None = None
    conventional_match: bool = False
    conventional_rmse: float | None = None
    primitive_match: bool = False
    primitive_rmse: float | None = None
    standardized_frac_rmse: float | None = None
    standardized_frac_status: str | None = None
    species_errors: list[SpeciesMatchDiagnostics] = field(default_factory=list)


# Stops structure evaluation early if pymatgen is unavailable.
def _require_pymatgen() -> None:
    if None in (Element, Lattice, Structure, StructureMatcher):
        raise ImportError("sample_evaluation requires pymatgen.")


# Builds the lattice inverse transform used by the decoding helpers.
def _lattice_transform(transform: ContinuousIntervalLattice | None) -> ContinuousIntervalLattice:
    return transform or KLDMContinuousIntervalLattice(standardize=False)


# Converts an input value to a tensor and promotes vectors to shape [1, d].
def _row_tensor(x: TensorLike) -> torch.Tensor:
    tensor = torch.as_tensor(x, dtype=torch.get_default_dtype())
    return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor


# Tries a structure conversion step and falls back to the original object on failure.
def _try_convert(structure: Structure | None, fn) -> Structure | None:
    if structure is None:
        return None
    try:
        converted = fn(structure)
        return structure if converted is None else converted
    except Exception:
        return structure


def _torus_pairwise_distance_sq(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = source[:, None, :] - target[None, :, :]
    delta = delta - np.round(delta)
    return np.sum(delta * delta, axis=-1)


def _coerce_frac_coords_local(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise ValueError(f"Expected flat fractional coordinates with size multiple of 3, got shape {arr.shape}.")
        arr = arr.reshape(-1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected fractional coordinates with shape [N, 3], got {arr.shape}.")
    return np.array(arr, dtype=float, copy=False)


def _coerce_atomic_numbers_local(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=int)
    return np.array(arr, dtype=int, copy=False).reshape(-1)


def _match_cost_matrix_np(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        linear_sum_assignment = None

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


def _species_aware_torus_rmse_local(
    *,
    source_frac_coords: np.ndarray,
    source_atomic_numbers: np.ndarray,
    target_frac_coords: np.ndarray,
    target_atomic_numbers: np.ndarray,
) -> tuple[float | None, str | None]:
    source_frac_coords = _coerce_frac_coords_local(source_frac_coords)
    target_frac_coords = _coerce_frac_coords_local(target_frac_coords)
    source_atomic_numbers = _coerce_atomic_numbers_local(source_atomic_numbers)
    target_atomic_numbers = _coerce_atomic_numbers_local(target_atomic_numbers)

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
        row_idx, col_idx = _match_cost_matrix_np(cost_matrix)
        deltas = src[row_idx] - tgt[col_idx]
        deltas = deltas - np.round(deltas)
        total_sq += float(np.sum(deltas * deltas))
        total_count += int(deltas.size)

    if total_count == 0:
        return None, "empty_matching"
    return float(np.sqrt(total_sq / total_count)), "ok"


def _species_aware_torus_diagnostics_local(
    *,
    source_frac_coords: np.ndarray,
    source_atomic_numbers: np.ndarray,
    target_frac_coords: np.ndarray,
    target_atomic_numbers: np.ndarray,
) -> tuple[float | None, str | None, list[SpeciesMatchDiagnostics]]:
    source_frac_coords = _coerce_frac_coords_local(source_frac_coords)
    target_frac_coords = _coerce_frac_coords_local(target_frac_coords)
    source_atomic_numbers = _coerce_atomic_numbers_local(source_atomic_numbers)
    target_atomic_numbers = _coerce_atomic_numbers_local(target_atomic_numbers)

    if len(source_frac_coords) != len(target_frac_coords):
        return None, "num_atoms_mismatch", []

    total_sq = 0.0
    total_count = 0
    species_values = sorted(set(int(v) for v in source_atomic_numbers.tolist()))
    if species_values != sorted(set(int(v) for v in target_atomic_numbers.tolist())):
        return None, "species_mismatch", []

    diagnostics: list[SpeciesMatchDiagnostics] = []
    for atomic_number in species_values:
        source_mask = source_atomic_numbers == atomic_number
        target_mask = target_atomic_numbers == atomic_number
        if int(np.sum(source_mask)) != int(np.sum(target_mask)):
            return None, "species_count_mismatch", diagnostics

        src = source_frac_coords[source_mask]
        tgt = target_frac_coords[target_mask]
        if len(src) == 0:
            diagnostics.append(
                SpeciesMatchDiagnostics(
                    atomic_number=atomic_number,
                    symbol=Element.from_Z(int(atomic_number)).symbol,
                    count=0,
                    rmse=None,
                    mean_distance=None,
                    max_distance=None,
                )
            )
            continue

        cost_matrix = _torus_pairwise_distance_sq(src, tgt)
        row_idx, col_idx = _match_cost_matrix_np(cost_matrix)
        deltas = src[row_idx] - tgt[col_idx]
        deltas = deltas - np.round(deltas)
        norms = np.linalg.norm(deltas, axis=-1)
        mean_shift = np.mean(deltas, axis=0)
        shift_deviation = np.max(np.linalg.norm(deltas - mean_shift[None, :], axis=-1))
        species_sq = float(np.sum(deltas * deltas))
        total_sq += species_sq
        total_count += int(deltas.size)
        diagnostics.append(
            SpeciesMatchDiagnostics(
                atomic_number=atomic_number,
                symbol=Element.from_Z(int(atomic_number)).symbol,
                count=int(len(src)),
                rmse=float(np.sqrt(species_sq / max(int(deltas.size), 1))),
                mean_distance=float(np.mean(norms)),
                max_distance=float(np.max(norms)),
                mean_torus_shift=tuple(float(v) for v in mean_shift.tolist()),
                max_shift_deviation=float(shift_deviation),
            )
        )

    if total_count == 0:
        return None, "empty_matching", diagnostics
    return float(np.sqrt(total_sq / total_count)), "ok", diagnostics


def _conventional_standard_structure(structure: Structure) -> Structure:
    if SpacegroupAnalyzer is None:
        return structure
    return _try_convert(
        structure,
        lambda s: SpacegroupAnalyzer(s).get_conventional_standard_structure(),
    )


def _primitive_standard_structure(structure: Structure) -> Structure:
    if SpacegroupAnalyzer is None:
        return structure
    analyzer = SpacegroupAnalyzer(structure)
    try:
        return analyzer.get_primitive_standard_structure()
    except Exception:
        try:
            return analyzer.get_primitive_structure()
        except Exception:
            return structure


def _symmetrized_species_orbits(
    structure: Structure,
    *,
    symprec: float,
    angle_tolerance: float,
) -> dict[int, list[str]]:
    if SpacegroupAnalyzer is None:
        return {}
    try:
        symmetrized = SpacegroupAnalyzer(
            structure,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        ).get_symmetrized_structure()
    except Exception:
        return {}

    per_species: dict[int, list[str]] = {}
    try:
        equivalent_sites = list(symmetrized.equivalent_sites)
        wyckoff_symbols = list(symmetrized.wyckoff_symbols)
    except Exception:
        return {}

    for sites, label in zip(equivalent_sites, wyckoff_symbols):
        if not sites:
            continue
        atomic_number = int(sites[0].specie.Z)
        per_species.setdefault(atomic_number, []).append(str(label))

    for atomic_number in per_species:
        per_species[atomic_number] = sorted(per_species[atomic_number])
    return per_species


def _safe_rms_dist(
    matcher: StructureMatcher,
    source: Structure,
    target: Structure,
) -> float | None:
    try:
        rms = matcher.get_rms_dist(source, target)
    except Exception:
        return None
    return None if rms is None else float(rms[0])


def diagnose_structure_mismatch(
    *,
    predicted: Structure,
    target: Structure,
    stol: float = 0.5,
    angle_tol: float = 10.0,
    ltol: float = 0.3,
    sg_symprec: float = 1e-2,
    sg_angle_tolerance: float = 5.0,
) -> MatchFailureDiagnostics:
    matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
    predicted_standardized = _conventional_standard_structure(predicted)
    target_standardized = _conventional_standard_structure(target)
    predicted_primitive = _primitive_standard_structure(predicted)
    target_primitive = _primitive_standard_structure(target)

    conventional_rmse = _safe_rms_dist(matcher, predicted_standardized, target_standardized)
    primitive_rmse = _safe_rms_dist(matcher, predicted_primitive, target_primitive)

    aligned_standardized = _try_convert(
        predicted_standardized,
        lambda s: matcher.get_s2_like_s1(target_standardized, s),
    )
    standardized_frac_rmse, standardized_frac_status, species_errors = _species_aware_torus_diagnostics_local(
        source_frac_coords=np.asarray(aligned_standardized.frac_coords, dtype=float),
        source_atomic_numbers=np.asarray(aligned_standardized.atomic_numbers, dtype=int),
        target_frac_coords=np.asarray(target_standardized.frac_coords, dtype=float),
        target_atomic_numbers=np.asarray(target_standardized.atomic_numbers, dtype=int),
    )
    predicted_orbits = _symmetrized_species_orbits(
        predicted_standardized,
        symprec=sg_symprec,
        angle_tolerance=sg_angle_tolerance,
    )
    target_orbits = _symmetrized_species_orbits(
        target_standardized,
        symprec=sg_symprec,
        angle_tolerance=sg_angle_tolerance,
    )
    for species_diag in species_errors:
        species_diag.predicted_orbits = list(predicted_orbits.get(species_diag.atomic_number, []))
        species_diag.target_orbits = list(target_orbits.get(species_diag.atomic_number, []))

    if conventional_rmse is not None:
        diagnosis = "equivalent_after_conventional_standardization"
    elif primitive_rmse is not None:
        diagnosis = "equivalent_in_primitive_basis_only"
    else:
        diagnosis = "different_motif_after_standardization"

    return MatchFailureDiagnostics(
        diagnosis=diagnosis,
        predicted_standardized_structure=predicted_standardized,
        target_standardized_structure=target_standardized,
        predicted_primitive_structure=predicted_primitive,
        target_primitive_structure=target_primitive,
        standardized_predicted_space_group=detect_space_group_number(
            predicted_standardized,
            symprec=sg_symprec,
            angle_tolerance=sg_angle_tolerance,
        ),
        standardized_target_space_group=detect_space_group_number(
            target_standardized,
            symprec=sg_symprec,
            angle_tolerance=sg_angle_tolerance,
        ),
        primitive_predicted_space_group=detect_space_group_number(
            predicted_primitive,
            symprec=sg_symprec,
            angle_tolerance=sg_angle_tolerance,
        ),
        primitive_target_space_group=detect_space_group_number(
            target_primitive,
            symprec=sg_symprec,
            angle_tolerance=sg_angle_tolerance,
        ),
        conventional_match=conventional_rmse is not None,
        conventional_rmse=conventional_rmse,
        primitive_match=primitive_rmse is not None,
        primitive_rmse=primitive_rmse,
        standardized_frac_rmse=standardized_frac_rmse,
        standardized_frac_status=standardized_frac_status,
        species_errors=species_errors,
    )


# Decodes atom ids or logits into atomic numbers and element symbols.
def decode_atom_types(
    a: TensorLike,
    species_vocab: list[int] | None = None,
) -> tuple[list[int], list[str]]:
    _require_pymatgen()

    atom_tensor = torch.as_tensor(a)
    vocab = species_vocab or DEFAULT_ATOMIC_VOCAB
    atomic_numbers = (
        [int(v) for v in atom_tensor.tolist()]
        if atom_tensor.ndim == 1
        else [int(vocab[int(i)]) for i in atom_tensor.argmax(dim=-1).tolist()]
    )
    species = [Element.from_Z(z).symbol for z in atomic_numbers]
    return atomic_numbers, species


# Decodes the transformed lattice back to lengths and angles in degrees.
def decode_lattice(
    l: TensorLike,
    n_atoms: int,
    lattice_transform: ContinuousIntervalLattice | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Forward the graph size so x0 lattice decode can choose the right stats.
    lengths, angles = _lattice_transform(lattice_transform).invert_to_lengths_angles(
        l=_row_tensor(l),
        num_atoms=n_atoms,
    )
    return lengths.squeeze(0), torch.rad2deg(angles.squeeze(0))


def decode_lattice_matrix(
    l: TensorLike,
    n_atoms: int,
    lattice_transform: ContinuousIntervalLattice | None = None,
) -> torch.Tensor:
    transform = _lattice_transform(lattice_transform)
    l_tensor = _row_tensor(l)
    if hasattr(transform, "invert_to_matrix"):
        matrix = transform.invert_to_matrix(l=l_tensor, num_atoms=n_atoms)
        return matrix.squeeze(0)

    lengths, angles_deg = decode_lattice(
        l=l_tensor,
        n_atoms=n_atoms,
        lattice_transform=transform,
    )
    return torch.as_tensor(
        Lattice.from_parameters(
            a=float(lengths[0]),
            b=float(lengths[1]),
            c=float(lengths[2]),
            alpha=float(angles_deg[0]),
            beta=float(angles_deg[1]),
            gamma=float(angles_deg[2]),
        ).matrix,
        dtype=torch.get_default_dtype(),
    )


# Reconstructs one periodic structure from sampled coordinates, lattice, and atom types.
def build_structure_from_sample(
    f: TensorLike,
    l: TensorLike,
    a: TensorLike,
    *,
    species_vocab: list[int] | None = None,
    lattice_transform: ContinuousIntervalLattice | None = None,
) -> Structure:
    _require_pymatgen()

    frac = _row_tensor(f)
    if frac.shape[-1] != 3:
        raise ValueError(f"Expected coordinates with last dim 3, got {tuple(frac.shape)}")
    if not torch.isfinite(frac).all():
        raise ValueError("Fractional coordinates contain non-finite values.")

    _, species = decode_atom_types(a=a, species_vocab=species_vocab)
    transform = _lattice_transform(lattice_transform)

    if hasattr(transform, "invert_to_matrix"):
        matrix = decode_lattice_matrix(
            l=l,
            n_atoms=int(frac.shape[0]),
            lattice_transform=transform,
        )
        if not torch.isfinite(matrix).all():
            raise ValueError("Decoded lattice matrix contains non-finite values.")
        lattice = Lattice(matrix.detach().cpu().numpy())
    else:
        lengths, angles_deg = decode_lattice(
            l=l,
            n_atoms=int(frac.shape[0]),
            lattice_transform=transform,
        )
        if not torch.isfinite(lengths).all() or not torch.isfinite(angles_deg).all():
            raise ValueError("Decoded lattice contains non-finite values.")
        if not (lengths > 0.0).all():
            raise ValueError("Decoded lattice contains non-positive lengths.")
        lattice = Lattice.from_parameters(
            a=float(lengths[0]),
            b=float(lengths[1]),
            c=float(lengths[2]),
            alpha=float(angles_deg[0]),
            beta=float(angles_deg[1]),
            gamma=float(angles_deg[2]),
        )

    return Structure(
        lattice=lattice,
        species=species,
        coords=(frac % 1.0).detach().cpu().tolist(),
        coords_are_cartesian=False,
    ).get_sorted_structure()


# Rejects crystals with overlapping atoms or obviously broken cells.
def validity_structure(structure: Structure, cutoff: float = 0.5) -> bool:
    try:
        distances = np.asarray(structure.distance_matrix, dtype=float)
    except Exception:
        return False

    distances += np.diag(np.full(distances.shape[0], cutoff + 10.0))
    return not (
        distances.min() < cutoff
        or structure.volume < 0.1
        or max(structure.lattice.abc) > 40.0
    )


def validity_structure_reason(
    structure: Structure,
    *,
    cutoff: float = 0.5,
) -> tuple[bool, str, float | None, float | None, float | None]:
    try:
        distances = np.asarray(structure.distance_matrix, dtype=float)
    except Exception:
        return False, "distance_matrix_failed", None, None, None

    distances = distances + np.diag(np.full(distances.shape[0], cutoff + 10.0))
    min_pair_distance = float(np.min(distances))
    volume = float(structure.volume)
    max_lattice_length = float(max(structure.lattice.abc))

    if min_pair_distance < cutoff:
        return False, "close_contacts", min_pair_distance, volume, max_lattice_length
    if volume < 0.1:
        return False, "tiny_volume", min_pair_distance, volume, max_lattice_length
    if max_lattice_length > 40.0:
        return False, "huge_lattice", min_pair_distance, volume, max_lattice_length
    return True, "ok", min_pair_distance, volume, max_lattice_length


def _atomic_multiset_match(left: TensorLike, right: TensorLike) -> bool:
    left_tensor = torch.as_tensor(left, dtype=torch.long).reshape(-1)
    right_tensor = torch.as_tensor(right, dtype=torch.long).reshape(-1)
    if left_tensor.shape != right_tensor.shape:
        return False
    left_sorted = torch.sort(left_tensor).values
    right_sorted = torch.sort(right_tensor).values
    return bool(torch.equal(left_sorted, right_sorted))


def detect_space_group_number(
    structure: Structure,
    *,
    symprec: float = 1e-2,
    angle_tolerance: float = 5.0,
) -> int | None:
    if SpacegroupAnalyzer is None:
        return None


def _space_group_to_family(space_group_number: int | None) -> str | None:
    if space_group_number is None:
        return None
    sg = int(space_group_number)
    if not 1 <= sg <= 230:
        return None
    if sg <= 2:
        return "triclinic"
    if sg <= 15:
        return "monoclinic"
    if sg <= 74:
        return "orthorhombic"
    if sg <= 142:
        return "tetragonal"
    if sg <= 167:
        return "trigonal"
    if sg <= 194:
        return "hexagonal"
    return "cubic"
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


def _lattice_mae(predicted: Structure, target: Structure) -> tuple[float, float]:
    pred_lengths = np.asarray(predicted.lattice.abc, dtype=float)
    target_lengths = np.asarray(target.lattice.abc, dtype=float)
    pred_angles = np.asarray(predicted.lattice.angles, dtype=float)
    target_angles = np.asarray(target.lattice.angles, dtype=float)
    lengths_mae = float(np.mean(np.abs(pred_lengths - target_lengths)))
    angles_mae = float(np.mean(np.abs(pred_angles - target_angles)))
    return lengths_mae, angles_mae


def _lattice_rmse(predicted: Structure, target: Structure) -> tuple[float, float]:
    pred_lengths = np.asarray(predicted.lattice.abc, dtype=float)
    target_lengths = np.asarray(target.lattice.abc, dtype=float)
    pred_angles = np.asarray(predicted.lattice.angles, dtype=float)
    target_angles = np.asarray(target.lattice.angles, dtype=float)
    lengths_rmse = float(np.sqrt(np.mean(np.square(pred_lengths - target_lengths))))
    angles_rmse = float(np.sqrt(np.mean(np.square(pred_angles - target_angles))))
    return lengths_rmse, angles_rmse


def _volume_rel_error(predicted: Structure, target: Structure) -> float | None:
    target_volume = float(target.lattice.volume)
    if abs(target_volume) <= 1.0e-12:
        return None
    predicted_volume = float(predicted.lattice.volume)
    return float(abs(predicted_volume - target_volume) / abs(target_volume))


# Aligns and optionally standardizes structures for easier visualization.
def prepare_visualization_pair(
    predicted_structure: Structure | None,
    target_structure: Structure | None,
) -> tuple[Structure | None, Structure | None]:
    _require_pymatgen()

    if predicted_structure is None or target_structure is None:
        return predicted_structure, target_structure

    matcher = StructureMatcher()
    predicted_structure = _try_convert(
        predicted_structure,
        lambda s: matcher.get_s2_like_s1(target_structure, s),
    )
    if SpacegroupAnalyzer is None:
        return predicted_structure, target_structure

    return (
        _try_convert(
            predicted_structure,
            lambda s: SpacegroupAnalyzer(s).get_conventional_standard_structure(),
        ),
        _try_convert(
            target_structure,
            lambda s: SpacegroupAnalyzer(s).get_conventional_standard_structure(),
        ),
    )


# Evaluates one predicted-target pair with matcher-based validity, match, and RMSE.
def evaluate_csp_reconstruction(
    *,
    pred_f: TensorLike,
    pred_l: TensorLike,
    pred_a: TensorLike,
    target_f: TensorLike,
    target_l: TensorLike,
    target_a: TensorLike,
    species_vocab: list[int] | None = None,
    lattice_transform: ContinuousIntervalLattice | None = None,
    stol: float = 0.5,
    angle_tol: float = 10.0,
    ltol: float = 0.3,
    requested_space_group: int | None = None,
    sg_symprec: float = 1e-2,
    sg_angle_tolerance: float = 5.0,
    validity_cutoff: float = 0.5,
) -> CSPReconstructionResult:
    transform = _lattice_transform(lattice_transform)
    num_atoms = int(_row_tensor(target_f).shape[0])

    try:
        predicted = build_structure_from_sample(
            pred_f,
            pred_l,
            pred_a,
            species_vocab=species_vocab,
            lattice_transform=transform,
        )
    except Exception:
        return CSPReconstructionResult(
            False,
            False,
            None,
            None,
            None,
            None,
            num_atoms,
            composition_match=None,
            requested_space_group=requested_space_group,
            detected_space_group=None,
            requested_space_group_match=None,
            validity_reason="predicted_build_failed",
        )

    try:
        target = build_structure_from_sample(
            target_f,
            target_l,
            target_a,
            species_vocab=species_vocab,
            lattice_transform=transform,
        )
    except Exception:
        return CSPReconstructionResult(
            False,
            False,
            None,
            predicted,
            None,
            predicted.composition.formula,
            num_atoms,
            composition_match=None,
            requested_space_group=requested_space_group,
            detected_space_group=detect_space_group_number(
                predicted,
                symprec=sg_symprec,
                angle_tolerance=sg_angle_tolerance,
            ),
            requested_space_group_match=None,
            validity_reason="target_build_failed",
        )

    is_valid, validity_reason, min_pair_distance, volume, max_lattice_length = validity_structure_reason(
        predicted,
        cutoff=float(validity_cutoff),
    )
    matched = False
    rmse = None
    composition_match = _atomic_multiset_match(
        np.asarray(predicted.atomic_numbers, dtype=int),
        decode_atom_types(a=target_a, species_vocab=species_vocab)[0],
    )
    detected_space_group = detect_space_group_number(
        predicted,
        symprec=sg_symprec,
        angle_tolerance=sg_angle_tolerance,
    )
    requested_space_group_match = (
        None if requested_space_group is None or detected_space_group is None
        else bool(int(requested_space_group) == int(detected_space_group))
    )
    frac_rmse, frac_rmse_status = _species_aware_torus_rmse_local(
        source_frac_coords=np.asarray(predicted.frac_coords, dtype=float),
        source_atomic_numbers=np.asarray(predicted.atomic_numbers, dtype=int),
        target_frac_coords=np.asarray(target.frac_coords, dtype=float),
        target_atomic_numbers=np.asarray(target.atomic_numbers, dtype=int),
    )
    lattice_lengths_mae, lattice_angles_mae = _lattice_mae(predicted, target)
    lattice_lengths_rmse, lattice_angles_rmse = _lattice_rmse(predicted, target)
    volume_rel_error = _volume_rel_error(predicted, target)
    matcher_diagnostics = None

    if is_valid:
        try:
            rms = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol).get_rms_dist(
                predicted,
                target,
            )
            matched = rms is not None
            rmse = None if rms is None else float(rms[0])
        except Exception:
            pass
        if not matched:
            try:
                matcher_diagnostics = diagnose_structure_mismatch(
                    predicted=predicted,
                    target=target,
                    stol=stol,
                    angle_tol=angle_tol,
                    ltol=ltol,
                    sg_symprec=sg_symprec,
                    sg_angle_tolerance=sg_angle_tolerance,
                )
            except Exception:
                matcher_diagnostics = None

    return CSPReconstructionResult(
        valid=is_valid,
        match=matched,
        rmse=rmse,
        predicted_structure=predicted,
        target_structure=target,
        formula=predicted.composition.formula,
        num_atoms=num_atoms,
        composition_match=composition_match,
        requested_space_group=requested_space_group,
        detected_space_group=detected_space_group,
        requested_space_group_match=requested_space_group_match,
        validity_reason=validity_reason,
        min_pair_distance=min_pair_distance,
        volume=volume,
        max_lattice_length=max_lattice_length,
        frac_rmse=frac_rmse,
        frac_rmse_status=frac_rmse_status,
        lattice_lengths_mae=lattice_lengths_mae,
        lattice_angles_mae=lattice_angles_mae,
        lattice_lengths_rmse=lattice_lengths_rmse,
        lattice_angles_rmse=lattice_angles_rmse,
        volume_rel_error=volume_rel_error,
        matcher_diagnostics=matcher_diagnostics,
    )


# Aggregates per-sample CSP results into validity, match-rate, and RMSE summaries.
def aggregate_csp_reconstruction_metrics(
    results: list[CSPReconstructionResult],
) -> dict[str, Any]:
    if not results:
        return {"num_samples": 0, "valid": None, "match_rate": None, "rmse": None}

    valid = [float(result.valid) for result in results]
    match = [float(result.match) for result in results]
    rmse = [float(result.rmse) for result in results if result.rmse is not None]
    frac_rmse = [float(result.frac_rmse) for result in results if result.frac_rmse is not None]
    standardized_frac_rmse = [
        float(result.matcher_diagnostics.standardized_frac_rmse)
        for result in results
        if result.matcher_diagnostics is not None
        and result.matcher_diagnostics.standardized_frac_rmse is not None
    ]
    composition_matches = [
        float(result.composition_match)
        for result in results
        if result.composition_match is not None
    ]
    space_group_matches = [
        float(result.requested_space_group_match)
        for result in results
        if result.requested_space_group_match is not None
    ]
    matcher_diagnosis_counts: dict[str, int] = {}
    for result in results:
        diagnosis = None
        if result.matcher_diagnostics is not None:
            diagnosis = result.matcher_diagnostics.diagnosis
        if diagnosis and not result.match:
            matcher_diagnosis_counts[str(diagnosis)] = matcher_diagnosis_counts.get(str(diagnosis), 0) + 1

    detected_sg_agreement = [
        float(result.requested_space_group_match)
        for result in results
        if result.requested_space_group_match is not None
    ]
    detected_family_agreement = []
    lattice_lengths_rmse = []
    lattice_angles_rmse = []
    volume_rel_error = []
    for result in results:
        requested_family = _space_group_to_family(result.requested_space_group)
        detected_family = _space_group_to_family(result.detected_space_group)
        if requested_family is not None and detected_family is not None:
            detected_family_agreement.append(float(requested_family == detected_family))
        if result.lattice_lengths_rmse is not None:
            lattice_lengths_rmse.append(float(result.lattice_lengths_rmse))
        if result.lattice_angles_rmse is not None:
            lattice_angles_rmse.append(float(result.lattice_angles_rmse))
        if result.volume_rel_error is not None:
            volume_rel_error.append(float(result.volume_rel_error))

    return {
        "num_samples": len(results),
        "valid": float(sum(valid) / len(valid)),
        "match_rate": float(sum(match) / len(match)),
        "rmse": None if not rmse else float(sum(rmse) / len(rmse)),
        "rmse_defined_count": len(rmse),
        "frac_rmse": None if not frac_rmse else float(sum(frac_rmse) / len(frac_rmse)),
        "frac_rmse_defined_count": len(frac_rmse),
        "standardized_frac_rmse": (
            None
            if not standardized_frac_rmse
            else float(sum(standardized_frac_rmse) / len(standardized_frac_rmse))
        ),
        "standardized_frac_rmse_defined_count": len(standardized_frac_rmse),
        "composition_match_rate": (
            None if not composition_matches else float(sum(composition_matches) / len(composition_matches))
        ),
        "requested_space_group_match_rate": (
            None if not space_group_matches else float(sum(space_group_matches) / len(space_group_matches))
        ),
        "detected_sg_agreement": (
            None if not detected_sg_agreement else float(sum(detected_sg_agreement) / len(detected_sg_agreement))
        ),
        "detected_family_agreement": (
            None
            if not detected_family_agreement
            else float(sum(detected_family_agreement) / len(detected_family_agreement))
        ),
        "lattice_lengths_rmse": (
            None
            if not lattice_lengths_rmse
            else float(np.mean(np.asarray(lattice_lengths_rmse, dtype=float)))
        ),
        "lattice_angles_rmse": (
            None
            if not lattice_angles_rmse
            else float(np.mean(np.asarray(lattice_angles_rmse, dtype=float)))
        ),
        "volume_rel_error": (
            None if not volume_rel_error else float(sum(volume_rel_error) / len(volume_rel_error))
        ),
        "matcher_diagnosis_counts": matcher_diagnosis_counts,
    }


def aggregate_csp_reconstruction_metrics_by_size(
    results: list[CSPReconstructionResult],
) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[CSPReconstructionResult]] = {}
    for result in results:
        if result.num_atoms is None:
            continue
        grouped.setdefault(int(result.num_atoms), []).append(result)

    summary_by_size: dict[int, dict[str, Any]] = {}
    for num_atoms in sorted(grouped):
        summary_by_size[num_atoms] = aggregate_csp_reconstruction_metrics(grouped[num_atoms])
    return summary_by_size
