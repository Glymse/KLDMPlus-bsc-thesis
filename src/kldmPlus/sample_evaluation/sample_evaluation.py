from __future__ import annotations

from dataclasses import dataclass
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
        return CSPReconstructionResult(False, False, None, None, None, None, num_atoms)

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
            False, False, None, predicted, None, predicted.composition.formula, num_atoms
        )

    is_valid = validity_structure(predicted)
    matched = False
    rmse = None

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

    return CSPReconstructionResult(
        valid=is_valid,
        match=matched,
        rmse=rmse,
        predicted_structure=predicted,
        target_structure=target,
        formula=predicted.composition.formula,
        num_atoms=num_atoms,
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

    return {
        "num_samples": len(results),
        "valid": float(sum(valid) / len(valid)),
        "match_rate": float(sum(match) / len(match)),
        "rmse": None if not rmse else float(sum(rmse) / len(rmse)),
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
