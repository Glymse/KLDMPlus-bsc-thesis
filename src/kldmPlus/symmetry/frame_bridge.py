from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

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

    @property
    def vanilla_atomic_numbers(self) -> np.ndarray:
        return np.asarray(self.vanilla_structure.atomic_numbers, dtype=int)

    @property
    def standardized_atomic_numbers(self) -> np.ndarray:
        return np.asarray(self.standardized_structure.atomic_numbers, dtype=int)


def _require_pymatgen() -> None:
    if None in (Element, Lattice, SpacegroupAnalyzer, Structure, StructureMatcher):
        raise ImportError("Symmetry frame bridge requires pymatgen.")


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

    return SymmetryFrameBridge(
        vanilla_structure=vanilla_structure,
        standardized_structure=standardized,
        standardized_to_vanilla_structure=standardized_to_vanilla,
        detected_space_group=int(analyzer.get_space_group_number()),
        standardized_space_group=int(standardized_analyzer.get_space_group_number()),
        standardization=str(standardization),
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
    )


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
