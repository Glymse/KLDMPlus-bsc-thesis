from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kldmPlus.symmetry import (
    build_pyxtal_wyckoff_result,
    expand_wyckoff_template_torch,
    extract_wyckoff_templates,
    flatten_site_signature,
    requested_conventional_atomic_numbers,
    species_aware_torus_rmse,
)


def _require_symmetry_backends() -> None:
    pytest.importorskip("pymatgen")
    pytest.importorskip("pyxtal")


def _single_site_template(*, space_group: int, atomic_number: int, multiplicity: int, label: str):
    templates = extract_wyckoff_templates(
        space_group_number=space_group,
        atomic_numbers=[atomic_number] * multiplicity,
        max_templates=256,
        quick=False,
    )
    matches = [
        template
        for template in templates
        if len(template.site_templates) == 1
        and template.site_templates[0].atomic_number == atomic_number
        and template.site_templates[0].label == label
    ]
    assert matches, f"Could not find single-site template {atomic_number}@{label} in SG {space_group}."
    return matches[0]


def _template_with_signature(
    *,
    space_group: int,
    atomic_numbers: list[int],
    expected_signature: tuple[tuple[int, str], ...],
):
    templates = extract_wyckoff_templates(
        space_group_number=space_group,
        atomic_numbers=atomic_numbers,
        max_templates=256,
        quick=False,
    )
    matches = [
        template
        for template in templates
        if flatten_site_signature(template) == expected_signature
    ]
    assert matches, (
        f"Could not find template with signature={expected_signature!r} "
        f"in SG {space_group}. Found signatures: "
        f"{[flatten_site_signature(template) for template in templates[:16]]!r}"
    )
    return matches[0]


def _template_with_any_signature(
    *,
    space_group: int,
    atomic_numbers: list[int],
    candidate_signatures: tuple[tuple[tuple[int, str], ...], ...],
):
    templates = extract_wyckoff_templates(
        space_group_number=space_group,
        atomic_numbers=atomic_numbers,
        max_templates=256,
        quick=False,
    )
    by_signature = {flatten_site_signature(template): template for template in templates}
    for signature in candidate_signatures:
        if signature in by_signature:
            return by_signature[signature], signature
    assert False, (
        f"Could not find any candidate signature={candidate_signatures!r} in SG {space_group}. "
        f"Found signatures: {[flatten_site_signature(template) for template in templates[:32]]!r}"
    )


def _species_aware_torus_rmse_with_global_shift(
    *,
    source_frac_coords: np.ndarray,
    source_atomic_numbers: np.ndarray,
    target_frac_coords: np.ndarray,
    target_atomic_numbers: np.ndarray,
) -> tuple[float | None, str | None]:
    species_values = sorted(set(int(v) for v in source_atomic_numbers.tolist()))
    if species_values != sorted(set(int(v) for v in target_atomic_numbers.tolist())):
        return None, "species_mismatch"

    counts = {species: int(np.sum(source_atomic_numbers == species)) for species in species_values}
    anchor_species = min(species_values, key=lambda species: (counts[species], species))
    src_anchor = source_frac_coords[source_atomic_numbers == anchor_species]
    tgt_anchor = target_frac_coords[target_atomic_numbers == anchor_species]

    best_rmse: float | None = None
    best_status: str | None = None
    for src in src_anchor:
        for tgt in tgt_anchor:
            shift = (tgt - src) - np.round(tgt - src)
            shifted = np.remainder(source_frac_coords + shift, 1.0)
            rmse, status = species_aware_torus_rmse(
                source_frac_coords=shifted,
                source_atomic_numbers=source_atomic_numbers,
                target_frac_coords=target_frac_coords,
                target_atomic_numbers=target_atomic_numbers,
            )
            if rmse is None:
                best_status = status
                continue
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_status = status

    return best_rmse, best_status


def _structured_free_vars(num_dims: int) -> torch.Tensor:
    if num_dims == 0:
        return torch.empty((0,), dtype=torch.float64)
    values = torch.linspace(0.173, 0.173 + 0.111 * (num_dims - 1), num_dims, dtype=torch.float64)
    return torch.remainder(values, 0.45)


def _solve_site_free_vars(site_template, anchor_frac: np.ndarray) -> torch.Tensor:
    if site_template.dof == 0:
        return torch.empty((0,), dtype=torch.float64)
    basis = np.asarray(site_template.anchor_basis, dtype=float)
    offset = np.asarray(site_template.anchor_offset, dtype=float)
    solution, *_ = np.linalg.lstsq(basis, np.asarray(anchor_frac, dtype=float) - offset, rcond=None)
    return torch.tensor(solution, dtype=torch.float64)


def _expected_anchor_from_wp(site_template, free_vars: torch.Tensor) -> np.ndarray:
    basis = np.asarray(site_template.anchor_basis, dtype=float)
    offset = np.asarray(site_template.anchor_offset, dtype=float)
    if site_template.dof == 0:
        return np.asarray(offset, dtype=float)
    values = np.asarray(torch.as_tensor(free_vars, dtype=torch.float64).detach().cpu().numpy(), dtype=float)
    return values @ basis.T + offset


@pytest.mark.parametrize(
    ("space_group", "label", "atomic_number", "multiplicity", "lattice_kind", "lattice_params"),
    [
        (123, "4i", 12, 4, "tetragonal", (4.7, 6.5)),
        (123, "4j", 12, 4, "tetragonal", (4.7, 6.5)),
        (194, "6g", 22, 6, "hexagonal", (4.9, 6.1)),
        (194, "6h", 22, 6, "hexagonal", (4.9, 6.1)),
        (227, "8a", 70, 8, "cubic", (5.3,)),
    ],
)
def test_expand_wyckoff_template_matches_pyxtal_ops_single_site(
    space_group: int,
    label: str,
    atomic_number: int,
    multiplicity: int,
    lattice_kind: str,
    lattice_params: tuple[float, ...],
) -> None:
    _require_symmetry_backends()
    from pyxtal.symmetry import Group

    template = _single_site_template(
        space_group=space_group,
        atomic_number=atomic_number,
        multiplicity=multiplicity,
        label=label,
    )
    free_vars = _structured_free_vars(template.total_free_dims)
    expansion = expand_wyckoff_template_torch(
        template=template,
        free_vars=free_vars,
    )

    group = Group(space_group)
    wp = next(cand for cand in group if str(cand.get_label()) == label)
    anchor = _expected_anchor_from_wp(template.site_templates[0], free_vars) % 1.0
    expected = np.asarray([np.asarray(op.operate(anchor), dtype=float) % 1.0 for op in wp], dtype=float)

    rmse, status = species_aware_torus_rmse(
        source_frac_coords=expansion.frac_coords.detach().cpu().numpy(),
        source_atomic_numbers=expansion.atomic_numbers.detach().cpu().numpy(),
        target_frac_coords=expected,
        target_atomic_numbers=np.full((len(expected),), atomic_number, dtype=int),
    )

    assert status == "ok"
    assert rmse is not None
    assert rmse < 1e-6


@pytest.mark.parametrize(
    ("space_group", "primitive_atomic_numbers", "candidate_signatures", "lattice_kind", "lattice_params"),
    [
        (
            123,
            [3, 12, 12, 12, 12, 14, 14],
            (
                ((3, "1a"), (12, "4i"), (14, "2e")),
                ((3, "1a"), (12, "4i"), (14, "2f")),
            ),
            "tetragonal",
            (4.7, 6.5),
        ),
        (
            194,
            [72, 72, 22, 22, 22, 22, 22, 22],
            (((22, "6h"), (72, "2d")),),
            "hexagonal",
            (4.9, 6.1),
        ),
        (
            227,
            [70, 70, 77, 77, 77, 77],
            (((70, "8a"), (77, "16d")),),
            "cubic",
            (5.3,),
        ),
    ],
)
def test_pyxtal_wyckoff_roundtrip_recovers_equivalent_multisite_structure(
    space_group: int,
    primitive_atomic_numbers: list[int],
    candidate_signatures: tuple[tuple[tuple[int, str], ...], ...],
    lattice_kind: str,
    lattice_params: tuple[float, ...],
) -> None:
    _require_symmetry_backends()
    from pymatgen.core import Element, Lattice, Structure

    conventional_atomic_numbers = requested_conventional_atomic_numbers(
        primitive_atomic_numbers,
        space_group_number=space_group,
    ).tolist()
    template, selected_signature = _template_with_any_signature(
        space_group=space_group,
        atomic_numbers=conventional_atomic_numbers,
        candidate_signatures=candidate_signatures,
    )
    free_vars = _structured_free_vars(template.total_free_dims)
    expansion = expand_wyckoff_template_torch(
        template=template,
        free_vars=free_vars,
    )

    if lattice_kind == "tetragonal":
        lattice = Lattice.tetragonal(*lattice_params)
    elif lattice_kind == "hexagonal":
        lattice = Lattice.hexagonal(*lattice_params)
    elif lattice_kind == "cubic":
        lattice = Lattice.cubic(*lattice_params)
    else:
        raise ValueError(f"Unsupported lattice_kind={lattice_kind!r}.")

    structure = Structure(
        lattice=lattice,
        species=[Element.from_Z(int(z)).symbol for z in expansion.atomic_numbers.tolist()],
        coords=expansion.frac_coords.detach().cpu().numpy(),
        coords_are_cartesian=False,
    ).get_sorted_structure()

    result = build_pyxtal_wyckoff_result(
        structure,
        symprec=1e-3,
        pyxtal_tol=1e-3,
    )

    debug_context = (
        f"requested SG/signature={space_group}/{selected_signature}, "
        f"recovered SG/labels={result.space_group}/{tuple(result.site_labels)}, "
        f"anchor_count={result.anchor_count}, "
        f"site_multiplicities={tuple(int(v) for v in result.site_multiplicities.tolist())}, "
        f"site_dofs={tuple(int(v) for v in result.site_dofs.tolist())}"
    )

    assert result.space_group == space_group, debug_context

    rmse, status = _species_aware_torus_rmse_with_global_shift(
        source_frac_coords=expansion.frac_coords.detach().cpu().numpy(),
        source_atomic_numbers=expansion.atomic_numbers.detach().cpu().numpy(),
        target_frac_coords=result.expanded_frac_coords,
        target_atomic_numbers=result.expanded_atomic_numbers,
    )

    assert status == "ok", debug_context
    assert rmse is not None, debug_context
    assert rmse < 1e-6, f"{debug_context}, rmse={rmse}"
