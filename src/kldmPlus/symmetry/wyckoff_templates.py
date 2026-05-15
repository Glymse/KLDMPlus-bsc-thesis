from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

try:
    from pyxtal.symmetry import Group
except ImportError:  # pragma: no cover
    Group = None

try:
    from pymatgen.core import Element
except ImportError:  # pragma: no cover
    Element = None


@dataclass(frozen=True)
class WyckoffSiteTemplate:
    atomic_number: int
    label: str
    multiplicity: int
    dof: int
    free_coordinate_mask: tuple[bool, bool, bool]
    anchor_basis: np.ndarray
    anchor_offset: np.ndarray
    rotation_matrices: np.ndarray
    translation_vectors: np.ndarray
    site_symmetry: str | None = None


@dataclass(frozen=True)
class WyckoffTemplate:
    space_group: int
    group_symbol: str
    species_order: tuple[int, ...]
    species_counts: tuple[int, ...]
    site_templates: tuple[WyckoffSiteTemplate, ...]
    has_free_coordinates: bool
    pyxtal_site_index_groups: tuple[tuple[int, ...], ...] | None = None

    @property
    def total_atoms(self) -> int:
        return int(sum(site.multiplicity for site in self.site_templates))

    @property
    def total_sites(self) -> int:
        return len(self.site_templates)

    @property
    def total_free_dims(self) -> int:
        return int(sum(site.dof for site in self.site_templates))


@dataclass
class WyckoffExpansion:
    frac_coords: torch.Tensor
    atomic_numbers: torch.Tensor
    anchor_index: torch.Tensor
    anchor_coords: torch.Tensor


def _require_template_dependencies() -> None:
    missing: list[str] = []
    if Group is None:
        missing.append("pyxtal")
    if Element is None:
        missing.append("pymatgen")
    if missing:
        raise ImportError("Wyckoff template utilities require: " + ", ".join(missing) + ".")


def composition_to_species_counts(atomic_numbers: list[int] | np.ndarray | torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Preserve first-seen species order and return `(species_order, counts)`."""
    values = [int(v) for v in torch.as_tensor(atomic_numbers, dtype=torch.long).reshape(-1).tolist()]
    counts: dict[int, int] = {}
    order: list[int] = []
    for value in values:
        if value not in counts:
            counts[value] = 0
            order.append(value)
        counts[value] += 1
    return tuple(order), tuple(int(counts[value]) for value in order)


def conventional_cell_multiplicity(space_group_number: int) -> int:
    _require_template_dependencies()
    group = Group(int(space_group_number))
    symbol = str(group.symbol).strip().upper()
    centering = symbol[0] if symbol else "P"
    if centering in {"A", "B", "C", "I"}:
        return 2
    if centering == "F":
        return 4
    if centering == "R":
        return 3
    return 1


def requested_conventional_atomic_numbers(
    atomic_numbers: list[int] | np.ndarray | torch.Tensor,
    *,
    space_group_number: int,
) -> torch.Tensor:
    base = torch.as_tensor(atomic_numbers, dtype=torch.long).reshape(-1)
    multiplicity = conventional_cell_multiplicity(int(space_group_number))
    if multiplicity <= 1:
        return base.clone()
    return base.repeat(int(multiplicity))


def requested_composition_key(
    *,
    space_group_number: int,
    atomic_numbers: list[int] | np.ndarray | torch.Tensor,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    conventional_atomic_numbers = requested_conventional_atomic_numbers(
        atomic_numbers,
        space_group_number=int(space_group_number),
    )
    species_order, species_counts = composition_to_species_counts(conventional_atomic_numbers)
    return int(space_group_number), species_order, species_counts


def _lookup_wyckoff_position(group: Any, label: str):
    for wp in group:
        if str(wp.get_label()) == str(label):
            return wp
    raise KeyError(f"Could not find Wyckoff position {label!r} in space group {group.number}.")


def _pivot_columns(matrix: np.ndarray, tol: float = 1e-8) -> list[int]:
    rank = int(np.linalg.matrix_rank(matrix, tol=tol))
    pivots: list[int] = []
    if rank == 0:
        return pivots

    current = np.zeros((matrix.shape[0], 0), dtype=float)
    for col_idx in range(matrix.shape[1]):
        candidate = matrix[:, pivots + [col_idx]]
        if np.linalg.matrix_rank(candidate, tol=tol) > current.shape[1]:
            pivots.append(col_idx)
            current = candidate
        if len(pivots) == rank:
            break
    return pivots


def _canonical_free_axes_from_wp(wp: Any) -> list[int] | None:
    """Return PyXtal's canonical free-coordinate axes when available.

    PyXtal's `get_frozen_axis()` describes which of the ambient fractional axes
    are fixed for the canonical Wyckoff parameterization. We prefer that chart
    over rank/pivot inference because it preserves PyXtal's intended local
    coordinates while still letting the anchor rotation encode correlations such
    as `(x, x, z)`.
    """
    if not hasattr(wp, "get_frozen_axis"):
        return None
    try:
        frozen_axes = list(wp.get_frozen_axis() or [])
    except Exception:
        return None

    frozen_set = {int(axis) for axis in frozen_axes if 0 <= int(axis) < 3}
    free_axes = [axis for axis in range(3) if axis not in frozen_set]
    expected_dof = int(wp.get_dof())
    if len(free_axes) != expected_dof:
        return None
    return free_axes


def _site_template_from_wp(*, atomic_number: int, wp: Any) -> WyckoffSiteTemplate:
    anchor_op = wp.ops[0]
    anchor_rotation = np.asarray(anchor_op.rotation_matrix, dtype=float)
    anchor_offset = np.asarray(anchor_op.translation_vector, dtype=float)
    canonical_free_axes = _canonical_free_axes_from_wp(wp)
    if canonical_free_axes is not None:
        anchor_basis = anchor_rotation[:, canonical_free_axes] if canonical_free_axes else np.zeros((3, 0), dtype=float)
        free_mask = tuple(bool(i in canonical_free_axes) for i in range(3))
    else:
        pivots = _pivot_columns(anchor_rotation)
        anchor_basis = anchor_rotation[:, pivots] if pivots else np.zeros((3, 0), dtype=float)
        free_mask = tuple(bool(i in pivots) for i in range(3))

    if not hasattr(wp, "site_symm"):
        try:
            wp.get_site_symmetry()
        except Exception:
            pass

    rotations = np.asarray([np.asarray(op.rotation_matrix, dtype=float) for op in wp.ops], dtype=float)
    translations = np.asarray([np.asarray(op.translation_vector, dtype=float) for op in wp.ops], dtype=float)

    return WyckoffSiteTemplate(
        atomic_number=int(atomic_number),
        label=str(wp.get_label()),
        multiplicity=int(wp.multiplicity),
        dof=int(wp.get_dof()),
        free_coordinate_mask=free_mask,
        anchor_basis=anchor_basis,
        anchor_offset=anchor_offset,
        rotation_matrices=rotations,
        translation_vectors=translations,
        site_symmetry=getattr(wp, "site_symm", None),
    )


def extract_wyckoff_templates(
    *,
    space_group_number: int,
    atomic_numbers: list[int] | np.ndarray | torch.Tensor,
    max_templates: int | None = 32,
    quick: bool = False,
    num_wp: tuple[int | None, int | None] = (None, None),
    nmax: int = 10_000_000,
) -> list[WyckoffTemplate]:
    """Enumerate candidate PyXtal Wyckoff templates for `(composition, SG)`.

    Uses `Group.list_wyckoff_combinations(...)` to turn composition counts into
    candidate assignments of Wyckoff letters, then converts each assignment into
    a differentiable template containing:

    - species -> site assignments
    - multiplicities
    - per-site free-coordinate masks
    - affine basis/offset for anchor coordinates
    - symmetry ops for orbit expansion
    """
    _require_template_dependencies()

    group = Group(int(space_group_number))
    species_order, species_counts = composition_to_species_counts(atomic_numbers)

    try:
        combinations, freedom_flags, wp_index_groups = group.list_wyckoff_combinations(
            list(species_counts),
            quick=quick,
            numWp=num_wp,
            Nmax=nmax,
        )
    except TypeError:
        combinations, freedom_flags, wp_index_groups = group.list_wyckoff_combinations(
            list(species_counts),
            quick=quick,
        )

    templates: list[WyckoffTemplate] = []
    limit = len(combinations) if max_templates is None else min(len(combinations), int(max_templates))
    for combo_idx in range(limit):
        combo = combinations[combo_idx]
        has_free = bool(freedom_flags[combo_idx]) if combo_idx < len(freedom_flags) else False
        site_templates: list[WyckoffSiteTemplate] = []
        for atomic_number, labels in zip(species_order, combo):
            for label in labels:
                wp = _lookup_wyckoff_position(group, str(label))
                site_templates.append(_site_template_from_wp(atomic_number=int(atomic_number), wp=wp))

        templates.append(
            WyckoffTemplate(
                space_group=int(group.number),
                group_symbol=str(group.symbol),
                species_order=species_order,
                species_counts=species_counts,
                site_templates=tuple(site_templates),
                has_free_coordinates=has_free,
                pyxtal_site_index_groups=(
                    tuple(tuple(int(v) for v in group_ids) for group_ids in wp_index_groups[combo_idx])
                    if combo_idx < len(wp_index_groups)
                    else None
                ),
            )
        )

    return templates


def flatten_site_signature(template: WyckoffTemplate) -> tuple[tuple[int, str], ...]:
    """Canonical `(species, Wyckoff label)` multiset signature for matching."""
    pairs = [(site.atomic_number, site.label) for site in template.site_templates]
    return tuple(sorted(pairs, key=lambda item: (item[0], item[1])))


def sample_random_free_vars(
    template: WyckoffTemplate,
    *,
    batch_size: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    total_dims = template.total_free_dims
    shape = (total_dims,) if batch_size is None else (int(batch_size), total_dims)
    return torch.rand(shape, device=device, dtype=dtype)


def expand_wyckoff_template_torch(
    *,
    template: WyckoffTemplate,
    free_vars: torch.Tensor,
    wrap: bool = True,
) -> WyckoffExpansion:
    """Expand flattened free Wyckoff variables into full fractional coordinates."""
    tensor = torch.as_tensor(free_vars)
    squeeze = tensor.ndim == 1
    if squeeze:
        tensor = tensor.unsqueeze(0)

    if tensor.shape[-1] != template.total_free_dims:
        raise ValueError(
            f"Expected free_vars.shape[-1] == {template.total_free_dims}, got {tensor.shape[-1]}.",
        )

    batch_size = tensor.shape[0]
    device = tensor.device
    dtype = tensor.dtype

    coord_blocks: list[torch.Tensor] = []
    species_blocks: list[torch.Tensor] = []
    anchor_blocks: list[torch.Tensor] = []
    anchor_index_blocks: list[torch.Tensor] = []
    cursor = 0

    for site_idx, site in enumerate(template.site_templates):
        basis = torch.as_tensor(site.anchor_basis, device=device, dtype=dtype)
        offset = torch.as_tensor(site.anchor_offset, device=device, dtype=dtype)
        if site.dof == 0:
            anchor = offset.expand(batch_size, 3)
        else:
            site_free = tensor[:, cursor : cursor + site.dof]
            anchor = site_free @ basis.transpose(0, 1) + offset
            cursor += site.dof

        rotations = torch.as_tensor(site.rotation_matrices, device=device, dtype=dtype)
        translations = torch.as_tensor(site.translation_vectors, device=device, dtype=dtype)
        expanded = torch.einsum("bd,med->bme", anchor, rotations) + translations.unsqueeze(0)
        if wrap:
            expanded = torch.remainder(expanded, 1.0)

        coord_blocks.append(expanded)
        anchor_blocks.append(anchor.unsqueeze(1))
        species_blocks.append(
            torch.full(
                (site.multiplicity,),
                int(site.atomic_number),
                device=device,
                dtype=torch.long,
            )
        )
        anchor_index_blocks.append(
            torch.full((site.multiplicity,), int(site_idx), device=device, dtype=torch.long)
        )

    frac_coords = torch.cat(coord_blocks, dim=1) if coord_blocks else torch.zeros((batch_size, 0, 3), device=device, dtype=dtype)
    anchor_coords = torch.cat(anchor_blocks, dim=1) if anchor_blocks else torch.zeros((batch_size, 0, 3), device=device, dtype=dtype)
    atomic_numbers = (
        torch.cat(species_blocks, dim=0)
        if species_blocks
        else torch.zeros((0,), device=device, dtype=torch.long)
    )
    anchor_index = (
        torch.cat(anchor_index_blocks, dim=0)
        if anchor_index_blocks
        else torch.zeros((0,), device=device, dtype=torch.long)
    )

    if squeeze:
        frac_coords = frac_coords.squeeze(0)
        anchor_coords = anchor_coords.squeeze(0)

    return WyckoffExpansion(
        frac_coords=frac_coords,
        atomic_numbers=atomic_numbers,
        anchor_index=anchor_index,
        anchor_coords=anchor_coords,
    )
