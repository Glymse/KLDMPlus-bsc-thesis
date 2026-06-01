from __future__ import annotations

# Inspired heavily by DiffCSP++

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from kldmPlus.symmetry.wyckoff_templates import (
    WyckoffTemplate,
    extract_wyckoff_templates,
    flatten_site_signature,
    recover_template_free_vars_from_anchor_entries,
)

try:
    from pymatgen.core import Element, Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:  # pragma: no cover
    Element = Lattice = Structure = SpacegroupAnalyzer = None

try:
    from pyxtal import pyxtal
    from pyxtal.symmetry import Group
except ImportError:  # pragma: no cover
    pyxtal = Group = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


@dataclass(frozen=True)
class DiffCSPPPSymmetryPayload:
    spacegroup: int
    anchor_index: np.ndarray
    wyckoff_ops: np.ndarray
    wyckoff_ops_inv: np.ndarray
    wyckoff_letters: tuple[str, ...]
    atom_types: np.ndarray
    anchor_frac_coords: np.ndarray
    expanded_frac_coords: np.ndarray
    anchor_atomic_numbers: np.ndarray
    expanded_atomic_numbers: np.ndarray
    lattice_matrix: np.ndarray
    anchor_dofs: np.ndarray | None = None
    anchor_free_coordinate_masks: np.ndarray | None = None
    standardized_structure: Any | None = None
    debug_info: dict[str, Any] | None = None

    @property
    def anchors(self) -> np.ndarray:
        """DiffCSP++ compatibility alias for `anchor_index`."""
        return self.anchor_index

    @property
    def ops(self) -> np.ndarray:
        """DiffCSP++ compatibility alias for `wyckoff_ops`."""
        return self.wyckoff_ops

    @property
    def ops_inv(self) -> np.ndarray:
        """DiffCSP++ compatibility alias for `wyckoff_ops_inv`."""
        return self.wyckoff_ops_inv

    @property
    def num_nodes(self) -> int:
        return int(self.expanded_frac_coords.shape[0])

    @property
    def num_atoms(self) -> int:
        return int(self.expanded_frac_coords.shape[0])

    @property
    def anchor_entries(self) -> list[dict[str, Any]]:
        return [
            {
                "atomic_number": int(self.anchor_atomic_numbers[i]),
                "label": str(self.wyckoff_letters[i]),
                "anchor_frac": np.asarray(self.anchor_frac_coords[i], dtype=float),
            }
            for i in range(int(len(self.wyckoff_letters)))
        ]

    @property
    def site_signature(self) -> tuple[tuple[int, str], ...]:
        return tuple(
            sorted(
                (int(self.anchor_atomic_numbers[i]), str(self.wyckoff_letters[i]))
                for i in range(int(len(self.wyckoff_letters)))
            )
        )

    def to_diffcsppp_dict(self) -> dict[str, Any]:
        """Return the core symmetry fields in a DiffCSP++-style layout."""
        return {
            "spacegroup": int(self.spacegroup),
            "ops": np.asarray(self.wyckoff_ops, dtype=float),
            "ops_inv": np.asarray(self.wyckoff_ops_inv, dtype=float),
            "anchor_index": np.asarray(self.anchor_index, dtype=int),
            "num_nodes": int(self.num_nodes),
            "num_atoms": int(self.num_atoms),
            "atom_types": np.asarray(self.atom_types, dtype=int),
            "wyckoff_letters": tuple(self.wyckoff_letters),
            "anchor_dofs": None if self.anchor_dofs is None else np.asarray(self.anchor_dofs, dtype=int),
            "anchor_free_coordinate_masks": (
                None if self.anchor_free_coordinate_masks is None
                else np.asarray(self.anchor_free_coordinate_masks, dtype=bool)
            ),
        }

    def to_torch_dict(self, *, device: torch.device | None = None) -> dict[str, torch.Tensor]:
        """Return tensors that match the fields DiffCSP++ passes into its sampler."""
        kwargs = {} if device is None else {"device": device}
        return {
            "spacegroup": torch.as_tensor([int(self.spacegroup)], dtype=torch.long, **kwargs),
            "ops": torch.as_tensor(self.wyckoff_ops, dtype=torch.float32, **kwargs),
            "ops_inv": torch.as_tensor(self.wyckoff_ops_inv, dtype=torch.float32, **kwargs),
            "anchor_index": torch.as_tensor(self.anchor_index, dtype=torch.long, **kwargs),
            "num_nodes": torch.as_tensor(int(self.num_nodes), dtype=torch.long, **kwargs),
            "num_atoms": torch.as_tensor(int(self.num_atoms), dtype=torch.long, **kwargs),
            "atom_types": torch.as_tensor(self.atom_types, dtype=torch.long, **kwargs),
            "anchor_dofs": torch.as_tensor(
                np.zeros((len(self.wyckoff_letters),), dtype=int) if self.anchor_dofs is None else self.anchor_dofs,
                dtype=torch.long,
                **kwargs,
            ),
            "anchor_free_coordinate_masks": torch.as_tensor(
                np.zeros((len(self.wyckoff_letters), 3), dtype=bool)
                if self.anchor_free_coordinate_masks is None else self.anchor_free_coordinate_masks,
                dtype=torch.bool,
                **kwargs,
            ),
        }


@dataclass(frozen=True)
class WyckoffDOFChart:
    payload: DiffCSPPPSymmetryPayload
    template: WyckoffTemplate
    q_ref: np.ndarray
    site_dof_slices: tuple[slice, ...]
    site_row_slices: tuple[tuple[int, int], ...]
    site_anchor_bases: tuple[np.ndarray, ...]
    site_anchor_offsets: tuple[np.ndarray, ...]
    site_reference_expanded_rows: tuple[np.ndarray, ...]
    site_dofs: np.ndarray

    @property
    def total_dof(self) -> int:
        return int(self.q_ref.shape[0])

    def anchor_from_q(self, site_idx: int, q_s: torch.Tensor | np.ndarray) -> torch.Tensor:
        basis = torch.as_tensor(self.site_anchor_bases[int(site_idx)], dtype=torch.get_default_dtype())
        offset = torch.as_tensor(self.site_anchor_offsets[int(site_idx)], dtype=torch.get_default_dtype())
        q_tensor = torch.as_tensor(q_s, dtype=offset.dtype)
        if basis.shape[1] == 0:
            return torch.remainder(offset, 1.0)
        return torch.remainder(q_tensor.reshape(1, -1) @ basis.transpose(0, 1) + offset.reshape(1, 3), 1.0).reshape(3)

    def expand_q(self, q: torch.Tensor, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        q_tensor = torch.as_tensor(q, device=device, dtype=dtype if dtype is not None else None)
        if q_tensor.ndim != 1:
            raise ValueError(f"Expected flattened q with shape [Q], got {tuple(q_tensor.shape)}.")
        if q_tensor.shape[0] != self.total_dof:
            raise ValueError(f"Expected q with {self.total_dof} dims, got {q_tensor.shape[0]}.")

        out = torch.zeros((int(self.payload.num_atoms), 3), device=q_tensor.device, dtype=q_tensor.dtype)
        ops = torch.as_tensor(self.payload.wyckoff_ops, device=q_tensor.device, dtype=q_tensor.dtype)
        rotations = ops[:, :3, :3]
        translations = ops[:, :3, 3]
        for site_idx, (row_start, row_stop) in enumerate(self.site_row_slices):
            dof_slice = self.site_dof_slices[site_idx]
            basis = torch.as_tensor(self.site_anchor_bases[site_idx], device=q_tensor.device, dtype=q_tensor.dtype)
            offset = torch.as_tensor(self.site_anchor_offsets[site_idx], device=q_tensor.device, dtype=q_tensor.dtype)
            if basis.shape[1] == 0:
                # A 0-DOF site is a constant orbit in the chosen reference chart.
                # Re-emitting the stored representative rows is faithful to the local
                # chart semantics and prevents representative drift for rigid sites.
                out[row_start:row_stop] = torch.as_tensor(
                    self.site_reference_expanded_rows[site_idx],
                    device=q_tensor.device,
                    dtype=q_tensor.dtype,
                )
                continue
            anchor = q_tensor[dof_slice] @ basis.transpose(0, 1) + offset
            out[row_start:row_stop] = torch.remainder(
                torch.einsum("nij,j->ni", rotations[row_start:row_stop], anchor) + translations[row_start:row_stop],
                1.0,
            )
        return out

    def jacobian_q(self, q: torch.Tensor | np.ndarray | None = None, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        q_dtype = dtype if dtype is not None else (torch.as_tensor(q).dtype if q is not None else torch.get_default_dtype())
        q_device = device if device is not None else (torch.as_tensor(q).device if q is not None else torch.device("cpu"))
        jac = torch.zeros((int(self.payload.num_atoms), 3, self.total_dof), device=q_device, dtype=q_dtype)
        ops = torch.as_tensor(self.payload.wyckoff_ops, device=q_device, dtype=q_dtype)
        rotations = ops[:, :3, :3]
        for site_idx, (row_start, row_stop) in enumerate(self.site_row_slices):
            dof_slice = self.site_dof_slices[site_idx]
            basis = torch.as_tensor(self.site_anchor_bases[site_idx], device=q_device, dtype=q_dtype)
            if basis.shape[1] == 0:
                continue
            jac[row_start:row_stop, :, dof_slice] = torch.einsum("nij,jd->nid", rotations[row_start:row_stop], basis)
        return jac

    def project_local(
        self,
        z_payload: torch.Tensor,
        *,
        q_ref: torch.Tensor | np.ndarray | None = None,
        lambda_q: float = 1.0e-6,
    ) -> dict[str, Any]:
        q_ref_t = torch.as_tensor(
            self.q_ref if q_ref is None else q_ref,
            device=z_payload.device,
            dtype=z_payload.dtype,
        ).reshape(-1)
        z_ref = self.expand_q(q_ref_t, device=z_payload.device, dtype=z_payload.dtype)
        jac = self.jacobian_q(q_ref_t, device=z_payload.device, dtype=z_payload.dtype)
        delta_z = torch.remainder(z_payload - z_ref + 0.5, 1.0) - 0.5
        rhs = delta_z.reshape(-1, 1)
        jac_flat = jac.reshape(-1, int(q_ref_t.shape[0]))
        if jac_flat.numel() == 0 or jac_flat.shape[1] == 0:
            delta_q = torch.zeros_like(q_ref_t)
        else:
            gram = jac_flat.transpose(0, 1) @ jac_flat
            gram = gram + float(lambda_q) * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            delta_q = torch.linalg.solve(gram, jac_flat.transpose(0, 1) @ rhs).reshape(-1)
        q_star = torch.remainder(q_ref_t + delta_q, 1.0)
        z_proj = self.expand_q(q_star, device=z_payload.device, dtype=z_payload.dtype)
        residual = torch.remainder(z_payload - z_proj + 0.5, 1.0) - 0.5
        per_site_c: list[float] = []
        per_site_q_step: list[float] = []
        for site_idx, (row_start, row_stop) in enumerate(self.site_row_slices):
            site_residual = residual[row_start:row_stop]
            per_site_c.append(float(site_residual.square().mean().detach().item()) if site_residual.numel() else 0.0)
            dof_slice = self.site_dof_slices[site_idx]
            q_delta_site = delta_q[dof_slice]
            per_site_q_step.append(float(torch.linalg.norm(q_delta_site).detach().item()) if q_delta_site.numel() else 0.0)
        return {
            "loss": residual.square().mean(),
            "q_ref": q_ref_t,
            "q_star": q_star,
            "delta_q": delta_q,
            "z_ref": z_ref,
            "z_proj": z_proj,
            "jacobian": jac,
            "per_site_c": per_site_c,
            "per_site_q_step": per_site_q_step,
        }


def _wrap01_numpy(value: np.ndarray) -> np.ndarray:
    return np.remainder(np.asarray(value, dtype=float), 1.0)


def _signed_wrap_numpy(value: np.ndarray) -> np.ndarray:
    return np.remainder(np.asarray(value, dtype=float) + 0.5, 1.0) - 0.5


def _torus_rmse_numpy(source: np.ndarray, target: np.ndarray) -> float:
    delta = _signed_wrap_numpy(np.asarray(source, dtype=float) - np.asarray(target, dtype=float))
    if delta.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(delta * delta)))


def _torus_pairwise_distance_sq_numpy(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = _signed_wrap_numpy(np.asarray(source, dtype=float)[:, None, :] - np.asarray(target, dtype=float)[None, :, :])
    return np.sum(delta * delta, axis=-1)


def _match_cost_matrix_numpy(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if linear_sum_assignment is not None:
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        return np.asarray(row_idx, dtype=int), np.asarray(col_idx, dtype=int)

    remaining_rows = list(range(cost_matrix.shape[0]))
    remaining_cols = list(range(cost_matrix.shape[1]))
    chosen_rows: list[int] = []
    chosen_cols: list[int] = []
    while remaining_rows:
        submatrix = cost_matrix[np.ix_(remaining_rows, remaining_cols)]
        flat_idx = int(np.argmin(submatrix))
        n_cols = submatrix.shape[1]
        row_pos = flat_idx // n_cols
        col_pos = flat_idx % n_cols
        chosen_rows.append(remaining_rows.pop(row_pos))
        chosen_cols.append(remaining_cols.pop(col_pos))
    order = np.argsort(np.asarray(chosen_rows, dtype=int))
    return np.asarray(chosen_rows, dtype=int)[order], np.asarray(chosen_cols, dtype=int)[order]


def _free_coordinate_mask_from_wp(wp: Any) -> np.ndarray:
    if hasattr(wp, "get_frozen_axis"):
        frozen_axes = list(wp.get_frozen_axis() or [])
        free_mask = np.ones(3, dtype=bool)
        free_mask[frozen_axes] = False
        return free_mask

    anchor_rotation = np.asarray(wp.ops[0].rotation_matrix, dtype=float)
    column_norms = np.linalg.norm(anchor_rotation, axis=0)
    return column_norms > 1e-8


def _unique_anchor_starts(anchor_index: np.ndarray) -> np.ndarray:
    values = np.asarray(anchor_index, dtype=int).reshape(-1)
    if values.size == 0:
        return np.zeros((0,), dtype=int)
    return np.unique(values)


def payload_site_slices(payload: DiffCSPPPSymmetryPayload) -> list[tuple[int, int]]:
    starts = _unique_anchor_starts(payload.anchor_index)
    stops = list(starts[1:]) + [int(payload.num_atoms)]
    return [(int(start), int(stop)) for start, stop in zip(starts.tolist(), stops)]


def payload_site_ids(payload: DiffCSPPPSymmetryPayload) -> np.ndarray:
    starts = _unique_anchor_starts(payload.anchor_index)
    if starts.size == 0:
        return np.zeros((0,), dtype=int)
    return np.searchsorted(starts, np.asarray(payload.anchor_index, dtype=int))


def reconstruct_expanded_frac_from_anchor_coords(
    payload: DiffCSPPPSymmetryPayload,
    anchor_frac_coords: np.ndarray,
) -> np.ndarray:
    """Lift anchor coordinates to full fractional coordinates via DiffCSP++ affine ops."""
    anchors = _wrap01_numpy(np.asarray(anchor_frac_coords, dtype=float))
    site_slices = payload_site_slices(payload)
    if anchors.shape != (len(site_slices), 3):
        raise ValueError(
            f"Expected anchor_frac_coords with shape ({len(site_slices)}, 3), got {anchors.shape}."
        )
    expanded = np.zeros((int(payload.num_atoms), 3), dtype=float)
    for site_idx, (start, stop) in enumerate(site_slices):
        anchor_h = np.concatenate([anchors[site_idx], np.ones(1, dtype=float)], axis=0)
        ops = np.asarray(payload.wyckoff_ops[start:stop], dtype=float)
        expanded[start:stop] = _wrap01_numpy((ops @ anchor_h.reshape(4, 1)).reshape(-1, 4)[:, :3])
    return expanded


def align_expanded_frac_to_reference_chart(
    payload: DiffCSPPPSymmetryPayload,
    expanded_frac_coords: np.ndarray,
    *,
    reference_expanded_frac_coords: np.ndarray | None = None,
    expanded_atomic_numbers: np.ndarray | None = None,
) -> dict[str, Any]:
    """Align payload-frame coordinates to a fixed local representative.

    The DiffCSP++ payload defines a feasible set, but nearby model-frame points can
    still land on a symmetry-equivalent representative with a different row order or
    origin choice. This helper picks a *fixed* representative near
    ``reference_expanded_frac_coords`` by minimizing species-aware torus RMSE over:

    - a global torus shift
    - a species-preserving row permutation

    The returned ``aligned_frac_coords`` uses the reference row order.
    """
    frac = _wrap01_numpy(np.asarray(expanded_frac_coords, dtype=float))
    ref = _wrap01_numpy(
        np.asarray(
            payload.expanded_frac_coords if reference_expanded_frac_coords is None else reference_expanded_frac_coords,
            dtype=float,
        )
    )
    species = np.asarray(
        payload.expanded_atomic_numbers if expanded_atomic_numbers is None else expanded_atomic_numbers,
        dtype=int,
    ).reshape(-1)
    ref_species = np.asarray(payload.expanded_atomic_numbers, dtype=int).reshape(-1)
    if frac.shape != ref.shape:
        raise ValueError(f"Expected expanded_frac_coords with shape {ref.shape}, got {frac.shape}.")
    if species.shape != ref_species.shape:
        raise ValueError(f"Expected expanded_atomic_numbers with shape {ref_species.shape}, got {species.shape}.")
    if not np.array_equal(np.sort(species), np.sort(ref_species)):
        raise ValueError("expanded_atomic_numbers do not match payload expanded_atomic_numbers up to permutation.")
    if frac.shape[0] == 0:
        return {
            "aligned_frac_coords": frac.copy(),
            "tau": np.zeros((3,), dtype=float),
            "reference_order": np.zeros((0,), dtype=int),
            "rmse": 0.0,
            "status": "empty",
        }

    species_values = sorted(set(int(v) for v in ref_species.tolist()))
    first_species = int(species[0])
    reference_candidates = np.where(ref_species == first_species)[0]
    if reference_candidates.size == 0:
        reference_candidates = np.arange(ref.shape[0], dtype=int)

    best_rmse = float("inf")
    best_tau = np.zeros((3,), dtype=float)
    best_order = np.arange(ref.shape[0], dtype=int)
    best_aligned = frac.copy()

    for ref_idx in reference_candidates.tolist():
        tau = _signed_wrap_numpy(ref[ref_idx] - frac[0])
        shifted = _wrap01_numpy(frac + tau.reshape(1, 3))
        order = np.zeros((shifted.shape[0],), dtype=int)
        total_sq = 0.0
        total_count = 0
        valid = True
        for atomic_number in species_values:
            src_rows = np.where(species == atomic_number)[0]
            tgt_rows = np.where(ref_species == atomic_number)[0]
            if src_rows.size != tgt_rows.size:
                valid = False
                break
            if src_rows.size == 0:
                continue
            cost = _torus_pairwise_distance_sq_numpy(shifted[src_rows], ref[tgt_rows])
            row_idx, col_idx = _match_cost_matrix_numpy(cost)
            matched_src = src_rows[row_idx]
            matched_tgt = tgt_rows[col_idx]
            order[matched_tgt] = matched_src
            deltas = _signed_wrap_numpy(shifted[matched_src] - ref[matched_tgt])
            total_sq += float(np.sum(deltas * deltas))
            total_count += int(deltas.size)
        if not valid or total_count == 0:
            continue
        aligned = shifted[order]
        rmse = float(np.sqrt(total_sq / total_count))
        if rmse < best_rmse:
            best_rmse = rmse
            best_tau = tau
            best_order = order.copy()
            best_aligned = aligned.copy()

    return {
        "aligned_frac_coords": best_aligned,
        "tau": _wrap01_numpy(best_tau),
        "reference_order": best_order,
        "rmse": float(best_rmse),
        "status": "ok" if np.isfinite(best_rmse) else "failed",
    }


def align_expanded_frac_to_reference_chart_orbit_aware(
    payload: DiffCSPPPSymmetryPayload,
    expanded_frac_coords: np.ndarray,
    *,
    reference_expanded_frac_coords: np.ndarray | None = None,
    expanded_atomic_numbers: np.ndarray | None = None,
) -> dict[str, Any]:
    """Align payload-frame coordinates to the reference representative orbit by orbit.

    This is stricter than the species-only alignment above: it preserves the
    payload's Wyckoff orbit partition and only permutes rows *within* each orbit
    block. That keeps nearby points in the same local chart instead of allowing
    a species-compatible but orbit-incompatible representative jump.
    """
    frac = _wrap01_numpy(np.asarray(expanded_frac_coords, dtype=float))
    ref = _wrap01_numpy(
        np.asarray(
            payload.expanded_frac_coords if reference_expanded_frac_coords is None else reference_expanded_frac_coords,
            dtype=float,
        )
    )
    species = np.asarray(
        payload.expanded_atomic_numbers if expanded_atomic_numbers is None else expanded_atomic_numbers,
        dtype=int,
    ).reshape(-1)
    ref_species = np.asarray(payload.expanded_atomic_numbers, dtype=int).reshape(-1)
    if frac.shape != ref.shape:
        raise ValueError(f"Expected expanded_frac_coords with shape {ref.shape}, got {frac.shape}.")
    if species.shape != ref_species.shape:
        raise ValueError(f"Expected expanded_atomic_numbers with shape {ref_species.shape}, got {species.shape}.")

    site_slices = payload_site_slices(payload)
    if frac.shape[0] == 0:
        return {
            "aligned_frac_coords": frac.copy(),
            "tau": np.zeros((3,), dtype=float),
            "reference_order": np.zeros((0,), dtype=int),
            "rmse": 0.0,
            "status": "empty",
        }

    first_start, first_stop = site_slices[0]
    reference_candidates = list(range(first_start, first_stop))
    if not reference_candidates:
        reference_candidates = [0]

    best_rmse = float("inf")
    best_tau = np.zeros((3,), dtype=float)
    best_order = np.arange(ref.shape[0], dtype=int)
    best_aligned = frac.copy()

    for ref_idx in reference_candidates:
        tau = _signed_wrap_numpy(ref[ref_idx] - frac[first_start])
        shifted = _wrap01_numpy(frac + tau.reshape(1, 3))
        order = np.zeros((shifted.shape[0],), dtype=int)
        total_sq = 0.0
        total_count = 0
        valid = True

        for start, stop in site_slices:
            src_rows = np.arange(start, stop, dtype=int)
            tgt_rows = np.arange(start, stop, dtype=int)
            if src_rows.size != tgt_rows.size:
                valid = False
                break
            if src_rows.size == 0:
                continue
            src_species = species[src_rows]
            tgt_species = ref_species[tgt_rows]
            if not np.array_equal(np.sort(src_species), np.sort(tgt_species)):
                valid = False
                break

            cost = _torus_pairwise_distance_sq_numpy(shifted[src_rows], ref[tgt_rows])
            penalty = (src_species[:, None] != tgt_species[None, :]).astype(float) * 1.0e6
            row_idx, col_idx = _match_cost_matrix_numpy(cost + penalty)
            matched_src = src_rows[row_idx]
            matched_tgt = tgt_rows[col_idx]
            if not np.array_equal(np.sort(src_species[row_idx]), np.sort(tgt_species[col_idx])):
                valid = False
                break
            order[matched_tgt] = matched_src
            deltas = _signed_wrap_numpy(shifted[matched_src] - ref[matched_tgt])
            total_sq += float(np.sum(deltas * deltas))
            total_count += int(deltas.size)

        if not valid or total_count == 0:
            continue
        aligned = shifted[order]
        rmse = float(np.sqrt(total_sq / total_count))
        if rmse < best_rmse:
            best_rmse = rmse
            best_tau = tau
            best_order = order.copy()
            best_aligned = aligned.copy()

    return {
        "aligned_frac_coords": best_aligned,
        "tau": _wrap01_numpy(best_tau),
        "reference_order": best_order,
        "rmse": float(best_rmse),
        "status": "ok" if np.isfinite(best_rmse) else "failed",
    }


def attach_payload_reference_chart(
    payload: DiffCSPPPSymmetryPayload,
    expanded_frac_coords: np.ndarray,
    *,
    expanded_atomic_numbers: np.ndarray | None = None,
    reference_expanded_frac_coords: np.ndarray | None = None,
) -> DiffCSPPPSymmetryPayload:
    """Attach a fixed local payload representative alignment to ``payload.debug_info``."""
    alignment = align_expanded_frac_to_reference_chart_orbit_aware(
        payload,
        expanded_frac_coords,
        reference_expanded_frac_coords=reference_expanded_frac_coords,
        expanded_atomic_numbers=expanded_atomic_numbers,
    )
    debug_info = dict(payload.debug_info or {})
    debug_info.update(
        {
            "payload_reference_tau": np.asarray(alignment["tau"], dtype=float),
            "payload_reference_order": np.asarray(alignment["reference_order"], dtype=int),
            "payload_reference_rmse": float(alignment["rmse"]),
            "payload_reference_status": str(alignment["status"]),
            "payload_reference_alignment_mode": "orbit_aware",
        }
    )
    return replace(payload, debug_info=debug_info)


def project_expanded_frac_to_anchor_space(
    payload: DiffCSPPPSymmetryPayload,
    expanded_frac_coords: np.ndarray,
    *,
    align_to_reference_chart: bool = False,
) -> dict[str, Any]:
    """Project full fractional coordinates into DiffCSP++ anchor space and lift them back.

    This mirrors the key Wyckoff-side idea in DiffCSP++: pull coordinates back to
    anchor space, average within each Wyckoff orbit, then reconstruct all atoms
    through the affine orbit operators.
    """
    frac = _wrap01_numpy(np.asarray(expanded_frac_coords, dtype=float))
    if frac.shape != (int(payload.num_atoms), 3):
        raise ValueError(f"Expected expanded_frac_coords with shape ({payload.num_atoms}, 3), got {frac.shape}.")
    alignment = None
    if align_to_reference_chart:
        alignment = align_expanded_frac_to_reference_chart_orbit_aware(payload, frac)
        frac = np.asarray(alignment["aligned_frac_coords"], dtype=float)

    ops = np.asarray(payload.wyckoff_ops, dtype=float)
    site_slices = payload_site_slices(payload)
    anchor_estimates = np.zeros_like(frac)
    anchor_means = np.zeros((len(site_slices), 3), dtype=float)
    site_debug: list[dict[str, Any]] = []

    for site_idx, (start, stop) in enumerate(site_slices):
        rows = slice(start, stop)
        site_ops = ops[rows]
        site_frac = frac[rows]
        site_estimates = []
        for atom_idx in range(site_frac.shape[0]):
            rotation = np.asarray(site_ops[atom_idx, :3, :3], dtype=float)
            translation = np.asarray(site_ops[atom_idx, :3, 3], dtype=float)
            delta = _signed_wrap_numpy(site_frac[atom_idx] - translation)
            anchor_est = np.linalg.pinv(rotation) @ delta
            site_estimates.append(anchor_est.reshape(1, 3))
        site_estimates_np = np.concatenate(site_estimates, axis=0) if site_estimates else np.zeros((0, 3), dtype=float)
        if site_estimates_np.shape[0] == 0:
            anchor_mean = np.zeros(3, dtype=float)
        else:
            reference = site_estimates_np[0]
            centered = _signed_wrap_numpy(site_estimates_np - reference.reshape(1, 3))
            anchor_mean = _wrap01_numpy(reference + centered.mean(axis=0))
        anchor_estimates[rows] = _wrap01_numpy(site_estimates_np)
        anchor_means[site_idx] = anchor_mean
        site_debug.append(
            {
                "site_index": int(site_idx),
                "label": str(payload.wyckoff_letters[site_idx]),
                "start": int(start),
                "stop": int(stop),
                "multiplicity": int(stop - start),
                "anchor_mean": anchor_mean.tolist(),
                "anchor_estimate_rmse": _torus_rmse_numpy(site_estimates_np, anchor_mean.reshape(1, 3)),
            }
        )

    lifted_expanded = reconstruct_expanded_frac_from_anchor_coords(payload, anchor_means)
    return {
        "anchor_estimates": anchor_estimates,
        "anchor_means": anchor_means,
        "lifted_expanded_frac_coords": lifted_expanded,
        "rmse": _torus_rmse_numpy(lifted_expanded, frac),
        "site_debug": site_debug,
        "alignment": alignment,
    }


def _require_dependencies() -> None:
    missing: list[str] = []
    if None in (Element, Lattice, Structure, SpacegroupAnalyzer):
        missing.append("pymatgen")
    if None in (pyxtal, Group):
        missing.append("pyxtal")
    if missing:
        raise ImportError("DiffCSP++-style symmetry backend requires: " + ", ".join(missing) + ".")


def _atomic_number(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        return int(Element(value).Z)
    if hasattr(value, "Z"):
        return int(value.Z)
    return int(Element(str(value)).Z)


def refine_spacegroup_structure(
    structure,
    *,
    symprec: float = 1e-2,
    standardization: str = "conventional",
) -> tuple[Any, int]:
    """DiffCSP++-style refinement into a canonical space-group structure."""
    _require_dependencies()
    spga = SpacegroupAnalyzer(structure, symprec=symprec)
    if standardization == "conventional":
        refined = spga.get_conventional_standard_structure()
    elif standardization == "refined":
        refined = spga.get_refined_structure()
    else:
        raise ValueError(f"Unsupported standardization={standardization!r}.")
    refined = Structure(
        lattice=Lattice.from_parameters(*refined.lattice.parameters),
        species=refined.species,
        coords=refined.frac_coords,
        coords_are_cartesian=False,
    )
    return refined, int(spga.get_space_group_number())


def build_diffcsppp_symmetry_payload(
    structure,
    *,
    tol: float = 1e-2,
) -> DiffCSPPPSymmetryPayload:
    """Mirror DiffCSP++ `get_symmetry_info(...)` as a reusable backend payload."""
    _require_dependencies()
    spga = SpacegroupAnalyzer(structure, symprec=tol)
    refined = spga.get_refined_structure()
    crystal = pyxtal()
    try:
        crystal.from_seed(refined, tol=tol)
    except Exception:
        crystal.from_seed(refined, tol=max(float(tol) * 1.0e-2, 1.0e-4))

    space_group = int(crystal.group.number)
    expanded_species: list[int] = []
    anchor_index: list[int] = []
    matrices: list[np.ndarray] = []
    coords: list[np.ndarray] = []
    anchor_coords: list[np.ndarray] = []
    anchor_atomic_numbers: list[int] = []
    anchor_dofs: list[int] = []
    anchor_free_coordinate_masks: list[np.ndarray] = []
    wyckoff_letters: list[str] = []
    running_anchor = 0
    for site in crystal.atom_sites:
        specie_z = _atomic_number(site.specie)
        coord = np.asarray(site.position, dtype=float) % 1.0
        anchor_coords.append(coord)
        anchor_atomic_numbers.append(specie_z)
        wyckoff_letters.append(str(site.wp.get_label()))
        anchor_dofs.append(int(site.wp.get_dof()))
        anchor_free_coordinate_masks.append(_free_coordinate_mask_from_wp(site.wp))
        for op in site.wp:
            expanded_species.append(specie_z)
            matrices.append(np.asarray(op.affine_matrix, dtype=float))
            coords.append(np.asarray(op.operate(site.position), dtype=float) % 1.0)
            anchor_index.append(running_anchor)
        running_anchor += len(site.wp)

    matrices_np = np.asarray(matrices, dtype=float)
    ops_inv = np.linalg.pinv(matrices_np[:, :3, :3]) if matrices_np.size else np.zeros((0, 3, 3), dtype=float)
    standardized_structure = Structure(
        lattice=Lattice.from_parameters(*np.asarray(crystal.lattice.get_para(degree=True), dtype=float)),
        species=[Element.from_Z(int(z)).symbol for z in expanded_species],
        coords=np.asarray(coords, dtype=float),
        coords_are_cartesian=False,
    )
    return DiffCSPPPSymmetryPayload(
        spacegroup=space_group,
        anchor_index=np.asarray(anchor_index, dtype=int),
        wyckoff_ops=matrices_np,
        wyckoff_ops_inv=np.asarray(ops_inv, dtype=float),
        wyckoff_letters=tuple(wyckoff_letters),
        atom_types=np.asarray(expanded_species, dtype=int),
        anchor_frac_coords=np.asarray(anchor_coords, dtype=float),
        expanded_frac_coords=np.asarray(coords, dtype=float),
        anchor_atomic_numbers=np.asarray(anchor_atomic_numbers, dtype=int),
        expanded_atomic_numbers=np.asarray(expanded_species, dtype=int),
        lattice_matrix=np.asarray(crystal.lattice.matrix, dtype=float),
        anchor_dofs=np.asarray(anchor_dofs, dtype=int),
        anchor_free_coordinate_masks=np.asarray(anchor_free_coordinate_masks, dtype=bool),
        standardized_structure=standardized_structure,
        debug_info={
            "tol": float(tol),
            "refined_spacegroup": int(spga.get_space_group_number()),
            "num_anchor_sites": int(len(wyckoff_letters)),
            "num_atoms": int(len(expanded_species)),
        },
    )


def build_diffcsppp_payload_from_syminfo(
    *,
    spacegroup_number: int,
    wyckoff_letters: list[str] | tuple[str, ...] | str,
    atom_types: list[int | str] | tuple[int | str, ...] | np.ndarray | None = None,
) -> DiffCSPPPSymmetryPayload:
    """Mirror DiffCSP++ `get_data_from_syminfo(...)` as a backend payload."""
    _require_dependencies()
    if isinstance(wyckoff_letters, str):
        if "," in wyckoff_letters:
            letters = [token.strip() for token in wyckoff_letters.split(",") if token.strip()]
        else:
            letters = [token.strip() for token in wyckoff_letters if str(token).strip()]
    else:
        letters = [str(token).strip() for token in wyckoff_letters]
    group = Group(int(spacegroup_number))
    ops_tot: list[np.ndarray] = []
    anchor_index: list[int] = []
    expanded_atomic_numbers: list[int] = []
    anchor_atomic_numbers: list[int] = []
    anchor_frac_coords: list[np.ndarray] = []
    anchor_dofs: list[int] = []
    anchor_free_coordinate_masks: list[np.ndarray] = []
    num_atoms = 0
    if atom_types is not None:
        atom_tokens = list(atom_types)
        if len(atom_tokens) != len(letters):
            raise ValueError("wyckoff_letters and atom_types must have the same length.")
    else:
        atom_tokens = [0 for _ in letters]
    for idx, label in enumerate(letters):
        letter = str(label)[-1]
        wp = group[letter]
        site_atomic_number = _atomic_number(atom_tokens[idx]) if atom_types is not None else 0
        anchor_atomic_numbers.append(site_atomic_number)
        anchor_frac_coords.append(np.zeros(3, dtype=float))
        anchor_dofs.append(int(wp.get_dof()))
        anchor_free_coordinate_masks.append(_free_coordinate_mask_from_wp(wp))
        for op in wp.ops:
            ops_tot.append(np.asarray(op.affine_matrix, dtype=float))
            anchor_index.append(num_atoms)
            expanded_atomic_numbers.append(site_atomic_number)
        num_atoms += len(wp.ops)
    ops_np = np.asarray(ops_tot, dtype=float)
    ops_inv = np.linalg.pinv(ops_np[:, :3, :3]) if ops_np.size else np.zeros((0, 3, 3), dtype=float)
    coords = np.zeros((len(expanded_atomic_numbers), 3), dtype=float)
    return DiffCSPPPSymmetryPayload(
        spacegroup=int(spacegroup_number),
        anchor_index=np.asarray(anchor_index, dtype=int),
        wyckoff_ops=ops_np,
        wyckoff_ops_inv=np.asarray(ops_inv, dtype=float),
        wyckoff_letters=tuple(letters),
        atom_types=np.asarray(expanded_atomic_numbers, dtype=int),
        anchor_frac_coords=np.asarray(anchor_frac_coords, dtype=float),
        expanded_frac_coords=coords,
        anchor_atomic_numbers=np.asarray(anchor_atomic_numbers, dtype=int),
        expanded_atomic_numbers=np.asarray(expanded_atomic_numbers, dtype=int),
        lattice_matrix=np.eye(3, dtype=float),
        anchor_dofs=np.asarray(anchor_dofs, dtype=int),
        anchor_free_coordinate_masks=np.asarray(anchor_free_coordinate_masks, dtype=bool),
        standardized_structure=None,
        debug_info={
            "source": "syminfo",
            "num_anchor_sites": int(len(letters)),
            "num_atoms": int(len(expanded_atomic_numbers)),
        },
    )


def select_template_and_free_vars_from_payload(
    *,
    payload: DiffCSPPPSymmetryPayload,
    atomic_numbers_standardized: torch.Tensor,
    max_templates: int = 64,
) -> tuple[WyckoffTemplate, torch.Tensor, list[WyckoffTemplate]]:
    templates = extract_wyckoff_templates(
        space_group_number=int(payload.spacegroup),
        atomic_numbers=atomic_numbers_standardized,
        max_templates=int(max_templates),
        quick=False,
    )
    template = None
    signature = payload.site_signature
    for candidate in templates:
        if flatten_site_signature(candidate) == signature:
            template = candidate
            break
    if template is None:
        raise RuntimeError(
            f"No template matched DiffCSP++ payload signature for SG={int(payload.spacegroup)} signature={signature!r}."
        )
    free_vars = recover_template_free_vars_from_anchor_entries(template, payload.anchor_entries)
    return template, free_vars, templates


def _match_template_sites_to_payload(
    payload: DiffCSPPPSymmetryPayload,
    template: WyckoffTemplate,
) -> tuple[tuple[Any, torch.Tensor], ...]:
    grouped_template_indices: dict[tuple[int, str], list[int]] = {}
    for template_idx, site in enumerate(template.site_templates):
        key = (int(site.atomic_number), str(site.label))
        grouped_template_indices.setdefault(key, []).append(int(template_idx))

    matches: list[tuple[Any, torch.Tensor]] = []
    for payload_site_idx, entry in enumerate(payload.anchor_entries):
        key = (int(entry["atomic_number"]), str(entry["label"]))
        if key not in grouped_template_indices or not grouped_template_indices[key]:
            raise RuntimeError(
                f"Could not match payload site {payload_site_idx} signature={key!r} to template site."
            )
        anchor = np.asarray(entry["anchor_frac"], dtype=float).reshape(3)
        best_choice = None
        best_q = None
        best_residual = float("inf")
        for candidate_pos, template_idx in enumerate(grouped_template_indices[key]):
            template_site = template.site_templates[int(template_idx)]
            basis = np.asarray(template_site.anchor_basis, dtype=float)
            offset = np.asarray(template_site.anchor_offset, dtype=float)
            delta = _signed_wrap_numpy(anchor - offset)
            if basis.shape[1] == 0:
                q_candidate = np.zeros((0,), dtype=float)
                residual = float(np.linalg.norm(delta))
            else:
                q_candidate, *_ = np.linalg.lstsq(basis, delta, rcond=None)
                recon = basis @ q_candidate
                residual = float(np.linalg.norm(delta - recon))
            if residual < best_residual:
                best_choice = (candidate_pos, template_site)
                best_q = q_candidate
                best_residual = residual
        if best_choice is None or best_q is None:
            raise RuntimeError(f"Failed to recover chart coordinates for payload site {payload_site_idx}.")
        candidate_pos, template_site = best_choice
        grouped_template_indices[key].pop(int(candidate_pos))
        matches.append(
            (
                template_site,
                torch.as_tensor(np.remainder(np.asarray(best_q, dtype=float), 1.0), dtype=torch.get_default_dtype()),
            )
        )
    return tuple(matches)


def build_wyckoff_dof_chart(
    payload: DiffCSPPPSymmetryPayload,
    *,
    atomic_numbers_standardized: torch.Tensor | None = None,
    max_templates: int = 64,
) -> WyckoffDOFChart:
    atomic_numbers = (
        torch.as_tensor(payload.expanded_atomic_numbers, dtype=torch.long)
        if atomic_numbers_standardized is None
        else torch.as_tensor(atomic_numbers_standardized, dtype=torch.long)
    )
    template, _free_vars_unused, _templates = select_template_and_free_vars_from_payload(
        payload=payload,
        atomic_numbers_standardized=atomic_numbers,
        max_templates=int(max_templates),
    )
    matched_sites = _match_template_sites_to_payload(payload, template)
    site_row_slices = tuple(payload_site_slices(payload))

    q_parts: list[np.ndarray] = []
    site_dof_slices: list[slice] = []
    site_anchor_bases: list[np.ndarray] = []
    site_anchor_offsets: list[np.ndarray] = []
    site_reference_expanded_rows: list[np.ndarray] = []
    site_dofs: list[int] = []
    cursor = 0
    for site_idx, (template_site, q_site_t) in enumerate(matched_sites):
        dof = int(template_site.dof)
        q_site = np.asarray(q_site_t.detach().cpu(), dtype=float).reshape(-1)
        if q_site.shape[0] != dof:
            raise RuntimeError(
                f"Recovered q for site {site_idx} has shape {q_site.shape}, expected dof={dof}."
            )
        q_parts.append(q_site)
        site_dof_slices.append(slice(cursor, cursor + dof))
        site_anchor_bases.append(np.asarray(template_site.anchor_basis, dtype=float))
        if dof == 0:
            site_anchor_offsets.append(np.asarray(payload.anchor_frac_coords[site_idx], dtype=float))
        else:
            site_anchor_offsets.append(np.asarray(template_site.anchor_offset, dtype=float))
        row_start, row_stop = site_row_slices[site_idx]
        site_reference_expanded_rows.append(
            np.asarray(payload.expanded_frac_coords[row_start:row_stop], dtype=float)
        )
        site_dofs.append(dof)
        cursor += dof

    q_ref = np.concatenate(q_parts, axis=0) if q_parts else np.zeros((0,), dtype=float)
    chart = WyckoffDOFChart(
        payload=payload,
        template=template,
        q_ref=np.remainder(q_ref, 1.0),
        site_dof_slices=tuple(site_dof_slices),
        site_row_slices=site_row_slices,
        site_anchor_bases=tuple(site_anchor_bases),
        site_anchor_offsets=tuple(site_anchor_offsets),
        site_reference_expanded_rows=tuple(site_reference_expanded_rows),
        site_dofs=np.asarray(site_dofs, dtype=int),
    )
    return chart


def project_payload_to_wyckoff_dof_chart(
    z_payload: torch.Tensor,
    chart: WyckoffDOFChart,
    *,
    q_ref: torch.Tensor | np.ndarray | None = None,
    lambda_q: float = 1.0e-6,
) -> dict[str, Any]:
    return chart.project_local(z_payload, q_ref=q_ref, lambda_q=float(lambda_q))


def local_wyckoff_dof_chart_loss(
    z_payload: torch.Tensor,
    chart: WyckoffDOFChart,
    *,
    q_ref: torch.Tensor | np.ndarray | None = None,
    lambda_q: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    result = chart.project_local(z_payload, q_ref=q_ref, lambda_q=float(lambda_q))
    return result["loss"], result["q_star"], result["z_proj"], result


def oracle_spacegroup_from_task(*, requested_spacegroup: int) -> int:
    """Temporary oracle CMPSL: returns the ground-truth/requested space group."""
    return int(requested_spacegroup)


def oracle_spacegroup_from_case(case: Any) -> int:
    """Temporary oracle CMPSL from a task/case object with `requested_sg`/`space_group`."""
    if hasattr(case, "requested_sg"):
        return int(getattr(case, "requested_sg"))
    if hasattr(case, "space_group"):
        return int(getattr(case, "space_group"))
    if isinstance(case, dict):
        if "requested_sg" in case:
            return int(case["requested_sg"])
        if "space_group" in case:
            return int(case["space_group"])
    raise KeyError("Could not infer oracle space group from case/task object.")
