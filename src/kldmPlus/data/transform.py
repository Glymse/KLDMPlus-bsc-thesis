from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import torch

from kldmPlus.symmetry.latticeSymmetry import LatticeSymmetry

try:
    from torch_geometric.utils import dense_to_sparse
except ImportError:  # pragma: no cover
    dense_to_sparse = None

try:
    from mattergen.common.data.chemgraph import ChemGraph
    from mattergen.common.data.transform import Transform
except ImportError:  # pragma: no cover
    ChemGraph = Any

    class Transform:  # type: ignore[override]
        pass

try:
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:  # pragma: no cover
    Lattice = None
    Structure = None
    SpacegroupAnalyzer = None


@contextmanager
def _suppress_native_stderr():
    """Silence noisy C-library stderr output from spglib during standardization."""
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


# Atomic vocabulary used when atom types are represented by indices.
# Here the vocabulary is simply all elements with atomic number 1 to 118.
DEFAULT_ATOMIC_VOCAB: list[int] = list(range(1, 119))


"""
Lattice preprocessing for the KLDM CSP pipeline.

Each crystal cell is converted from a 3x3 basis matrix to a 6D feature vector:

    [log(a), log(b), log(c), tan(alpha - pi/2), tan(beta - pi/2), tan(gamma - pi/2)]

where:
    - a, b, c are lattice lengths
    - alpha is the angle between b and c
    - beta  is the angle between a and c
    - gamma is the angle between a and b

Two lattice modes are supported:

1. eps lattice mode
   The 6D feature vector is used directly.

2. x0 lattice mode
   The same 6D feature vector is split into:
     - transformed lengths
     - transformed angles

   The x0 branch stores train-set statistics in a JSON cache with:

       {
         "lengths_loc_scale": {
           "<num_atoms>": [[loc_a, loc_b, loc_c], [scale_a, scale_b, scale_c]],
           ...
         },
         "angles_loc_scale": [loc, scale]
       }

   Lengths are standardized separately for each atom count, while the angle
   features share one fixed loc/scale pair.

At dataset time, ContinuousIntervalLattice.__call__() writes the transformed
feature vector to `sample.l`. During sampling or evaluation, the transform is
inverted back to physical lengths and angles with the matching x0 statistics
for the current number of atoms.
"""

def cell_lengths_and_angles(cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a 3x3 lattice matrix to lengths and angles.
            alpha = angle between b and c
            beta  = angle between a and c
            gamma = angle between a and b
    """

    lengths = torch.linalg.norm(cell, dim=1)

    alpha = torch.acos(
        torch.clamp(
            torch.dot(cell[1], cell[2]) / (lengths[1] * lengths[2]),
            -1.0,
            1.0,
        )
    )
    beta = torch.acos(
        torch.clamp(
            torch.dot(cell[0], cell[2]) / (lengths[0] * lengths[2]),
            -1.0,
            1.0,
        )
    )
    gamma = torch.acos(
        torch.clamp(
            torch.dot(cell[0], cell[1]) / (lengths[0] * lengths[1]),
            -1.0,
            1.0,
        )
    )

    return lengths, torch.stack([alpha, beta, gamma])


def lengths_angles_to_cell_matrix(
    lengths: torch.Tensor,
    angles: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    a, b, c = lengths.unbind(dim=-1)
    alpha, beta, gamma = angles.unbind(dim=-1)

    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma).clamp_min(eps)

    zeros = torch.zeros_like(a)
    row_a = torch.stack([a, zeros, zeros], dim=-1)
    row_b = torch.stack([b * cos_gamma, b * sin_gamma, zeros], dim=-1)
    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz = torch.sqrt((c.square() - cx.square() - cy.square()).clamp_min(eps))
    row_c = torch.stack([cx, cy, cz], dim=-1)
    return torch.stack([row_a, row_b, row_c], dim=-2)


def lattice_feature_vector(cell: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """

    Output:
        Tensor of shape (6,):

            [
                log(a),
                log(b),
                log(c),
                tan(alpha - pi/2),
                tan(beta  - pi/2),
                tan(gamma - pi/2),
            ]

    """
    log_lengths, angle_features = lattice_feature_components(cell, eps=eps)

    return torch.cat([log_lengths, angle_features], dim=0)


def lattice_feature_components(
    cell: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a cell into transformed length and angle features."""
    lengths, angles = cell_lengths_and_angles(cell)
    log_lengths = torch.log(lengths.clamp_min(eps))
    angle_features = torch.tan(angles - torch.pi / 2.0)
    return log_lengths, angle_features

def _has_x0_lattice_stats(payload: dict) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("lengths_loc_scale"), dict)
        and isinstance(payload.get("angles_loc_scale"), list)
        and len(payload["angles_loc_scale"]) == 2
    )

def _pack_x0_stats(
    lengths_by_num_atoms: dict[int, list[torch.Tensor]],
    *,
    eps: float,
) -> dict[str, object]:
    stats_by_size: dict[str, list[list[float]]] = {}
    for num_atoms, values in sorted(lengths_by_num_atoms.items()):
        stacked = torch.stack(values, dim=0)
        center = stacked.mean(dim=0)
        spread = stacked.std(dim=0, unbiased=False).clamp_min(eps)
        stats_by_size[str(num_atoms)] = [center.tolist(), spread.tolist()]

    return {
        "lengths_loc_scale": stats_by_size,
        "angles_loc_scale": [0.0, 0.35],
    }


def _restore_x0_stats(
    payload: dict,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]:
    ######## Code segment is from original KLDM preprocessing code. ######
    lengths_loc_scale = {
        int(num_atoms): (
            torch.tensor(center, dtype=torch.get_default_dtype()),
            torch.tensor(spread, dtype=torch.get_default_dtype()),
        )
        for num_atoms, (center, spread) in payload["lengths_loc_scale"].items()
    }
    angle_center, angle_spread = payload["angles_loc_scale"]
    angles_loc_scale = (
        torch.tensor(angle_center, dtype=torch.get_default_dtype()),
        torch.tensor(angle_spread, dtype=torch.get_default_dtype()),
    )
    return lengths_loc_scale, angles_loc_scale

def ensure_lattice_standardization_cache(
    *,
    cache_file: str | Path,
    processed_dir: str | Path,
    eps: float = 1e-8,
) -> Path:
    """Create train-set statistics for x0 lattice preprocessing."""
    cache_path = Path(cache_file)
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                existing_payload = json.load(handle)
            if _has_x0_lattice_stats(existing_payload):
                return cache_path
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    cell_path = Path(processed_dir) / "cell.npy"
    num_atoms_path = Path(processed_dir) / "num_atoms.npy"
    cells = np.load(cell_path, allow_pickle=True)
    num_atoms: Any = np.load(num_atoms_path, allow_pickle=True)

    ######## Code segment is from original KLDM preprocessing code. ######
    lengths_by_num_atoms: dict[int, list[torch.Tensor]] = {}
    for cell, n_atoms in zip(cells, num_atoms):
        cell = torch.as_tensor(cell, dtype=torch.get_default_dtype())

        # MatterGen may store a cell as shape (1, 3, 3).
        # The transform expects shape (3, 3).
        if cell.ndim == 3 and cell.shape[0] == 1:
            cell = cell.squeeze(0)

        log_lengths, _ = lattice_feature_components(cell, eps=eps)
        lengths_by_num_atoms.setdefault(int(n_atoms), []).append(log_lengths)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(_pack_x0_stats(lengths_by_num_atoms, eps=eps), handle, indent=2)

    return cache_path

class FullyConnectedGraph(Transform):
    """
    Add fully connected directed edges to a crystal graph.

    """

    def __init__(self, key: str = "edge_node_index", len_from: str = "pos") -> None:
        """Store transform configuration.

        Input:
            key:
                Name of the output edge-index field.

            len_from:
                Name of the tensor used to infer the number of atoms.
        """
        self.key = key
        self.len_from = len_from
        self._edge_cache: dict[int, torch.Tensor] = {}

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        if dense_to_sparse is None:
            raise ImportError("torch_geometric is required to build fully connected crystal graphs.")

        n = len(getattr(sample, self.len_from))

        edge_index = self._edge_cache.get(int(n))
        if edge_index is None:
            adjacency = torch.ones(n, n)
            adjacency = adjacency - torch.eye(n)
            edge_index, _ = dense_to_sparse(adjacency)
            self._edge_cache[int(n)] = edge_index.detach().cpu()
        edge_index = edge_index.to(device=sample.pos.device)

        return sample.replace(**{self.key: edge_index})


class ContinuousIntervalLattice(Transform, ABC):
    """Base class for lattice representations used by the CSP pipeline."""

    def __init__(
        self,
        out_key: str = "l",
        standardize: bool = False,
        cache_file: str | Path | None = None,
        eps: float = 1e-8,
        representation: str = "kldm",
    ) -> None:
        if representation not in {"kldm", "diffcsp_k"}:
            raise ValueError("representation must be 'kldm' or 'diffcsp_k'.")
        self.out_key = out_key
        self.standardize = standardize
        self.cache_file = Path(cache_file) if cache_file is not None else None
        self.eps = eps
        self.representation = representation

    @abstractmethod
    def __call__(self, sample: ChemGraph) -> ChemGraph:
        raise NotImplementedError

    @abstractmethod
    def invert_to_lengths_angles(
        self,
        l: torch.Tensor,
        num_atoms: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def invert_to_matrix(
        self,
        l: torch.Tensor,
        num_atoms: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        lengths, angles = self.invert_to_lengths_angles(l=l, num_atoms=num_atoms)
        return lengths_angles_to_cell_matrix(lengths=lengths, angles=angles, eps=self.eps)


class KLDMContinuousIntervalLattice(ContinuousIntervalLattice):
    """
    Encode/decode the original KLDM lattice parameterization.

    Forward transform:
        cell matrix -> 6D lattice vector

    standardization is x0:
        l_standardized = (l - loc) / scale

    Inverse transform:
        l -> unstandardize -> lengths and angles
    """

    def __init__(
        self,
        out_key: str = "l",
        standardize: bool = False,
        cache_file: str | Path | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            out_key=out_key,
            standardize=standardize,
            cache_file=cache_file,
            eps=eps,
            representation="kldm",
        )

        self.loc: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.lengths_loc_scale: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
        self.angles_loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None

        if self.standardize and self.cache_file is not None and self.cache_file.exists():
            with self.cache_file.open("r", encoding="utf-8") as handle:
                stats = json.load(handle)

            if _has_x0_lattice_stats(stats):
                self.lengths_loc_scale, self.angles_loc_scale = _restore_x0_stats(stats)
            else:
                self.loc = torch.tensor(stats["loc"], dtype=torch.get_default_dtype())
                self.scale = torch.tensor(stats["scale"], dtype=torch.get_default_dtype())

    def _move_stats_to(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        loc = self.loc.to(device=value.device, dtype=value.dtype)
        scale = self.scale.to(device=value.device, dtype=value.dtype).clamp_min(self.eps)

        # Make loc and scale broadcastable to value.
        while loc.ndim < value.ndim:
            loc = loc.unsqueeze(0)
            scale = scale.unsqueeze(0)

        return loc, scale

    def standardize_value(self, value: torch.Tensor) -> torch.Tensor:
        """
        Standardize lattice features.

        Output:
            If standardization is enabled:
                (value - loc) / scale

            Otherwise:
                value unchanged.
        """
        if not self.standardize or self.loc is None or self.scale is None:
            return value

        loc, scale = self._move_stats_to(value)
        return (value - loc) / scale

    def _move_scalar_stats_to(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loc = loc.to(device=value.device, dtype=value.dtype)
        scale = scale.to(device=value.device, dtype=value.dtype).clamp_min(self.eps)
        while loc.ndim < value.ndim:
            loc = loc.unsqueeze(0)
            scale = scale.unsqueeze(0)
        return loc, scale

    def _length_stats_for_num_atoms(
        self,
        num_atoms: int,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.lengths_loc_scale is None or num_atoms not in self.lengths_loc_scale:
            raise KeyError(f"Missing x0 length statistics for num_atoms={num_atoms}.")
        loc, scale = self.lengths_loc_scale[num_atoms]
        return self._move_scalar_stats_to(loc, scale, value)

    def _encode_x0_parts(
        self,
        *,
        log_lengths: torch.Tensor,
        angle_features: torch.Tensor,
        num_atoms: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ######## Code segment is from original KLDM preprocessing code. ######
        loc_lengths, scale_lengths = self._length_stats_for_num_atoms(num_atoms, log_lengths)
        log_lengths = (log_lengths - loc_lengths) / scale_lengths

        if self.angles_loc_scale is not None:
            angle_loc, angle_scale = self.angles_loc_scale
            angle_loc, angle_scale = self._move_scalar_stats_to(angle_loc, angle_scale, angle_features)
            angle_features = (angle_features - angle_loc) / angle_scale

        return log_lengths, angle_features

    def unstandardize(self, value: torch.Tensor) -> torch.Tensor:
        """Undo lattice standardization.

        Code is heavily inspired by original KLDM.
        """
        if not self.standardize or self.loc is None or self.scale is None:
            return value

        loc, scale = self._move_stats_to(value)
        return value * scale + loc

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        """Encode a sample's cell matrix into `sample.l`.

        Input:
            sample:
                ChemGraph containing `cell`.

        Output:
            ChemGraph with added lattice tensor:
                l shape = (1, 6)
        """
        cell = sample.cell.squeeze(0)

        log_lengths, angle_features = lattice_feature_components(cell, eps=self.eps)
        if self.standardize and self.lengths_loc_scale is not None:
            ######## Code segment is from original KLDM preprocessing code. ######
            log_lengths, angle_features = self._encode_x0_parts(
                log_lengths=log_lengths,
                angle_features=angle_features,
                num_atoms=int(len(sample.pos)),
            )
            features = torch.cat([log_lengths, angle_features], dim=0)
        else:
            features = torch.cat([log_lengths, angle_features], dim=0)
            features = self.standardize_value(features)

        return sample.replace(**{self.out_key: features.view(1, 6)})

    def invert_to_lengths_angles(
        self,
        l: torch.Tensor,
        num_atoms: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode lattice vector into physical lengths and angles.

        Input:
            l:
                Tensor with last dimension 6.
                May be standardized or unstandardized depending on the transform.
        Output:
            lengths:
                Tensor with last dimension 3.

            angles:
                Tensor with last dimension 3, in radians.

        Inverse map:
            lengths = exp(l[..., :3])
            angles  = atan(l[..., 3:]) + pi/2
        """
        if self.standardize and self.lengths_loc_scale is not None:
            if num_atoms is None:
                raise ValueError("num_atoms is required to invert x0-standardized lattice features.")

            ######## Code segment is from original KLDM preprocessing code. ######
            flat_features = l.reshape(-1, 6)
            if isinstance(num_atoms, torch.Tensor):
                flat_num_atoms = num_atoms.reshape(-1).detach().cpu().tolist()
            elif isinstance(num_atoms, int):
                flat_num_atoms = [num_atoms] * flat_features.shape[0]
            else:
                flat_num_atoms = list(num_atoms)

            if len(flat_num_atoms) != flat_features.shape[0]:
                raise ValueError("num_atoms must match the batch size of lattice features.")

            log_lengths = flat_features[:, :3].clone()
            angle_features = flat_features[:, 3:].clone()

            restored_lengths = []
            for row_idx, n_atoms in enumerate(flat_num_atoms):
                loc_lengths, scale_lengths = self._length_stats_for_num_atoms(int(n_atoms), log_lengths[row_idx])
                restored_lengths.append(log_lengths[row_idx] * scale_lengths + loc_lengths)
            log_lengths = torch.stack(restored_lengths, dim=0)

            if self.angles_loc_scale is not None:
                angle_loc, angle_scale = self.angles_loc_scale
                angle_loc, angle_scale = self._move_scalar_stats_to(angle_loc, angle_scale, angle_features)
                angle_features = angle_features * angle_scale + angle_loc

            log_lengths = log_lengths.reshape(*l.shape[:-1], 3)
            angle_features = angle_features.reshape(*l.shape[:-1], 3)
        else:
            features = self.unstandardize(l)
            log_lengths = features[..., :3]
            angle_features = features[..., 3:]

        lengths = torch.exp(log_lengths)
        angles = torch.atan(angle_features) + torch.pi / 2.0

        return lengths, angles


class DiffCSPKContinuousIntervalLattice(ContinuousIntervalLattice):
    """Encode/decode DiffCSP++ invariant lattice k-vectors."""

    def __init__(
        self,
        out_key: str = "l",
        standardize: bool = False,
        cache_file: str | Path | None = None,
        eps: float = 1e-8,
    ) -> None:
        if standardize:
            raise ValueError("diffcsp_k stores raw k values; old KLDM x0 standardization is disabled.")
        if cache_file is not None:
            raise ValueError("diffcsp_k does not use the old KLDM lattice statistics cache.")
        super().__init__(
            out_key=out_key,
            standardize=False,
            cache_file=None,
            eps=eps,
            representation="diffcsp_k",
        )
        self.lattice_symmetry = LatticeSymmetry(eps=eps)
        self._feature_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _sample_debug_id(sample: ChemGraph) -> str:
        for key in ("material_id", "structure_id", "id"):
            if hasattr(sample, key):
                value = getattr(sample, key)
                if isinstance(value, torch.Tensor):
                    value = value.reshape(-1)[0].item() if value.numel() else "empty"
                return f"{key}={value}"
        return "sample_id=unknown"

    @staticmethod
    def _sample_cache_key(sample: ChemGraph) -> str | None:
        for key in ("material_id", "structure_id", "id"):
            if hasattr(sample, key):
                value = getattr(sample, key)
                if isinstance(value, torch.Tensor):
                    value = value.reshape(-1)[0].item() if value.numel() else "empty"
                return f"{key}={value}"
        try:
            digest = hashlib.blake2b(digest_size=16)
            for value in (sample.cell, sample.pos, sample.atomic_numbers):
                tensor = torch.as_tensor(value).detach().cpu().contiguous()
                digest.update(str(tuple(tensor.shape)).encode("utf-8"))
                digest.update(str(tensor.dtype).encode("utf-8"))
                digest.update(tensor.numpy().tobytes())
            return f"content={digest.hexdigest()}"
        except Exception:
            return None

    def _warn_missing_conventional_chart(self, sample: ChemGraph, reason: str) -> None:
        print(
            "conv_sg_aux_warning=missing_conventional_chart "
            f"{self._sample_debug_id(sample)} reason={reason}",
            flush=True,
        )

    def _conventional_chart_transform(
        self,
        sample: ChemGraph,
        raw_cell: torch.Tensor,
        primitive_chart_cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute row-convention C with L_conventional ~= C @ L_primitive."""
        eye = torch.eye(3, device=raw_cell.device, dtype=raw_cell.dtype)
        zero = torch.zeros((), device=raw_cell.device, dtype=raw_cell.dtype)
        if Lattice is None or Structure is None or SpacegroupAnalyzer is None:
            self._warn_missing_conventional_chart(sample, "pymatgen_unavailable")
            return eye, zero, torch.full((), float("inf"), device=raw_cell.device, dtype=raw_cell.dtype)

        try:
            species = torch.as_tensor(sample.atomic_numbers).reshape(-1).detach().cpu().tolist()
            frac = torch.as_tensor(sample.pos).detach().cpu().numpy()
            cell_np = raw_cell.detach().cpu().numpy()
            with warnings.catch_warnings(), _suppress_native_stderr():
                warnings.filterwarnings("ignore", message="No Pauling electronegativity.*")
                structure = Structure(
                    Lattice(cell_np),
                    species,
                    frac,
                    coords_are_cartesian=False,
                )
                conventional = SpacegroupAnalyzer(
                    structure,
                    symprec=0.1,
                    angle_tolerance=5.0,
                ).get_conventional_standard_structure()
            conv_cell = torch.as_tensor(
                np.array(conventional.lattice.matrix, copy=True),
                device=raw_cell.device,
                dtype=raw_cell.dtype,
            )
            transform = conv_cell @ torch.linalg.pinv(primitive_chart_cell)
            fit_error = (transform @ primitive_chart_cell - conv_cell).abs().max()
            weight = (fit_error <= 1.0e-4).to(dtype=raw_cell.dtype)
            if float(weight.detach().cpu().item()) <= 0.0:
                self._warn_missing_conventional_chart(
                    sample,
                    f"fit_error={float(fit_error.detach().cpu().item()):.6g}",
                )
            return transform, weight, fit_error
        except Exception as exc:
            self._warn_missing_conventional_chart(sample, f"{type(exc).__name__}:{exc}")
            return eye, zero, torch.full((), float("inf"), device=raw_cell.device, dtype=raw_cell.dtype)

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        cell = sample.cell.squeeze(0)
        cache_key = self._sample_cache_key(sample)
        cached = self._feature_cache.get(cache_key) if cache_key is not None else None
        if cached is not None:
            k, conv_C, conv_weight, conv_fit_error = cached
            return sample.replace(
                **{
                    self.out_key: k.to(device=cell.device, dtype=cell.dtype).view(1, 6),
                    "conv_C": conv_C.to(device=cell.device, dtype=cell.dtype).view(1, 3, 3),
                    "conv_weight": conv_weight.to(device=cell.device, dtype=cell.dtype).view(1),
                    "conv_fit_error": conv_fit_error.to(device=cell.device, dtype=cell.dtype).view(1),
                }
            )
        with torch.no_grad():
            cell_batch = cell.reshape(1, 3, 3)
            primitive_chart_cell = self.lattice_symmetry.de_so3(cell_batch).squeeze(0)
            k = self.lattice_symmetry.m2v(primitive_chart_cell.reshape(1, 3, 3)).squeeze(0)
            conv_C, conv_weight, conv_fit_error = self._conventional_chart_transform(
                sample,
                raw_cell=cell,
                primitive_chart_cell=primitive_chart_cell,
            )
        if cache_key is not None:
            self._feature_cache[cache_key] = (
                k.detach().cpu(),
                conv_C.detach().cpu(),
                conv_weight.detach().cpu(),
                conv_fit_error.detach().cpu(),
            )
        return sample.replace(
            **{
                self.out_key: k.view(1, 6),
                "conv_C": conv_C.view(1, 3, 3),
                "conv_weight": conv_weight.view(1),
                "conv_fit_error": conv_fit_error.view(1),
            }
        )

    def invert_to_matrix(
        self,
        l: torch.Tensor,
        num_atoms: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        del num_atoms
        original_shape = l.shape[:-1]
        flat_l = l.reshape(-1, 6)
        return self.lattice_symmetry.v2m(flat_l).reshape(*original_shape, 3, 3)

    def invert_to_lengths_angles(
        self,
        l: torch.Tensor,
        num_atoms: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = self.invert_to_matrix(l=l, num_atoms=num_atoms)
        flat = matrix.reshape(-1, 3, 3)
        lengths = []
        angles = []
        for cell in flat:
            cell_lengths, cell_angles = cell_lengths_and_angles(cell)
            lengths.append(cell_lengths)
            angles.append(cell_angles)
        return (
            torch.stack(lengths, dim=0).reshape(*l.shape[:-1], 3),
            torch.stack(angles, dim=0).reshape(*l.shape[:-1], 3),
        )
