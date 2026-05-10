from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

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

def mattergen_symmetric_cell(cell: torch.Tensor) -> torch.Tensor:
    """Return the MatterGen-style rotation-fixed symmetric cell.

    Adapted from:
        src/mattergen/mattergen-main/mattergen/common/utils/data_utils.py
        ::compute_lattice_polar_decomposition

    KLDM-specific note:
    we keep the official MatterGen polar/SVD construction, but expose it as a
    small helper that accepts either one 3x3 matrix or a batch of them. The
    returned matrix is the symmetric lattice equivalent up to rotation.
    """
    # Code segment inspired from mattergen
    # (mattergen/common/utils/data_utils.py:373-386).
    w, singular_values, v_transpose = torch.linalg.svd(cell)
    s_square = torch.diag_embed(singular_values)
    v = v_transpose.transpose(-1, -2)
    u = w @ v_transpose
    p = v @ s_square @ v_transpose
    p_prime = u @ p @ u.transpose(-1, -2)
    return p_prime


def symmetric_matrix_to_vector(matrix: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            matrix[..., 0, 0],
            matrix[..., 1, 1],
            matrix[..., 2, 2],
            matrix[..., 0, 1],
            matrix[..., 0, 2],
            matrix[..., 1, 2],
        ],
        dim=-1,
    )


def vector_to_symmetric_matrix(vector: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros(*vector.shape[:-1], 3, 3, device=vector.device, dtype=vector.dtype)
    matrix[..., 0, 0] = vector[..., 0]
    matrix[..., 1, 1] = vector[..., 1]
    matrix[..., 2, 2] = vector[..., 2]
    matrix[..., 0, 1] = matrix[..., 1, 0] = vector[..., 3]
    matrix[..., 0, 2] = matrix[..., 2, 0] = vector[..., 4]
    matrix[..., 1, 2] = matrix[..., 2, 1] = vector[..., 5]
    return matrix


def mattergen_lattice_feature_vector(cell: torch.Tensor) -> torch.Tensor:
    # Code segment inspired from mattergen
    # (mattergen/common/data/transform.py:22-23,
    #  mattergen/common/utils/data_utils.py:373-386).
    # KLDM-specific adapter:
    # official MatterGen diffuses a symmetric 3x3 matrix, while the KLDM port
    # stores the six unique entries as a 6D vector.
    return symmetric_matrix_to_vector(mattergen_symmetric_cell(cell))


def lattice_spd_stats(l6: torch.Tensor) -> dict[str, float]:
    """Summarize whether symmetric 6D lattice vectors decode to valid SPD cells."""
    matrices = vector_to_symmetric_matrix(l6)
    eigvals = torch.linalg.eigvalsh(matrices)
    min_per_graph = eigvals.min(dim=-1).values
    determinants = torch.det(matrices)
    return {
        "min_eig": float(min_per_graph.min().item()),
        "frac_non_spd": float((min_per_graph <= 0).to(torch.float32).mean().item()),
        "frac_small_det": float((determinants.abs() < 0.1).to(torch.float32).mean().item()),
    }


def _has_x0_lattice_stats(payload: dict) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("lengths_loc_scale"), dict)
        and isinstance(payload.get("angles_loc_scale"), list)
        and len(payload["angles_loc_scale"]) == 2
    )


def _has_mattergen_lattice_stats(payload: dict) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("representation") == "mattergen"
        and "average_density" in payload
        and "c" in payload
        and "limit_var_scaling_constant" in payload
        and "nu" in payload
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


def _pack_mattergen_stats(
    *,
    average_density: float,
    c: float,
    limit_var_scaling_constant: float,
    nu: float,
    eps: float,
) -> dict[str, object]:
    return {
        "representation": "mattergen",
        "average_density": float(max(average_density, eps)),
        "c": float(max(c, eps)),
        "limit_var_scaling_constant": float(max(limit_var_scaling_constant, eps)),
        "nu": float(max(nu, eps)),
        "vec6_order": ["00", "11", "22", "01", "02", "12"],
        # KLDM-specific cache note:
        # the public MatterGen repo does not expose this exact JSON cache format.
        # We keep a tiny cache here so the KLDM runner can inject c/nu into the
        # lattice diffusion config without recomputing train-set statistics.
        "note": (
            "MatterGen-style lattice prior statistics for the KLDM port. "
            "Official MatterGen stores limit_density and "
            "limit_var_scaling_constant in config rather than a JSON cache. "
            "Here we cache average_density, plus the KLDM adapter values "
            "c = 1 / average_density and nu = limit_var_scaling_constant^(3/2)."
        ),
    }


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


def ensure_mattergen_lattice_cache(
    *,
    cache_file: str | Path,
    processed_dir: str | Path,
    limit_var_scaling_constant: float = 0.25,
    eps: float = 1e-8,
) -> Path:
    """Create train-set statistics for the MatterGen-style lattice prior.

    Official MatterGen uses a lattice prior with

        mu(n)    = (n / average_density)^(1/3)
        Var(n)   = n^(2/3) * limit_var_scaling_constant

    as implemented in
        src/mattergen/mattergen-main/mattergen/common/diffusion/corruption.py
        ::LatticeVPSDE.get_limit_mean
        ::LatticeVPSDE.get_limit_var

    The KLDM port keeps the existing `mu(n), sigma(n)` interface, so we cache

        c  = 1 / average_density
        nu = limit_var_scaling_constant^(3/2)

    which makes

        mu(n)    = (n c)^(1/3)
        sigma(n) = (n nu)^(1/3)

    exactly match the official MatterGen limit mean and variance.
    """
    cache_path = Path(cache_file)
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                existing_payload = json.load(handle)
            cached_scaling = existing_payload.get("limit_var_scaling_constant")
            requested_scaling = float(max(limit_var_scaling_constant, eps))
            if (
                _has_mattergen_lattice_stats(existing_payload)
                and isinstance(cached_scaling, (int, float))
                and abs(float(cached_scaling) - requested_scaling) <= max(eps, 1e-12)
            ):
                return cache_path
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    cell_path = Path(processed_dir) / "cell.npy"
    num_atoms_path = Path(processed_dir) / "num_atoms.npy"
    cells = np.load(cell_path, allow_pickle=True)
    num_atoms: Any = np.load(num_atoms_path, allow_pickle=True)

    densities: list[torch.Tensor] = []
    for cell, n_atoms in zip(cells, num_atoms):
        cell = torch.as_tensor(cell, dtype=torch.get_default_dtype())
        if cell.ndim == 3 and cell.shape[0] == 1:
            cell = cell.squeeze(0)

        # Code segment inspired from mattergen
        # (mattergen/common/data/transform.py:22-23,
        #  mattergen/common/utils/data_utils.py:373-386).
        sym_cell = mattergen_symmetric_cell(cell)
        volume = torch.det(sym_cell).abs().clamp_min(eps)
        densities.append(torch.as_tensor(float(int(n_atoms)), dtype=volume.dtype) / volume)

    # Code segment inspired from mattergen
    # (mattergen/conf/data_module/mp_20.yaml:19,
    #  mattergen/common/diffusion/corruption.py:48-65,
    #  mattergen/common/diffusion/corruption.py:110-152).
    average_density = torch.stack(densities).mean().clamp_min(eps)
    c = average_density.reciprocal()
    limit_var_scaling_constant_t = torch.as_tensor(
        float(limit_var_scaling_constant),
        dtype=torch.get_default_dtype(),
    ).clamp_min(eps)
    nu = limit_var_scaling_constant_t.pow(1.5)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _pack_mattergen_stats(
                average_density=float(average_density.item()),
                c=float(c.item()),
                limit_var_scaling_constant=float(limit_var_scaling_constant_t.item()),
                nu=float(nu.item()),
                eps=eps,
            ),
            handle,
            indent=2,
        )

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

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        if dense_to_sparse is None:
            raise ImportError("torch_geometric is required to build fully connected crystal graphs.")

        n = len(getattr(sample, self.len_from))

        adjacency = torch.ones(n, n, device=sample.pos.device)
        adjacency = adjacency - torch.eye(n, device=sample.pos.device)

        edge_index, _ = dense_to_sparse(adjacency)

        return sample.replace(**{self.key: edge_index})


class ContinuousIntervalLattice(Transform, ABC):
    """Base class for lattice representations used by the CSP pipeline."""

    def __init__(
        self,
        out_key: str = "l",
        standardize: bool = False,
        cache_file: str | Path | None = None,
        eps: float = 1e-8,
    ) -> None:
        self.out_key = out_key
        self.standardize = standardize
        self.cache_file = Path(cache_file) if cache_file is not None else None
        self.eps = eps

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
        )
        self.representation = "kldm"

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


class MatterGenContinuousIntervalLattice(ContinuousIntervalLattice):
    """Encode/decode the MatterGen-style symmetric lattice representation."""

    def __init__(
        self,
        out_key: str = "l",
        standardize: bool = False,
        cache_file: str | Path | None = None,
        limit_var_scaling_constant: float | None = None,
        eps: float = 1e-8,
    ) -> None:
        if standardize:
            raise ValueError("MatterGen lattice representation only supports eps parameterization.")
        super().__init__(
            out_key=out_key,
            standardize=standardize,
            cache_file=cache_file,
            eps=eps,
        )
        self.representation = "mattergen"
        self.average_density: float | None = None
        self.c: float | None = None
        self.limit_var_scaling_constant: float | None = None
        self.nu: float | None = None
        if self.cache_file is not None and self.cache_file.exists():
            with self.cache_file.open("r", encoding="utf-8") as handle:
                stats = json.load(handle)
            if _has_mattergen_lattice_stats(stats):
                self.average_density = float(stats["average_density"])
                self.c = float(stats["c"])
                self.limit_var_scaling_constant = float(stats["limit_var_scaling_constant"])
                self.nu = float(stats["nu"])
        if limit_var_scaling_constant is not None:
            scaling = float(max(limit_var_scaling_constant, eps))
            self.limit_var_scaling_constant = scaling
            self.nu = scaling ** 1.5

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        cell = sample.cell.squeeze(0)
        # KLDM-specific preprocessing note:
        # official MatterGen applies Niggli reduction before the polar/SVD step,
        # but it does so on the full structure during dataset preprocessing, not
        # on the cell matrix in isolation. Applying Niggli here would be wrong
        # unless fractional coordinates were transformed with the same basis
        # change. We therefore rely on the processed MatterGen-style caches
        # already containing primitive + Niggli-reduced cells upstream.
        # Code segment inspired from mattergen
        # (mattergen/common/data/transform.py:22-23,
        #  mattergen/common/utils/data_utils.py:373-386).
        # MatterGen-style symmetric lattice representation used by the KLDM port.
        # The polar factor preserves the lattice metric, so keeping fractional
        # coordinates represents the same periodic crystal up to global rotation.
        features = mattergen_lattice_feature_vector(cell)
        return sample.replace(**{self.out_key: features.view(1, 6)})

    def stats(self) -> tuple[float, float]:
        if self.c is None or self.nu is None:
            raise ValueError("MatterGen lattice stats are unavailable.")
        return self.c, self.nu

    def invert_to_matrix(
        self,
        l: torch.Tensor,
        num_atoms: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        del num_atoms
        flat_features = l.reshape(-1, 6)
        matrices = vector_to_symmetric_matrix(flat_features)
        return matrices.reshape(*l.shape[:-1], 3, 3)

    def invert_to_lengths_angles(
        self,
        l: torch.Tensor,
        num_atoms: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cells = self.invert_to_matrix(l, num_atoms=num_atoms).reshape(-1, 3, 3)

        lengths = []
        angles = []
        for cell in cells:
            cell_lengths, cell_angles = cell_lengths_and_angles(cell)
            lengths.append(cell_lengths)
            angles.append(cell_angles)

        lengths_tensor = torch.stack(lengths, dim=0).reshape(*l.shape[:-1], 3)
        angles_tensor = torch.stack(angles, dim=0).reshape(*l.shape[:-1], 3)
        return lengths_tensor, angles_tensor
