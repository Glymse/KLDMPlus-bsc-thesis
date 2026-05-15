from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KFamilyConstraint:
    space_group: int
    mask: torch.Tensor
    target: torch.Tensor
    free_indices: tuple[int, ...]
    fixed_indices: tuple[int, ...]


def matrix_log_spd(matrix: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = eigvals.clamp_min(eps)
    log_diag = torch.diag_embed(torch.log(eigvals))
    return eigvecs @ log_diag @ eigvecs.transpose(-1, -2)


def matrix_exp_symmetric(matrix: torch.Tensor) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    exp_diag = torch.diag_embed(torch.exp(eigvals))
    return eigvecs @ exp_diag @ eigvecs.transpose(-1, -2)


def cell_to_k(cell: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    gram = cell @ cell.transpose(-1, -2)
    s_matrix = 0.5 * matrix_log_spd(gram, eps=eps)

    s00 = s_matrix[..., 0, 0]
    s11 = s_matrix[..., 1, 1]
    s22 = s_matrix[..., 2, 2]
    k1 = s_matrix[..., 0, 1]
    k2 = s_matrix[..., 0, 2]
    k3 = s_matrix[..., 1, 2]
    k4 = 0.5 * (s00 - s11)
    k5 = (s00 + s11 - 2.0 * s22) / 6.0
    k6 = (s00 + s11 + s22) / 3.0
    return torch.stack([k1, k2, k3, k4, k5, k6], dim=-1)


def k_to_s_matrix(k: torch.Tensor) -> torch.Tensor:
    k1, k2, k3, k4, k5, k6 = k.unbind(dim=-1)
    s00 = k4 + k5 + k6
    s11 = -k4 + k5 + k6
    s22 = -2.0 * k5 + k6

    matrix = torch.zeros(*k.shape[:-1], 3, 3, device=k.device, dtype=k.dtype)
    matrix[..., 0, 0] = s00
    matrix[..., 1, 1] = s11
    matrix[..., 2, 2] = s22
    matrix[..., 0, 1] = matrix[..., 1, 0] = k1
    matrix[..., 0, 2] = matrix[..., 2, 0] = k2
    matrix[..., 1, 2] = matrix[..., 2, 1] = k3
    return matrix


def k_to_cell_matrix(k: torch.Tensor) -> torch.Tensor:
    s_matrix = k_to_s_matrix(k)
    return matrix_exp_symmetric(s_matrix)


def space_group_k_constraint(
    *,
    space_group_number: int,
    device: torch.device,
    dtype: torch.dtype,
) -> KFamilyConstraint:
    if not 1 <= int(space_group_number) <= 230:
        raise ValueError(f"space_group must be in [1, 230], got {space_group_number}.")

    mask = torch.zeros(6, device=device, dtype=dtype)
    target = torch.zeros(6, device=device, dtype=dtype)

    sg = int(space_group_number)
    if 3 <= sg <= 15:
        mask[[0, 2]] = 1.0
    elif 16 <= sg <= 74:
        mask[[0, 1, 2]] = 1.0
    elif 75 <= sg <= 142:
        mask[[0, 1, 2, 3]] = 1.0
    elif 143 <= sg <= 194:
        mask[[0, 1, 2, 3]] = 1.0
        target[0] = -torch.log(torch.tensor(3.0, device=device, dtype=dtype)) / 4.0
    elif 195 <= sg <= 230:
        mask[[0, 1, 2, 3, 4]] = 1.0

    fixed_indices = tuple(int(i) for i in torch.nonzero(mask > 0, as_tuple=False).reshape(-1).tolist())
    free_indices = tuple(i for i in range(6) if i not in fixed_indices)
    return KFamilyConstraint(
        space_group=sg,
        mask=mask,
        target=target,
        free_indices=free_indices,
        fixed_indices=fixed_indices,
    )


def k_to_free_vars(k: torch.Tensor, constraint: KFamilyConstraint) -> torch.Tensor:
    if len(constraint.free_indices) == 0:
        return k.new_zeros((*k.shape[:-1], 0))
    return k[..., list(constraint.free_indices)]


def free_vars_to_k(free_vars: torch.Tensor, constraint: KFamilyConstraint) -> torch.Tensor:
    target_shape = (*free_vars.shape[:-1], 6)
    k = constraint.target.expand(target_shape).clone()
    if len(constraint.free_indices) > 0:
        k[..., list(constraint.free_indices)] = free_vars
    return k
