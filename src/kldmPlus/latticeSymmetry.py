from __future__ import annotations

import math

import torch
from torch import nn


class LatticeSymmetry(nn.Module):
    """DiffCSP++ lattice k-space helpers and soft space-group constraints."""

    def __init__(self, eps: float = 1.0e-8) -> None:
        super().__init__()
        self.eps = float(eps)

        basis = self._make_basis()
        masks, biases = self._make_spacegroup_constraints()

        self.register_buffer("basis", basis)
        self.register_buffer("masks", masks)
        self.register_buffer("biases", biases)

    def _make_basis(self) -> torch.Tensor:
        basis = torch.tensor(
            [
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=torch.get_default_dtype(),
        )
        return basis / basis.norm(dim=(-1, -2), keepdim=True).clamp_min(self.eps)

    def _constraint_for_spacegroup(self, sg: int) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.ones(6, dtype=torch.get_default_dtype())
        bias = torch.zeros(6, dtype=torch.get_default_dtype())

        if 195 <= int(sg) <= 230:
            mask[[0, 1, 2, 3, 4]] = 0.0
        elif 143 <= int(sg) <= 194:
            mask[[0, 1, 2, 3]] = 0.0
            bias[0] = -0.25 * math.log(3.0) * math.sqrt(2.0)
        elif 75 <= int(sg) <= 142:
            mask[[0, 1, 2, 3]] = 0.0
        elif 16 <= int(sg) <= 74:
            mask[[0, 1, 2]] = 0.0
        elif 3 <= int(sg) <= 15:
            mask[[0, 2]] = 0.0

        return mask, bias

    def _make_spacegroup_constraints(self) -> tuple[torch.Tensor, torch.Tensor]:
        masks = []
        biases = []
        for sg in range(231):
            mask, bias = self._constraint_for_spacegroup(sg)
            masks.append(mask[None, :])
            biases.append(bias[None, :])
        return torch.cat(masks, dim=0), torch.cat(biases, dim=0)

    @staticmethod
    def _sanitize_spacegroup(spacegroup: torch.Tensor) -> torch.Tensor:
        return spacegroup.reshape(-1).long().clamp(min=0, max=230)

    def de_so3(self, lattices: torch.Tensor) -> torch.Tensor:
        gram = lattices @ lattices.transpose(-1, -2)
        return self.sqrtm_spd(gram)

    def m2v(self, mats: torch.Tensor) -> torch.Tensor:
        log_mat = self.logm_spd(mats)
        basis = self.basis.to(device=mats.device, dtype=mats.dtype)
        return torch.einsum("bij,kij->bk", log_mat, basis)

    def v2m(self, vecs: torch.Tensor) -> torch.Tensor:
        basis = self.basis.to(device=vecs.device, dtype=vecs.dtype)
        log_mat = torch.einsum("bk,kij->bij", vecs, basis)
        return torch.matrix_exp(log_mat)

    def proj_k_to_spacegroup(self, vecs: torch.Tensor, spacegroup: torch.Tensor) -> torch.Tensor:
        mask, bias = self.mask_bias(spacegroup, vecs)
        return vecs * mask + bias

    def mask_bias(self, spacegroup: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sg = self._sanitize_spacegroup(spacegroup).to(device=ref.device)
        masks = self.masks[sg].to(device=ref.device, dtype=ref.dtype)
        biases = self.biases[sg].to(device=ref.device, dtype=ref.dtype)
        return masks, biases

    def soft_lattice_sg_loss_per_graph(
        self,
        pred_k0: torch.Tensor,
        spacegroup: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        mask, bias = self.mask_bias(spacegroup, pred_k0)
        constrained = 1.0 - mask
        sq = (constrained * (pred_k0 - bias)).pow(2)
        if normalize:
            return sq.sum(dim=1) / constrained.sum(dim=1).clamp_min(1.0)
        return sq.mean(dim=1)

    def soft_lattice_sg_loss(
        self,
        pred_k0: torch.Tensor,
        spacegroup: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        return self.soft_lattice_sg_loss_per_graph(
            pred_k0=pred_k0,
            spacegroup=spacegroup,
            normalize=normalize,
        ).mean()

    def logm_spd(self, mats: torch.Tensor) -> torch.Tensor:
        mats = 0.5 * (mats + mats.transpose(-1, -2))
        evals, evecs = torch.linalg.eigh(mats)
        evals = evals.clamp_min(self.eps)
        log_evals = evals.log()
        return (evecs * log_evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)

    def sqrtm_spd(self, mats: torch.Tensor) -> torch.Tensor:
        mats = 0.5 * (mats + mats.transpose(-1, -2))
        evals, evecs = torch.linalg.eigh(mats)
        evals = evals.clamp_min(self.eps)
        sqrt_evals = evals.sqrt()
        return (evecs * sqrt_evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)
