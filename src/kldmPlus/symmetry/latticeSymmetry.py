from __future__ import annotations

import math
from itertools import product

import torch
from torch import nn


class LatticeSymmetry(nn.Module):
    """DiffCSP++ lattice k-space helpers and soft space-group constraints."""

    def __init__(self, eps: float = 1.0e-8) -> None:
        super().__init__()
        self.eps = float(eps)

        basis = self._make_basis()
        masks, biases = self._make_spacegroup_constraints()
        unimodular_basis_candidates = self._make_unimodular_basis_candidates()

        self.register_buffer("basis", basis)
        self.register_buffer("masks", masks)
        self.register_buffer("biases", biases)
        self.register_buffer("unimodular_basis_candidates", unimodular_basis_candidates)

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

    def _make_unimodular_basis_candidates(self) -> torch.Tensor:
        candidates = []
        eye = torch.eye(3, dtype=torch.get_default_dtype())
        for vals in product((-1.0, 0.0, 1.0), repeat=9):
            matrix = torch.tensor(vals, dtype=torch.get_default_dtype()).reshape(3, 3)
            det = int(round(float(torch.linalg.det(matrix).item())))
            if abs(det) == 1:
                candidates.append(matrix)
        candidates.sort(key=lambda matrix: 0 if torch.equal(matrix, eye) else 1)
        return torch.stack(candidates, dim=0)

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

    def direct_sg_residual_abs_mean(
        self,
        vecs: torch.Tensor,
        spacegroup: torch.Tensor,
    ) -> torch.Tensor:
        proj = self.proj_k_to_spacegroup(vecs, spacegroup)
        return (vecs - proj).abs().mean(dim=1)

    def orbit_sg_residual_abs_mean(
        self,
        vecs: torch.Tensor,
        spacegroup: torch.Tensor,
        *,
        chunk_size: int = 512,
        max_candidates: int | None = None,
    ) -> torch.Tensor:
        """Return min_U mean abs SG residual for equivalent primitive bases."""
        vecs = vecs.reshape(-1, 6)
        sg = self._sanitize_spacegroup(spacegroup).to(device=vecs.device)
        cells = self.v2m(vecs).reshape(-1, 3, 3)
        batch_size = int(vecs.shape[0])
        best = vecs.new_full((batch_size,), float("inf"))

        candidates = self.unimodular_basis_candidates.to(device=vecs.device, dtype=vecs.dtype)
        if max_candidates is not None:
            candidates = candidates[: int(max_candidates)]

        for start in range(0, int(candidates.shape[0]), int(chunk_size)):
            u_chunk = candidates[start : start + int(chunk_size)]
            transformed_cells = torch.einsum("mij,bjk->mbik", u_chunk, cells)
            k_u = self.m2v(self.de_so3(transformed_cells.reshape(-1, 3, 3))).reshape(
                int(u_chunk.shape[0]),
                batch_size,
                6,
            )
            sg_u = sg.unsqueeze(0).expand(int(u_chunk.shape[0]), batch_size).reshape(-1)
            proj_u = self.proj_k_to_spacegroup(k_u.reshape(-1, 6), sg_u).reshape_as(k_u)
            residual = (k_u - proj_u).abs().mean(dim=2)
            best = torch.minimum(best, residual.min(dim=0).values)
        return best

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

    def conventional_sg_residual_abs_mean(
        self,
        primitive_k: torch.Tensor,
        conv_C: torch.Tensor,
        spacegroup: torch.Tensor,
    ) -> torch.Tensor:
        """Measure SG residual after mapping primitive k to the conventional chart."""
        primitive_k = primitive_k.reshape(-1, 6)
        transform = conv_C.reshape(-1, 3, 3).to(device=primitive_k.device, dtype=primitive_k.dtype)
        primitive_cell = self.v2m(primitive_k)
        conventional_cell = transform @ primitive_cell
        conventional_k = self.m2v(self.de_so3(conventional_cell))
        projected_k = self.proj_k_to_spacegroup(conventional_k, spacegroup)
        return (conventional_k - projected_k).abs().mean(dim=1)

    def conventional_sg_loss_per_graph(
        self,
        primitive_k0: torch.Tensor,
        conv_C: torch.Tensor,
        spacegroup: torch.Tensor,
    ) -> torch.Tensor:
        """DiffCSP++ mask penalty in the conventional chart, averaged over six k dims."""
        primitive_k0 = primitive_k0.reshape(-1, 6)
        transform = conv_C.reshape(-1, 3, 3).to(device=primitive_k0.device, dtype=primitive_k0.dtype)
        primitive_cell = self.v2m(primitive_k0)
        conventional_cell = transform @ primitive_cell
        conventional_k = self.m2v(self.de_so3(conventional_cell))
        projected_k = self.proj_k_to_spacegroup(conventional_k, spacegroup)
        return (conventional_k - projected_k).pow(2).mean(dim=1)

    def conventional_sg_loss_and_residual_per_graph(
        self,
        primitive_k0: torch.Tensor,
        conv_C: torch.Tensor,
        spacegroup: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return squared loss and abs residual from one conventional-chart pass."""
        primitive_k0 = primitive_k0.reshape(-1, 6)
        transform = conv_C.reshape(-1, 3, 3).to(device=primitive_k0.device, dtype=primitive_k0.dtype)
        primitive_cell = self.v2m(primitive_k0)
        conventional_cell = transform @ primitive_cell
        conventional_k = self.m2v(self.de_so3(conventional_cell))
        projected_k = self.proj_k_to_spacegroup(conventional_k, spacegroup)
        diff = conventional_k - projected_k
        return diff.pow(2).mean(dim=1), diff.abs().mean(dim=1)

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

__all__ = ["LatticeSymmetry"]
