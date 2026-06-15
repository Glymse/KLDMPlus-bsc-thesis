# We use ruff
# To format our code!!!
# Remember to write this in paper if relevant.

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, Batch

from kldmPlus.data.transform import ContinuousIntervalLattice
from kldmPlus.diffusionModels.continuous import (
    ContinuousDiffusion,
    ContinuousVPDiffusion,
)
from kldmPlus.diffusionModels.tdm import TrivialisedDiffusion as TDM
from kldmPlus.scoreNetwork.scoreNetwork import CSPVNet
from kldmPlus.symmetry.latticeSymmetry import LatticeSymmetry
from kldmPlus.utils.device import get_default_device
from kldmPlus.utils.time import BatchTimes, iter_sampling_times, make_times, sampling_grid


def _lengths_angles_to_cell_matrix(
    lengths: torch.Tensor,
    angles: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build a 3x3 lattice matrix from lengths and angles."""
    a, b, c = lengths.unbind(dim=-1)
    alpha, beta, gamma = angles.unbind(dim=-1)

    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma).clamp_min(eps)

    zeros = torch.zeros_like(a)
    ax = a
    bx = b * cos_gamma
    by = b * sin_gamma
    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz_sq = (c.square() - cx.square() - cy.square()).clamp_min(eps)
    cz = torch.sqrt(cz_sq)

    row_a = torch.stack([ax, zeros, zeros], dim=-1)
    row_b = torch.stack([bx, by, zeros], dim=-1)
    row_c = torch.stack([cx, cy, cz], dim=-1)
    return torch.stack([row_a, row_b, row_c], dim=-2)

def _decode_lattice_matrix(
    *,
    l: torch.Tensor,
    num_atoms: int,
    lattice_transform: ContinuousIntervalLattice | None,
) -> torch.Tensor:
    """Decode one or more 6D lattice states into 3x3 cell matrices."""
    if lattice_transform is not None and hasattr(lattice_transform, "invert_to_matrix"):
        matrix = lattice_transform.invert_to_matrix(l=l, num_atoms=num_atoms)
        return matrix.reshape(*l.shape[:-1], 3, 3)

    if lattice_transform is not None:
        lengths, angles = lattice_transform.invert_to_lengths_angles(l=l, num_atoms=num_atoms)
    else:
        lengths = torch.exp(l[..., :3])
        angles = torch.atan(l[..., 3:]) + torch.pi / 2.0

    return _lengths_angles_to_cell_matrix(lengths=lengths, angles=angles)



class ModelKLDM(nn.Module):
    """
    KLDM model

    """

    def __init__(
        self,
        device: torch.device | None = None,
        eps: float = 1e-6,
        wrapped_normal_K: int = 3,
        tdm_n_sigmas: int | None = None,
        tdm_compute_sigma_norm: bool = True,
        tdm_velocity_scale: float | None = None,
        tdm_sigma_norm_estimator: str = "quadrature",
        tdm_sigma_norm_density_K: int | None = None,
        tdm_sigma_norm_grid_points: int = 8193,
        tdm_sigma_norm_mc_samples: int = 20000,
        lattice_parameterization: str = "eps",
        lattice_diffusion_type: str = "VP",
        lattice_representation: str = "kldm",
        lambda_l: float = 1.0,
        lattice_sg_lambda: float = 0.0,
        lattice_sg_normalize: bool = True,
        lattice_sg_time_weight: str = "quadratic_late",
        lambda_conv_sg: float = 0.0,
        conv_sg_time_weight: str = "alpha_squared",
        conv_sg_require_valid_transform: bool = True,
        conv_sg_control_mode: str = "none",
        lattice_debug: bool = False,
        lattice_orbit_metric_max_candidates: int | None = 512,
        *,
        score_network_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")

        #Load network from our config.
        self.score_network_kwargs = dict(score_network_kwargs)
        self.score_network = CSPVNet(**self.score_network_kwargs)

        self.tdm = TDM(
            eps=eps,
            wrapped_normal_K=wrapped_normal_K,
            n_sigmas=(2000 if self.device.type == "cuda" else 512) if tdm_n_sigmas is None else int(tdm_n_sigmas),
            compute_sigma_norm=tdm_compute_sigma_norm,
            velocity_scale=tdm_velocity_scale,
            sigma_norm_estimator=tdm_sigma_norm_estimator,
            sigma_norm_density_K=tdm_sigma_norm_density_K,
            sigma_norm_grid_points=tdm_sigma_norm_grid_points,
            sigma_norm_mc_samples=tdm_sigma_norm_mc_samples,
        )
        self.diffusion_l = self._build_lattice_diffusion(
            lattice_diffusion_type=lattice_diffusion_type,
            eps=eps,
            lattice_parameterization=lattice_parameterization,
        )
        self.eps = eps
        self.lattice_parameterization = lattice_parameterization
        self.lattice_diffusion_type = lattice_diffusion_type
        self.lattice_representation = lattice_representation
        self.lambda_l = float(lambda_l)
        self.lattice_sg_lambda = float(lattice_sg_lambda)
        self.lattice_sg_normalize = bool(lattice_sg_normalize)
        self.lattice_sg_time_weight = str(lattice_sg_time_weight)
        self.lambda_conv_sg = float(lambda_conv_sg)
        self.conv_sg_time_weight = "alpha_squared" if str(conv_sg_time_weight) == "alpha2" else str(conv_sg_time_weight)
        self.conv_sg_require_valid_transform = bool(conv_sg_require_valid_transform)
        self.conv_sg_control_mode = str(conv_sg_control_mode)
        self.lattice_debug = bool(lattice_debug)
        self.lattice_orbit_metric_max_candidates = lattice_orbit_metric_max_candidates
        self.lattice_symmetry = LatticeSymmetry(eps=eps)

        if self.lattice_representation not in {"kldm", "diffcsp_k"}:
            raise ValueError("lattice_representation must be 'kldm' or 'diffcsp_k'.")
        if self.lattice_representation == "diffcsp_k" and self.lattice_parameterization != "x0":
            raise ValueError("lattice_representation='diffcsp_k' requires lattice_parameterization='x0'.")
        if self.lattice_sg_lambda > 0.0 and self.lattice_representation != "diffcsp_k":
            raise ValueError("lattice_sg_lambda > 0 requires lattice_representation='diffcsp_k'.")
        if self.lambda_conv_sg > 0.0 and self.lattice_representation != "diffcsp_k":
            raise ValueError("lambda_conv_sg > 0 requires lattice_representation='diffcsp_k'.")
        if self.lattice_sg_time_weight not in {"none", "quadratic_late", "alpha_squared"}:
            raise ValueError("lattice_sg_time_weight must be 'none', 'quadratic_late', or 'alpha_squared'.")
        if self.conv_sg_time_weight not in {"none", "quadratic_late", "alpha_squared"}:
            raise ValueError("conv_sg_time_weight must be 'none', 'quadratic_late', 'alpha2', or 'alpha_squared'.")
        if self.conv_sg_control_mode not in {"none", "shuffle_batch"}:
            raise ValueError("conv_sg_control_mode must be 'none' or 'shuffle_batch'.")

    def _lattice_sg_time_weight_values(self, t_lattice: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        t_graph = t_lattice.reshape(-1).to(dtype=torch.get_default_dtype())
        t_graph = t_graph.clamp(0.0, 1.0)
        weight_mode = self.lattice_sg_time_weight if mode is None else mode
        if weight_mode == "alpha2":
            weight_mode = "alpha_squared"
        if weight_mode == "none":
            return torch.ones_like(t_graph)
        if weight_mode == "quadratic_late":
            return (1.0 - t_graph).square()
        if weight_mode == "alpha_squared":
            return self.diffusion_l.alpha(t_graph).square()
        raise RuntimeError(f"Unsupported lattice SG time weight mode={weight_mode!r}.")

    @staticmethod
    def _build_lattice_diffusion(
        *,
        lattice_diffusion_type: str,
        eps: float,
        lattice_parameterization: str,
    ) -> ContinuousDiffusion:
        if lattice_diffusion_type != "VP":
            raise ValueError("lattice_diffusion_type must be 'VP'.")
        return ContinuousVPDiffusion(
            eps=eps,
            parameterization=lattice_parameterization,
        )

    # ============================================================================
    # ALGORITHM 1
    # ============================================================================

    def algorithm1_training_targets(
        self,
        batch: Data | Batch,
        times: BatchTimes,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """
        Algorithm 1 in KLDM:
        sample noisy variables and score targets.
        """
        index = batch.batch

        # Diffuse lattice, KLDM Alg. 1
        l_t, eps_l = self.diffusion_l.forward_sample(
            t=times.lattice,
            x0=batch.l,
            num_atoms=batch.num_atoms,
        )
        target_l = self.diffusion_l.training_target(
            t=times.lattice,
            x0=batch.l,
            noise=eps_l,
            num_atoms=batch.num_atoms,
        )

        f_t, v_t, epsilon_v, epsilon_r, r_t = self.tdm.sample_noisy_state(
            t=times.nodes,
            f0=batch.pos,
            index=index, # the reason we give the index is because, it has if a batch has 2 crystals with 3 and 2 atoms then index = [0, 0, 0, 1, 1]
                         # THis is used to zero-center velocity noise per graph
        )

        target_v = self.tdm.build_simplified_training_velocity_score(
            t=times.nodes,
            r_t=r_t,
            v_t=v_t,
            index=index,
        )


        return (v_t, f_t, l_t), (target_v, target_l)

    # ============================================================================
    # ALGORITHM 2
    # ============================================================================

    def _mse_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Algorithm-2 helper: plain MSE averaged over feature dimensions."""
        loss = F.mse_loss(pred, target, reduction="none")
        return loss.reshape(loss.shape[0], -1).mean(dim=1)

    def algorithm2_loss(
        self,
        batch: Data | Batch,
        t: torch.Tensor,
        time_weight: torch.Tensor | None = None,
        debug: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Algorithm 2 in KLDM:
        network prediction + denoising score matching loss.
        """
        device = next(self.parameters()).device
        batch = batch.to(device)
        index = batch.batch
        times = make_times(batch, t)

        (v_t, f_t, l_t), (target_v, target_l) = self.algorithm1_training_targets(
            batch=batch,
            times=times,
        )

        preds = self.score_network(
            t=times.graph,
            pos=f_t,
            v=v_t,
            h=batch.atomic_numbers,
            l=l_t,
            node_index=index,
            edge_node_index=batch.edge_node_index,
        )

        out_v = preds["v"]
        out_l = preds["l"]

        loss_v_node = self._mse_per_sample(out_v, target_v)
        loss_l_graph = self._mse_per_sample(out_l, target_l)

        num_graphs = int(batch.num_graphs)
        loss_v_sum = torch.zeros(
            num_graphs,
            device=loss_v_node.device,
            dtype=loss_v_node.dtype,
        )
        loss_v_sum = loss_v_sum.index_add(0, index, loss_v_node)

        counts = torch.bincount(index, minlength=num_graphs).to(
            device=loss_v_node.device,
            dtype=loss_v_node.dtype,
        ).clamp_min(1.0)

        loss_v_graph = loss_v_sum / counts
        loss_v = loss_v_node.mean()
        loss_v_weighted = loss_v

        loss_sg_lattice_graph = torch.zeros_like(loss_l_graph)
        sg_time_weight_graph = torch.zeros_like(loss_l_graph)
        loss_conv_sg_graph = torch.zeros_like(loss_l_graph)
        conv_sg_time_weight_graph = torch.zeros_like(loss_l_graph)
        conv_weight_graph = torch.zeros_like(loss_l_graph)
        projection_error_pred = out_l.new_tensor(0.0)
        projection_error_gt = out_l.new_tensor(0.0)
        conv_projection_error_pred = out_l.new_tensor(0.0)
        conv_projection_error_gt = out_l.new_tensor(0.0)
        projection_error_pred_orbit = out_l.new_tensor(0.0)
        projection_error_gt_orbit = out_l.new_tensor(0.0)
        space_group = getattr(batch, "space_group", None)
        has_sg_labels = space_group is not None

        if self.lattice_debug and self.lattice_representation == "diffcsp_k" and has_sg_labels:
            sg = space_group.to(device=out_l.device)
            projection_error_pred = self.lattice_symmetry.direct_sg_residual_abs_mean(out_l, sg).mean()
            projection_error_gt = self.lattice_symmetry.direct_sg_residual_abs_mean(target_l, sg).mean()
            if self.lattice_orbit_metric_max_candidates is not None and int(self.lattice_orbit_metric_max_candidates) > 0:
                with torch.no_grad():
                    projection_error_pred_orbit = self.lattice_symmetry.orbit_sg_residual_abs_mean(
                        out_l.detach(),
                        sg,
                        max_candidates=self.lattice_orbit_metric_max_candidates,
                    ).mean()
                    projection_error_gt_orbit = self.lattice_symmetry.orbit_sg_residual_abs_mean(
                        target_l.detach(),
                        sg,
                        max_candidates=self.lattice_orbit_metric_max_candidates,
                    ).mean()
        if self.lattice_sg_lambda > 0.0:
            if self.lattice_representation != "diffcsp_k":
                raise RuntimeError("Soft SG lattice loss requires lattice_representation='diffcsp_k'.")
            if self.lattice_parameterization != "x0":
                raise RuntimeError("Soft SG lattice loss expects x0 lattice parameterization.")
            if space_group is None:
                raise RuntimeError("lattice_sg_lambda > 0 requires batch.space_group.")
            loss_sg_lattice_graph = self.lattice_symmetry.soft_lattice_sg_loss_per_graph(
                pred_k0=out_l,
                spacegroup=space_group.to(device=out_l.device),
                normalize=self.lattice_sg_normalize,
            )
            sg_time_weight_graph = self._lattice_sg_time_weight_values(
                times.lattice.to(device=out_l.device),
            ).to(device=out_l.device, dtype=out_l.dtype)
            loss_sg_lattice_graph = sg_time_weight_graph * loss_sg_lattice_graph

        if self.lambda_conv_sg > 0.0:
            if self.lattice_representation != "diffcsp_k":
                raise RuntimeError("Conventional SG auxiliary loss requires lattice_representation='diffcsp_k'.")
            if self.lattice_parameterization != "x0":
                raise RuntimeError("Conventional SG auxiliary loss expects x0 lattice parameterization.")
            if space_group is None:
                raise RuntimeError("lambda_conv_sg > 0 requires batch.space_group.")
            if not hasattr(batch, "conv_C") or not hasattr(batch, "conv_weight"):
                if self.conv_sg_require_valid_transform:
                    raise RuntimeError("lambda_conv_sg > 0 requires batch.conv_C and batch.conv_weight.")
                conv_C = torch.eye(3, device=out_l.device, dtype=out_l.dtype).view(1, 3, 3).expand(num_graphs, -1, -1)
                conv_weight_graph = torch.ones_like(loss_l_graph)
            else:
                conv_C = batch.conv_C.to(device=out_l.device, dtype=out_l.dtype).reshape(num_graphs, 3, 3)
                conv_weight_graph = batch.conv_weight.to(device=out_l.device, dtype=out_l.dtype).reshape(-1)
            if self.conv_sg_require_valid_transform and float(conv_weight_graph.detach().sum().item()) <= 0.0:
                raise RuntimeError("Conventional SG auxiliary is enabled, but this batch has no valid conv_C transforms.")
            sg = space_group.to(device=out_l.device)
            if self.conv_sg_control_mode == "shuffle_batch":
                if num_graphs > 1:
                    # Fake-control: preserve the SG/conv_C distribution but break graph correspondence.
                    perm = torch.arange(num_graphs, device=out_l.device).roll(1)
                    sg = sg.reshape(-1)[perm]
                    conv_C = conv_C[perm]
                    conv_weight_graph = conv_weight_graph[perm]
                else:
                    conv_weight_graph = torch.zeros_like(conv_weight_graph)
            loss_conv_sg_graph_raw, conv_residual_pred_raw = self.lattice_symmetry.conventional_sg_loss_and_residual_per_graph(
                primitive_k0=out_l,
                conv_C=conv_C,
                spacegroup=sg,
            )
            conv_sg_time_weight_graph = self._lattice_sg_time_weight_values(
                times.lattice.to(device=out_l.device),
                mode=self.conv_sg_time_weight,
            ).to(device=out_l.device, dtype=out_l.dtype)
            loss_conv_sg_graph = conv_weight_graph * conv_sg_time_weight_graph * loss_conv_sg_graph_raw
            with torch.no_grad():
                conv_residual_gt = self.lattice_symmetry.conventional_sg_residual_abs_mean(
                    target_l.detach(),
                    conv_C,
                    sg,
                )
                conv_denom = conv_weight_graph.detach().sum().clamp_min(1.0)
                conv_projection_error_pred = (conv_weight_graph.detach() * conv_residual_pred_raw.detach()).sum() / conv_denom
                conv_projection_error_gt = (conv_weight_graph.detach() * conv_residual_gt).sum() / conv_denom
        elif debug and self.lattice_representation == "diffcsp_k" and has_sg_labels and hasattr(batch, "conv_C") and hasattr(batch, "conv_weight"):
            with torch.no_grad():
                conv_C = batch.conv_C.to(device=out_l.device, dtype=out_l.dtype).reshape(num_graphs, 3, 3)
                conv_weight_graph = batch.conv_weight.to(device=out_l.device, dtype=out_l.dtype).reshape(-1)
                sg = space_group.to(device=out_l.device)
                conv_residual_pred = self.lattice_symmetry.conventional_sg_residual_abs_mean(
                    out_l.detach(),
                    conv_C,
                    sg,
                )
                conv_residual_gt = self.lattice_symmetry.conventional_sg_residual_abs_mean(
                    target_l.detach(),
                    conv_C,
                    sg,
                )
                conv_denom = conv_weight_graph.detach().sum().clamp_min(1.0)
                conv_projection_error_pred = (conv_weight_graph.detach() * conv_residual_pred).sum() / conv_denom
                conv_projection_error_gt = (conv_weight_graph.detach() * conv_residual_gt).sum() / conv_denom

        loss_l_weighted_graph = self.lambda_l * loss_l_graph
        loss_graph = (
            loss_v_graph
            + loss_l_weighted_graph
            + self.lattice_sg_lambda * loss_sg_lattice_graph
            + self.lambda_conv_sg * loss_conv_sg_graph
        )

        if time_weight is not None:
            weight_graph = time_weight.reshape(-1).to(device=loss_l_graph.device, dtype=loss_l_graph.dtype)
            weight_node = weight_graph[index]
            loss_v_weighted = (weight_node * loss_v_node).mean()
            loss_l_weighted = (weight_graph * loss_l_weighted_graph).mean()
            loss_sg_lattice_weighted = (weight_graph * loss_sg_lattice_graph).mean()
            loss_conv_sg_weighted = (weight_graph * loss_conv_sg_graph).mean()
        else:
            loss_l_weighted = loss_l_weighted_graph.mean()
            loss_sg_lattice_weighted = loss_sg_lattice_graph.mean()
            loss_conv_sg_weighted = loss_conv_sg_graph.mean()
        loss_sg_lattice_lambda_scaled = self.lattice_sg_lambda * loss_sg_lattice_weighted
        loss_conv_sg_lambda_scaled = self.lambda_conv_sg * loss_conv_sg_weighted
        total_loss = loss_v_weighted + loss_l_weighted + loss_sg_lattice_lambda_scaled + loss_conv_sg_lambda_scaled

        metrics = {
            "loss": total_loss.detach(),
            "loss_v": loss_v.detach(),
            "loss_l": loss_l_graph.mean().detach(),
            "loss_sg_lattice": loss_sg_lattice_graph.mean().detach(),
            "loss_sg_lattice_weighted": loss_sg_lattice_weighted.detach(),
            "loss_sg_lattice_lambda_scaled": loss_sg_lattice_lambda_scaled.detach(),
            "loss_conv_sg": loss_conv_sg_graph.mean().detach(),
            "loss_conv_sg_weighted": loss_conv_sg_weighted.detach(),
            "loss_conv_sg_lambda_scaled": loss_conv_sg_lambda_scaled.detach(),
            "lambda_l": out_l.new_tensor(self.lambda_l).detach(),
            "lambda_sg_lattice": out_l.new_tensor(self.lattice_sg_lambda).detach(),
            "lambda_conv_sg": out_l.new_tensor(self.lambda_conv_sg).detach(),
            "lattice_debug": out_l.new_tensor(float(self.lattice_debug)).detach(),
            "lattice_sg_time_weight_mean": sg_time_weight_graph.mean().detach(),
            "conv_sg_time_weight_mean": conv_sg_time_weight_graph.mean().detach(),
            "conv_weight_mean": conv_weight_graph.mean().detach(),
            "projection_error_pred_k": projection_error_pred.detach(),
            "projection_error_gt_k": projection_error_gt.detach(),
            "primitive_projection_error_pred_k": projection_error_pred.detach(),
            "primitive_projection_error_gt_k": projection_error_gt.detach(),
            "conv_projection_error_pred_k": conv_projection_error_pred.detach(),
            "conv_projection_error_gt_k": conv_projection_error_gt.detach(),
            "projection_error_pred_direct_k": projection_error_pred.detach(),
            "projection_error_gt_direct_k": projection_error_gt.detach(),
            "projection_error_pred_orbit_k": projection_error_pred_orbit.detach(),
            "projection_error_gt_orbit_k": projection_error_gt_orbit.detach(),
            "loss_v_weighted": loss_v_weighted.detach(),
            "loss_l_weighted": loss_l_weighted.detach(),
            "loss_graph": loss_graph.detach(),
            "loss_v_graph": loss_v_graph.detach(),
            "loss_l_graph": loss_l_graph.detach(),
            "loss_sg_lattice_graph": loss_sg_lattice_graph.detach(),
            "loss_conv_sg_graph": loss_conv_sg_graph.detach(),
            "loss_v_weighted_graph": loss_v_graph.detach(),
            "loss_l_weighted_graph": loss_l_weighted_graph.detach(),
        }
        return total_loss, metrics

    def _reverse_lattice_sampling_step(
        self,
        *,
        t: torch.Tensor,
        x_t: torch.Tensor,
        pred: torch.Tensor,
        dt: float,
        num_atoms: torch.Tensor,
    ) -> torch.Tensor:
        return self.diffusion_l.reverse_step(
            t=t,
            x_t=x_t,
            pred=pred,
            dt=dt,
            num_atoms=num_atoms,
        )



    def _prepare_csp_sampling(
        self,
        *,
        batch: Batch | Data,
        n_steps: int,
        t_start: float,
        t_final: float,
    ) -> dict[str, Any]:
        device = next(self.parameters()).device
        batch = batch.to(device)
        sampling_time_grid = sampling_grid(
            batch,
            n_steps=int(n_steps),
            t_start=float(t_start),
            t_final=float(t_final),
        )
        restore_training = self.score_network.training
        self.score_network.eval()

        f_t = self.tdm.wrap_displacements(torch.rand_like(batch.pos))
        v_t = self.tdm.sample_velocity_noise(ref=batch.pos, index=batch.batch)

        l_t = self.diffusion_l.sample_prior(
            x_like=batch.l,
            num_atoms=batch.num_atoms,
        )

        return {
            "batch": batch,
            "sampling_time_grid": sampling_time_grid,
            "restore_training": restore_training,
            "f_t": f_t,
            "v_t": v_t,
            "l_t": l_t,
            "a_t": batch.atomic_numbers,
            "node_index": batch.batch,
            "edge_node_index": batch.edge_node_index,
        }

    def _run_csp_em_reverse_chain(self, state: dict[str, Any]) -> dict[str, Any]:
        with torch.no_grad():
            for times in iter_sampling_times(batch=state["batch"], grid=state["sampling_time_grid"]):
                preds = self.score_network(
                    t=times.now.graph,
                    pos=state["f_t"],
                    v=state["v_t"],
                    h=state["a_t"],
                    l=state["l_t"],
                    node_index=state["node_index"],
                    edge_node_index=state["edge_node_index"],
                )

                score_v = self.tdm.reconstruct_full_reverse_velocity_score(
                    t=times.now.nodes,
                    v_t=state["v_t"],
                    pred_v=preds["v"],
                    index=state["node_index"],
                )

                state["f_t"], state["v_t"] = self.tdm.reverse_exp_step(
                    f_t=state["f_t"],
                    v_t=state["v_t"],
                    score_v=score_v,
                    index=state["node_index"],
                    dt=times.dt,
                )

                state["l_t"] = self._reverse_lattice_sampling_step(
                    t=times.now.lattice,
                    x_t=state["l_t"],
                    pred=preds["l"],
                    dt=times.dt,
                    num_atoms=state["batch"].num_atoms,
                )

        return state

    def _run_csp_pc_reverse_chain(
        self,
        state: dict[str, Any],
        *,
        tau: float,
        n_correction_steps: int,
    ) -> dict[str, Any]:
        del n_correction_steps
        with torch.no_grad():
            for times in iter_sampling_times(batch=state["batch"], grid=state["sampling_time_grid"]):
                preds = self.score_network(
                    t=times.now.graph,
                    pos=state["f_t"],
                    v=state["v_t"],
                    h=state["a_t"],
                    l=state["l_t"],
                    node_index=state["node_index"],
                    edge_node_index=state["edge_node_index"],
                )

                state["f_t"], state["v_t"] = self.tdm.reverse_step_predictor(
                    t=times.now.nodes,
                    f_t=state["f_t"],
                    v_t=state["v_t"],
                    pred_v=preds["v"],
                    index=state["node_index"],
                    dt=times.dt,
                )

                if times.t_next_float < 1e-3:
                    continue

                preds = self.score_network(
                    t=times.next.graph,
                    pos=state["f_t"],
                    v=state["v_t"],
                    h=state["a_t"],
                    l=state["l_t"],
                    node_index=state["node_index"],
                    edge_node_index=state["edge_node_index"],
                )

                state["f_t"], state["v_t"] = self.tdm.reverse_step_corrector(
                    t=times.next.nodes,
                    f_t=state["f_t"],
                    v_t=state["v_t"],
                    pred_v=preds["v"],
                    dt=times.dt,
                    index=state["node_index"],
                    tau=float(tau),
                )

                state["l_t"] = self._reverse_lattice_sampling_step(
                    t=times.next.lattice,
                    x_t=state["l_t"],
                    pred=preds["l"],
                    dt=times.dt,
                    num_atoms=state["batch"].num_atoms,
                )

        return state

    def sample_CSP_algorithm3(
        self,
        n_steps: int,
        batch: Batch | Data,
        t_start: float = 1.0,
        t_final: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Algorithm 3 from Appendix H: EM sampling for the CSP model.

        At each time level:
            1. evaluate the network
            2. build the full velocity score
            3. do one exponential-Euler step for (f_t, v_t)
            4. do one reverse diffusion step for l_t
        """
        state = self._prepare_csp_sampling(
            batch=batch,
            n_steps=n_steps,
            t_start=t_start,
            t_final=t_final,
        )
        state = self._run_csp_em_reverse_chain(state)

        if state["restore_training"]:
            self.score_network.train()

        return state["f_t"], state["v_t"], state["l_t"], state["a_t"]

    def sample_CSP_algorithm4(
        self,
        n_steps: int,
        batch: Batch | Data,
        tau: float,
        n_correction_steps: int,
        t_start: float = 1.0,
        t_final: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Algorithm 4 from Appendix H: vanilla predictor-corrector sampling.

        At each time level:
            1. do one predictor step for positions/velocities
            2. do one Langevin-style corrector update for positions/velocities
            3. do one EM reverse step for the lattice
        """
        state = self._prepare_csp_sampling(
            batch=batch,
            n_steps=n_steps,
            t_start=t_start,
            t_final=t_final,
        )
        state = self._run_csp_pc_reverse_chain(
            state,
            tau=float(tau),
            n_correction_steps=int(n_correction_steps),
        )

        if state["restore_training"]:
            self.score_network.train()

        return state["f_t"], state["v_t"], state["l_t"], state["a_t"]



def main() -> None:
    device = get_default_device()

    from kldmPlus.data import CSPTask, resolve_data_root
    root = resolve_data_root()

    loader = CSPTask().dataloader(
        root=root,
        split="val",
        batch_size=1,
        shuffle=False,
        download=True,
    )
    batch = next(iter(loader)).to(device)

    model = ModelKLDM(
        device=device,
        score_network_kwargs={
            "hidden_dim": 512,
            "time_dim": 256,
            "num_layers": 6,
            "num_freqs": 128,
            "ln": True,
            "h_dim": 100,
            "smooth": False,
            "pred_v": True,
            "pred_l": True,
            "pred_h": False,
            "zero_cog": True,
        },
    ).to(device)

    pos_t, v_t, l_t, h_t = model.sample_CSP_algorithm3(
        n_steps=1000,
        batch=batch,
    )

    print("Sampled one CSP crystal")
    print("pos shape:", tuple(pos_t.shape))
    print("v shape:", tuple(v_t.shape))
    print("l shape:", tuple(l_t.shape))
    print("h shape:", tuple(h_t.shape))

    print("\nFirst 3 sampled fractional coordinates:")
    print(pos_t[:3])

    print("\nSampled lattice:")
    print(l_t)

if __name__ == "__main__":
    main()
