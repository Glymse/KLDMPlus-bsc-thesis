# We use ruff
# To format our code!!!
# Remember to write this in paper if relevant.

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, Batch

from kldmPlus.diffusionModels.continuous import (
    ContinuousDiffusion,
    ContinuousMattergenVPDiffusion,
    ContinuousVPDiffusion,
)
from kldmPlus.diffusionModels.tdm import TrivialisedDiffusion as TDM
from kldmPlus.scoreNetwork.scoreNetwork import CSPVNet
from kldmPlus.utils.device import get_default_device
from kldmPlus.utils.time import BatchTimes, iter_sampling_times, make_times, sampling_grid


@dataclass
class PreparedTrainingBatch:
    """
    Fixed noisy training bundle for one KLDM++ loss evaluation.

    Inspired by the adaptive paper:
    the REINFORCE reward must compare before/after model losses on the same
    corruption, so the sampler needs a reusable container for noisy states and
    targets.
    """

    times: BatchTimes
    v_t: torch.Tensor
    f_t: torch.Tensor
    l_t: torch.Tensor
    target_v: torch.Tensor
    target_l: torch.Tensor
    atomic_numbers: torch.Tensor
    node_index: torch.Tensor
    edge_node_index: torch.Tensor
    num_graphs: int
    lattice_representation: str


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
        tdm_centered_sigma_norm_correction: bool = False,
        lattice_parameterization: str = "eps",
        lattice_diffusion_type: str = "VP",
        lattice_representation: str = "kldm",
        mattergen_lattice_c: float | None = None,
        mattergen_lattice_nu: float | None = None,
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
            centered_sigma_norm_correction=tdm_centered_sigma_norm_correction,
        )
        self.diffusion_l = self._build_lattice_diffusion(
            lattice_diffusion_type=lattice_diffusion_type,
            eps=eps,
            lattice_parameterization=lattice_parameterization,
            mattergen_lattice_c=mattergen_lattice_c,
            mattergen_lattice_nu=mattergen_lattice_nu,
        )
        self.eps = eps
        self.lattice_parameterization = lattice_parameterization
        self.lattice_diffusion_type = lattice_diffusion_type
        self.lattice_representation = lattice_representation

    @staticmethod
    def _build_lattice_diffusion(
        *,
        lattice_diffusion_type: str,
        eps: float,
        lattice_parameterization: str,
        mattergen_lattice_c: float | None,
        mattergen_lattice_nu: float | None,
    ) -> ContinuousDiffusion:
        if lattice_diffusion_type == "VP":
            diffusion_cls = ContinuousVPDiffusion
            diffusion_kwargs: dict[str, Any] = {}
        elif lattice_diffusion_type == "mattergenVP":
            diffusion_cls = ContinuousMattergenVPDiffusion
            if mattergen_lattice_c is None or mattergen_lattice_nu is None:
                raise ValueError(
                    "mattergenVP requires mattergen_lattice_c and mattergen_lattice_nu.",
                )
            diffusion_kwargs = {
                "c": float(mattergen_lattice_c),
                "nu": float(mattergen_lattice_nu),
            }
        else:
            raise ValueError(
                "lattice_diffusion_type must be 'VP' or 'mattergenVP'.",
            )

        return diffusion_cls(
            eps=eps,
            parameterization=lattice_parameterization,
            **diffusion_kwargs,
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
    # Loss calculators for algorithm 2
    # ============================================================================

    def mse_loss_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Plain MSE, averaged over feature dims.
        """
        loss = F.mse_loss(pred, target, reduction="none")
        return loss.reshape(loss.shape[0], -1).mean(dim=1)

    def mattergen_lattice_mse_6d_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted 6D MSE that matches averaging over the full symmetric 3x3 cell.
        """
        # Code segment inspired from mattergen
        # (mattergen/diffusion/training/field_loss.py:118-152,
        #  mattergen/common/loss.py:35-40).
        #
        # Official MatterGen computes denoising loss on the full `cell` tensor and
        # then averages over all matrix entries. In the KLDM port we store only the
        # six unique entries of the symmetric matrix, so off-diagonal terms need a
        # factor of 2 to match the full 3x3 averaging.
        weights = pred.new_tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        loss = (pred - target).square() * weights
        return loss.sum(dim=-1) / 9.0

    # ============================================================================
    # ALGORITHM 2
    # ============================================================================

    def prepare_training_batch(
        self,
        batch: Data | Batch,
        t: torch.Tensor,
        *,
        lattice_noise: torch.Tensor | None = None,
        velocity_noise: torch.Tensor | None = None,
        position_noise: torch.Tensor | None = None,
    ) -> PreparedTrainingBatch:
        """
        Build one fixed noisy training example bundle for Algorithm 2.

        Inspired by the adaptive paper:
        the sampler reward compares before/after improvement at probe times,
        which only makes sense when both evaluations reuse the same corruption.
        """
        device = next(self.parameters()).device
        batch = batch.to(device)
        index = batch.batch
        times = make_times(batch, t)

        l_t, eps_l = self.diffusion_l.forward_sample(
            t=times.lattice,
            x0=batch.l,
            noise=lattice_noise,
            num_atoms=batch.num_atoms,
        )
        target_l = self.diffusion_l.training_target(
            t=times.lattice,
            x0=batch.l,
            noise=eps_l,
            num_atoms=batch.num_atoms,
        )

        f_t, v_t, _epsilon_v, _epsilon_r, r_t = self.tdm.sample_noisy_state(
            t=times.nodes,
            f0=batch.pos,
            index=index,
            epsilon_v=velocity_noise,
            epsilon_r=position_noise,
        )
        target_v = self.tdm.build_simplified_training_velocity_score(
            t=times.nodes,
            r_t=r_t,
            v_t=v_t,
            index=index,
        )

        return PreparedTrainingBatch(
            times=times,
            v_t=v_t,
            f_t=f_t,
            l_t=l_t,
            target_v=target_v,
            target_l=target_l,
            atomic_numbers=batch.atomic_numbers,
            node_index=index,
            edge_node_index=batch.edge_node_index,
            num_graphs=int(batch.num_graphs),
            lattice_representation=self.lattice_representation,
        )

    def loss_from_prepared(
        self,
        prepared: PreparedTrainingBatch,
        *,
        time_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Evaluate the KLDM++ denoising loss from a fixed noisy bundle.

        This is the loss-only half of Algorithm 2 and pairs with
        `prepare_training_batch(...)` so adaptive samplers can reuse exact
        probe corruptions across before/after reward measurements.
        """
        preds = self.score_network(
            t=prepared.times.graph,
            pos=prepared.f_t,
            v=prepared.v_t,
            h=prepared.atomic_numbers,
            l=prepared.l_t,
            node_index=prepared.node_index,
            edge_node_index=prepared.edge_node_index,
        )

        out_v = preds["v"]
        out_l = preds["l"]

        loss_v_node = self.mse_loss_per_sample(out_v, prepared.target_v)
        if prepared.lattice_representation == "mattergen":
            loss_l_graph = self.mattergen_lattice_mse_6d_per_sample(out_l, prepared.target_l)
        else:
            loss_l_graph = self.mse_loss_per_sample(out_l, prepared.target_l)

        loss_v_sum = torch.zeros(
            prepared.num_graphs,
            device=loss_v_node.device,
            dtype=loss_v_node.dtype,
        )
        loss_v_sum = loss_v_sum.index_add(0, prepared.node_index, loss_v_node)

        counts = torch.bincount(prepared.node_index, minlength=prepared.num_graphs).to(
            device=loss_v_node.device,
            dtype=loss_v_node.dtype,
        ).clamp_min(1.0)

        loss_v_graph = loss_v_sum / counts
        loss_graph = loss_v_graph + loss_l_graph

        if time_weight is not None:
            weight = time_weight.reshape(-1).to(device=loss_graph.device, dtype=loss_graph.dtype)
            total_loss = (weight * loss_graph).mean()
        else:
            total_loss = loss_graph.mean()

        metrics = {
            "loss": total_loss.detach(),
            "loss_v": loss_v_graph.mean().detach(),
            "loss_l": loss_l_graph.mean().detach(),
            "loss_graph": loss_graph.detach(),
            "loss_v_graph": loss_v_graph.detach(),
            "loss_l_graph": loss_l_graph.detach(),
        }
        return total_loss, metrics

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
        del debug
        prepared = self.prepare_training_batch(batch=batch, t=t)
        return self.loss_from_prepared(prepared, time_weight=time_weight)



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

        with torch.no_grad():
            """
            Dt is a positive “backward step size”. hence why the sampler uses
            different sign than the appendix algorithms.
            """
            for times in iter_sampling_times(batch=state["batch"], grid=state["sampling_time_grid"]):

                preds_curr = state["score_network"](
                    t=times.now.graph,
                    pos=state["f_t"],
                    v=state["v_t"],
                    h=state["a_t"],
                    l=state["l_t"],
                    node_index=state["node_index"],
                    edge_node_index=state["edge_node_index"],
                )

                # Build the full KLDM velocity score from the predicted
                # simplified wrapped-normal term.
                score_v = state["sampling_tdm"].reconstruct_full_reverse_velocity_score(
                    t=times.now.nodes,
                    v_t=state["v_t"],
                    pred_v=preds_curr["v"],
                    index=state["node_index"],
                )

                # Algorithm 3 update for (f_t, v_t): one exponential-Euler step.
                state["f_t"], state["v_t"] = state["sampling_tdm"].reverse_exp_step(
                    f_t=state["f_t"],
                    v_t=state["v_t"],
                    score_v=score_v,
                    index=state["node_index"],
                    dt=times.dt,
                )

                # Lattice branch: one reverse step using the lattice prediction.
                state["l_t"] = state["sampling_diffusion_l"].reverse_step(
                    t=times.now.lattice,
                    x_t=state["l_t"],
                    pred=preds_curr["l"],
                    dt=times.dt,
                    num_atoms=state["batch"].num_atoms,
                )

        if state["restore_training"]:
            state["score_network"].train()

        return state["f_t"], state["v_t"], state["l_t"], state["a_t"]

    def sample_CSP_algorithm4(
        self,
        n_steps: int,
        batch: Batch | Data,
        t_start: float = 1.0,
        t_final: float = 1e-6,
        tau: float = 0.25,
        n_correction_steps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Algorithm 4 from Appendix H, adapted to our internal scaled chart.

        Per step:
            1. evaluate the network at the current time t_n
            2. predictor from t_n to t_{n+1}
            3. evaluate the network again at t_{n+1} on the predicted state
            4. one corrector step at t_{n+1}
            5. one EM step for the lattice branch

        Important:
            - TDM internally uses velocity_scale = 1 / (2*pi)
            - therefore TDM predictor/corrector must use
              reconstruct_full_reverse_velocity_score(...)
            - corrector Langevin noise must use sample_velocity_noise(...)
            - our time grid uses dt = t_n - t_{n+1} > 0 while integrating
              backward, so the predictor position update needs the sign change
              documented in tdm.py to stay equivalent to the paper step
        """
        state = self._prepare_csp_sampling(
            batch=batch,
            n_steps=n_steps,
            t_start=t_start,
            t_final=t_final,
        )

        with torch.no_grad():
            for times in iter_sampling_times(batch=state["batch"], grid=state["sampling_time_grid"]):
                # One predictor-corrector transition in the decreasing grid:
                # times.now is the current/noisier time, and times.next is the
                # predicted/cleaner time used for the second network evaluation.
                # 1. Evaluate the network at the current time level t_n.
                preds_curr = state["score_network"](
                    t=times.now.graph,
                    pos=state["f_t"],
                    v=state["v_t"],
                    h=state["a_t"],
                    l=state["l_t"],
                    node_index=state["node_index"],
                    edge_node_index=state["edge_node_index"],
                )

                # 2. Predictor from t_n to t_{n+1}. The TDM helper uses the
                # internally scaled full velocity score and the sign convention
                # corresponding to our positive backward-step dt.
                state["f_t"], state["v_t"] = state["sampling_tdm"].reverse_step_predictor(
                    t=times.now.nodes,
                    f_t=state["f_t"],
                    v_t=state["v_t"],
                    pred_v=preds_curr["v"],
                    index=state["node_index"],
                    dt=times.dt,
                )

                # Near t = 0 the reconstructed velocity score becomes very
                # stiff because the Gaussian variance term goes to zero.
                # Keep the predictor move, but skip the final corrector/lattice
                # update once the next time level is below 1e-3.
                if times.t_next_float < 1e-3:
                    continue

                preds_next = state["score_network"](
                    t=times.next.graph,
                    pos=state["f_t"],
                    v=state["v_t"],
                    h=state["a_t"],
                    l=state["l_t"],
                    node_index=state["node_index"],
                    edge_node_index=state["edge_node_index"],
                )

                # 4. Single corrector step at t_{n+1}.
                state["f_t"], state["v_t"] = state["sampling_tdm"].reverse_step_corrector(
                    t=times.next.nodes,
                    f_t=state["f_t"],
                    v_t=state["v_t"],
                    pred_v=preds_next["v"],
                    dt=times.dt,
                    index=state["node_index"],
                    tau=tau,
                )

                # 5. Lattice remains a single EM step in Appendix H.
                state["l_t"] = state["sampling_diffusion_l"].reverse_step(
                    t=times.next.lattice,
                    x_t=state["l_t"],
                    pred=preds_next["l"],
                    dt=times.dt,
                    num_atoms=state["batch"].num_atoms,
                )

        if state["restore_training"]:
            state["score_network"].train()

        return state["f_t"], state["v_t"], state["l_t"], state["a_t"]

    def _prepare_csp_sampling(
        self,
        batch: Batch | Data,
        n_steps: int,
        t_start: float,
        t_final: float,
    ) -> dict[str, Any]:
        device = next(self.parameters()).device
        batch = batch.to(device)

        node_index = batch.batch
        edge_node_index = batch.edge_node_index
        num_graphs = batch.num_graphs

        # Appendix H priors, kept in one place so the sampler owns its initial state:
        # f_T ~ U(0, 1) represented in TDM's signed chart, v_T ~ centered N_v(0, I),
        # and l_T ~ N(0, I).
        f_t = self.tdm.wrap_displacements(torch.rand_like(batch.pos))
        v_t = self.tdm.sample_velocity_noise(f_t, index=node_index)
        l_t = self.diffusion_l.sample_prior(
            x_like=batch.l,
            num_atoms=batch.num_atoms,
        )
        a_t = batch.atomic_numbers

        score_network = self.score_network
        restore_training = score_network.training
        score_network.eval()

        sampling_time_grid = sampling_grid(
            batch=batch,
            n_steps=n_steps,
            t_start=t_start,
            t_final=t_final,
        )

        return {
            "batch": batch,
            "device": device,
            "dtype": batch.pos.dtype,
            "n_steps": n_steps,
            "num_graphs": num_graphs,
            "node_index": node_index,
            "edge_node_index": edge_node_index,
            "sampling_tdm": self.tdm,
            "sampling_diffusion_l": self.diffusion_l,
            "score_network": score_network,
            "restore_training": restore_training,
            "f_t": f_t,
            "v_t": v_t,
            "l_t": l_t,
            "a_t": a_t,
            "sampling_time_grid": sampling_time_grid,
        }


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
