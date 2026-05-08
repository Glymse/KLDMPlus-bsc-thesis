from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kldmPlus.distribution.sigma_norm import WrappedNormalSigmaNorm
from kldmPlus.distribution.wrapped_normal import d_log_wrapped_normal
from kldmPlus.scoreNetwork.utils import scatter_center


class TrivialisedDiffusion(nn.Module):
    """
    TDM helper for the KLDM position/velocity branch.

    This module handles three different pieces of the pipeline.

    Forward process:
        1. sample a noisy velocity `v_t`
        2. sample a wrapped displacement `r_t`
        3. move the clean fractional coordinates to `f_t = wrap(f_0 + r_t)`

    Training target:
        1. compute the wrapped-normal part of the KLDM velocity score
        2. remove the known derivative constant
        3. normalize that simplified target with `sigma_norm`
        4. train the network to predict this simplified quantity

    Reverse sampling:
        1. take the network prediction of the simplified target
        2. undo the `sigma_norm` normalization
        3. restore the missing KLDM prefactor
        4. add the analytic Gaussian velocity score term
        5. use the reconstructed full score inside the sampler update

    So the key split is:
        - training predicts a simplified wrapped-normal score target
        - sampling reconstructs the full reverse velocity score from it

    Convention:
        external model time is in [0, 1], while TDM internally uses [0, 2].
    """

    def __init__(
        self,
        eps: float = 1e-6,
        wrapped_normal_K: int = 3,
        n_sigmas: int = 2000,
        compute_sigma_norm: bool = True,
        velocity_scale: float | None = None,
        sigma_norm_estimator: str = "quadrature",
        sigma_norm_density_K: int | None = None,
        sigma_norm_grid_points: int = 4096,
        sigma_norm_mc_samples: int = 20000,
        centered_sigma_norm_correction: bool = False,
    ) -> None:
        super().__init__()

        self.eps = float(eps)
        self.T = 2.0
        self.vel_scale = float(1.0 / (2.0 * math.pi) if velocity_scale is None else velocity_scale)
        self.wrapped_normal_K = int(wrapped_normal_K)
        self.compute_sigma_norm = bool(compute_sigma_norm)
        self.sigma_norm_estimator = str(sigma_norm_estimator)
        self.sigma_norm_density_K = sigma_norm_density_K
        self.sigma_norm_grid_points = int(sigma_norm_grid_points)
        self.sigma_norm_mc_samples = int(sigma_norm_mc_samples)
        self.centered_sigma_norm_correction = bool(centered_sigma_norm_correction)

        if self.compute_sigma_norm:
            sigma_grid = self.wrapped_gaussian_sigma_r_t(torch.linspace(0.0, self.T, int(n_sigmas)))
            sigma_norm_values = WrappedNormalSigmaNorm(
                K=self.wrapped_normal_K,
                estimator=self.sigma_norm_estimator,
                K_density=self.sigma_norm_density_K,
                num_grid_points=self.sigma_norm_grid_points,
                num_monte_carlo_samples=self.sigma_norm_mc_samples,
                eps=self.eps,
            )(sigma_grid)
        else:
            sigma_norm_values = torch.ones(int(n_sigmas), dtype=torch.get_default_dtype())

        self.register_buffer("_sigma_norms", sigma_norm_values)

    def sigma_norm_t(self, t: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate sigma_norm for internal time t in [0, T]."""
        n = self._sigma_norms.numel()

        x = (t / self.T).clamp(0.0, 1.0) * (n - 1)

        idx0 = torch.floor(x).long()
        idx1 = (idx0 + 1).clamp(max=n - 1)

        w = (x - idx0.to(dtype=x.dtype))

        return (1.0 - w) * self._sigma_norms[idx0] + w * self._sigma_norms[idx1]

    def sigma_norm_factor(
        self,
        t: torch.Tensor,
        index: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        # Build the sqrt sigma-norm factor, optionally corrected for graph-wise centering.
        sigma_norm = self.sigma_norm_t(t).clamp_min(self.eps)

        if self.centered_sigma_norm_correction:
            num_graphs = int(index.max().item()) + 1
            counts = torch.bincount(index, minlength=num_graphs).to(
                device=ref.device,
                dtype=ref.dtype,
            )
            count_per_atom = counts[index].clamp_min(1.0)
            center_factor = ((count_per_atom - 1.0) / count_per_atom).clamp_min(self.eps)
            sigma_norm = sigma_norm * center_factor

        return self.match_dims(torch.sqrt(sigma_norm).clamp_min(self.eps), ref)

    # -------------------------------------------------------------------------
    # Wrapping functions
    # -------------------------------------------------------------------------

    @staticmethod
    def wrap_positions(x: torch.Tensor) -> torch.Tensor:
        """Wrap unit-cell fractional coordinates into [0, 1)."""
        return torch.remainder(x, 1.0)

    @staticmethod
    def wrap_displacements(x: torch.Tensor) -> torch.Tensor:
        """Wrap signed periodic displacements into [-0.5, 0.5)."""
        return torch.remainder(x + 0.5, 1.0) - 0.5

    # -------------------------------------------------------------------------
    # Closed-form coefficients
    # -------------------------------------------------------------------------

    def gaussian_velocity_mean(self, t: torch.Tensor) -> torch.Tensor:
        """Mean coefficient of the Gaussian velocity forward kernel."""
        return torch.exp(-t)

    def gaussian_velocity_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Gaussian velocity standard deviation before epsilon scaling.

        The actual sampled velocity noise is:
            epsilon_v_scaled = vel_scale * epsilon_v

        so the full velocity standard deviation in the unit chart is
            vel_scale * gaussian_velocity_sigma(t).
        """
        return torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * t), min=self.eps))

    def wrapped_gaussian_mu_r_t(self, t: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        """
        Wrapped-Gaussian mean for the displacement r_t.

        Design choice: v0 = 0, so
            mu_r(t) = (1 - exp(-t)) / (1 + exp(-t)) * v_t
        """
        coeff = self.match_dims((1.0 - torch.exp(-t)) / (1.0 + torch.exp(-t)), v_t)
        return coeff * v_t

    def wrapped_gaussian_sigma_r_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Wrapped-Gaussian standard deviation for the displacement r_t.

        This already includes the unit-chart velocity scaling.
        """
        base_var = 2.0 * t + 8.0 / (1.0 + torch.exp(t)) - 4.0
        return self.vel_scale * torch.sqrt(torch.clamp(base_var, min=self.eps))



    # -------------------------------------------------------------------------
    # Sampling helpers
    # -------------------------------------------------------------------------

    def sample_velocity_noise(self, ref: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        Sample centered velocity noise in the unit chart.

        We first sample epsilon_v ~ N(0, I), center it per graph, and then
        apply the 1 / (2*pi) scaling through vel_scale.
        """
        epsilon_v = torch.randn_like(ref)
        epsilon_v = scatter_center(epsilon_v, index=index)
        return self.vel_scale * epsilon_v

    # -------------------------------------------------------------------------
    # Training helpers
    # -------------------------------------------------------------------------

    def sample_noisy_state(
        self,
        t: torch.Tensor,
        f0: torch.Tensor,
        index: torch.Tensor,
        epsilon_v: torch.Tensor | None = None,
        epsilon_r: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample one forward noisy state `(f_t, v_t)`.

        In KLDM we first sample a velocity, then use that velocity to define a
        wrapped displacement on the torus:

            p_t|0(f_t, v_t | f0, v0)
                = WN(r_t | mu_r(t), sigma_r(t)) * N(v_t | 0, sigma_v(t))

        with the design choice `v0 = 0`.
        """
        t = self.T * t


        # -------------------------------------------------
        # 1. Sample v_t
        # -------------------------------------------------
        #
        # The velocity kernel is Gaussian and zero-mean because we fix v0 = 0:
        #
        #   v_t ~ N(0, sigma_v(t)^2 I)
        #
        # In the unit-period chart we keep the 1 / (2*pi) factor inside the
        # sampled noise:
        #
        #   epsilon_v_scaled = vel_scale * epsilon_v
        #   v_t = gaussian_velocity_sigma(t) * epsilon_v_scaled
        #
        # We center epsilon_v per graph

        if epsilon_v is None:
            epsilon_v = torch.randn_like(f0)
        epsilon_v = scatter_center(epsilon_v, index=index)
        epsilon_v_scaled = self.vel_scale * epsilon_v

        gaussian_velocity_sigma_t = self.match_dims(self.gaussian_velocity_sigma(t), f0)
        v_t = gaussian_velocity_sigma_t * epsilon_v_scaled

        # -------------------------------------------------
        # 2. Sample r_t from the wrapped Gaussian
        # -------------------------------------------------
        #
        # After sampling v_t, the displacement kernel is wrapped Gaussian.
        # Since v0 = 0, its mean simplifies to:
        #
        #   mu_r(t) = (1.0 - exp(-t)) / (1.0 + exp(-t)) * v_t
        #
        # Then we sample centered Gaussian noise epsilon_r and wrap the result
        # back into [-0.5, 0.5):
        #
        #   r_t = wrap(mu_r(t) + sigma_r(t) * epsilon_r)

        mu_r = self.wrapped_gaussian_mu_r_t(t, v_t)

        # Keep the clean state in the centered displacement chart before adding
        # wrapped noise. This is the chart used throughout the TDM equations.
        f0 = self.wrap_displacements(f0)

        sigma_r = self.match_dims(self.wrapped_gaussian_sigma_r_t(t), f0)

        if epsilon_r is None:
            epsilon_r = torch.randn_like(f0)
        epsilon_r = scatter_center(epsilon_r, index=index)
        r_t = self.wrap_displacements(mu_r + sigma_r * epsilon_r)

        # -------------------------------------------------
        # 3. Move f0 on the torus
        # -------------------------------------------------
        #
        # The torus update is just the group action in the unit chart:
        #
        #   f_t = wrap(f0 + r_t)
        #
        # We do not center f_t afterwards. The target and network output are
        # centered, but the noisy sample itself should stay the actual forward
        # draw from the wrapped process.

        f_t = self.wrap_displacements(f0 + r_t)


        #As suspected, this is indeed a error in the KLDM paper appendix.
        #This will break the [-0.5, 0.5] space. Instead they meant to
        #Center the target position instead.
        #f_t = scatter_center(f_t, index=index)

        return f_t, v_t, epsilon_v, epsilon_r, r_t

    def build_simplified_training_velocity_score(
        self,
        t: torch.Tensor,
        r_t: torch.Tensor,
        v_t: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the simplified training target for the velocity head.

        The reverse sampler needs the full KLDM velocity score, which has two
        pieces:

            1. a wrapped-normal term coming from the torus displacement r_t
            2. a Gaussian term coming from the velocity prior

        We do not train the network on the full score directly. Instead we make
        the network predict a simplified wrapped-normal term only, because:

            - the Gaussian velocity term is known analytically
            - the wrapped-normal term is the hard part to learn
            - sigma_norm normalization keeps the target scale more stable across t

        The missing factors are restored later in
        `reconstruct_full_reverse_velocity_score(...)` before reverse sampling.
        """
        t = self.T * t

        # The wrapped-normal score is evaluated at the actual sampled
        # displacement r_t, with the mean implied by the sampled velocity v_t.
        # We wrap mu_r back into the centered displacement chart so the
        # truncated wrapped-normal score stays numerically stable.
        mu_r = self.wrap_displacements(self.wrapped_gaussian_mu_r_t(t, v_t))
        sigma_r = self.match_dims(self.wrapped_gaussian_sigma_r_t(t), r_t)

        # Full wrapped-normal contribution to the KLDM velocity score:
        #
        #   ((1.0 - exp(-t)) / (1.0 + exp(-t))) * d_log_WN(r_t | mu_r, sigma_r)
        #
        # We strip off the leading factor here so the network predicts the
        # simplified term only. That factor is restored in
        # reconstruct_full_reverse_velocity_score(...).
        wrapped_normal_score = d_log_wrapped_normal(
            r_t=r_t,
            mu_r_t=mu_r,
            sigma_r_t=sigma_r,
            K=self.wrapped_normal_K,
            eps=self.eps,
        )
        target = self.match_dims((1.0 - torch.exp(-t)) / (1.0 + torch.exp(-t)), r_t) * wrapped_normal_score

        #This should be done instead of
        #center(f_t) in the appendix. It is indeed a mistake in the original KLDM paper.
        target = scatter_center(target, index=index)

        # sigma_norm rescales the simplified target so the network.
        # This is done instead of lambda(t)*MSE. The idea
        # Is initally found in the torus diffusion paper (ref in theises).
        sigma_norm_t = self.sigma_norm_factor(t=t, index=index, ref=target)

        return target / self.match_dims(
            ((1.0 - torch.exp(-t)) / (1.0 + torch.exp(-t))).clamp_min(self.eps),
            target,
        ) / sigma_norm_t

    # -------------------------------------------------------------------------
    # Reverse-score helpers
    # -------------------------------------------------------------------------

    def reconstruct_full_reverse_velocity_score(
        self,
        t: torch.Tensor,
        v_t: torch.Tensor,
        pred_v: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct the full reverse velocity score used during sampling.

        During training, the network predicts only the simplified wrapped-normal
        score target from `build_simplified_training_velocity_score(...)`.

        For reverse sampling, KLDM needs the full velocity score:

            full score
                = wrapped-normal contribution
                + analytic Gaussian velocity contribution

        So this function:
            1. undoes the sigma_norm normalization
            2. restores the KLDM prefactor on the wrapped-normal term
            3. adds the closed-form Gaussian velocity score
        """

        t = self.T * t

        # Undo the sigma_norm normalization that was applied to the target
        # during training.
        sigma_norm_t = self.sigma_norm_factor(t=t, index=index, ref=pred_v)

        # gaussian_velocity_sigma(t) is the pre-scaling standard deviation, so
        # the actual velocity variance in the unit chart is
        #   (vel_scale * gaussian_velocity_sigma(t))^2
        #
        # This is the variance that appears in the analytic Gaussian score.
        gaussian_velocity_sigma_sq = self.match_dims(
            (self.vel_scale * self.gaussian_velocity_sigma(t)).pow(2),
            pred_v,
        ).clamp_min(self.eps)

        # Reconstruct the two pieces of the full KLDM score:
        #
        #   wrapped-normal piece:
        #       ((1 - exp(-t)) / (1 + exp(-t))) * sigma_norm_t * pred_v
        #
        #   Gaussian velocity piece:
        #       -v_t / sigma_v(t)^2
        #
        # Adding them gives the score used by the reverse sampler.
        return (
            self.match_dims((1.0 - torch.exp(-t)) / (1.0 + torch.exp(-t)), pred_v) * sigma_norm_t * pred_v
            - v_t / gaussian_velocity_sigma_sq
        )

    # -------------------------------------------------------------------------
    # Exponential integrator sampler steps
    # -------------------------------------------------------------------------

    def reverse_exp_step(
        self,
        f_t: torch.Tensor,
        v_t: torch.Tensor,
        score_v: torch.Tensor,
        index: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One reverse exponential-integrator step.

        Input:
            current f_t, current v_t, score_v, and step size dt.

        Output:
            previous f and v (an estimate of course)
        """
        f_t = self.wrap_displacements(f_t)

        dt = torch.as_tensor(self.T * dt, device=v_t.device, dtype=v_t.dtype)
        exp_dt = torch.exp(dt)
        expm1_dt = torch.expm1(dt)
        noise_scale = torch.sqrt(torch.expm1(2.0 * dt).clamp_min(self.eps))

        noise_v = self.sample_velocity_noise(v_t, index=index)

        # The forward velocity noise is scaled by vel_scale, so the reverse
        # score drift must carry vel_scale**2 as well. Otherwise the reverse
        # sampler becomes much too hot in the unit chart.
        score_scale = torch.as_tensor(self.vel_scale**2, device=v_t.device, dtype=v_t.dtype)

        v_prev = (
            exp_dt * v_t
            + 2.0 * score_scale * expm1_dt * score_v
            + noise_scale * noise_v
        )

        f_prev = self.wrap_displacements(f_t - dt * v_prev)
        return f_prev, v_prev

    # -------------------------------------------------------------------------
    # PC sampler steps
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def reverse_step_predictor(
        self,
        t: torch.Tensor,
        f_t: torch.Tensor,
        v_t: torch.Tensor,
        pred_v: torch.Tensor,
        index: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predictor step for Algorithm 4.

        Following Appendix H, the predictor is evaluated at time t_{n-1} and
        updates the velocity through

            v_n^pred = r * v_{n-1} + c * out_v^{(n-1)}

            f_n^pred = w(f_{n-1} + v_n^pred * dt)

        In our implementation, however, the network output `pred_v` is the
        normalized simplified target, not the effective appendix `out_v`.
        Therefore we first reconstruct the full reverse velocity score through
        Eq. (19) and then use that quantity in the predictor coefficients.

        Time-convention note:
            our sampler defines

                dt = t_n - t_{n+1} > 0

            while stepping backward in time.

            Facit writes the same backward move with a negative dt, so its
            position update appears as

                f <- w(f + dt * v)

            with dt < 0.

            Under our convention the equivalent update is therefore

                f <- w(f - dt * v)

            which is why the predictor uses the minus sign below.
        """
        t_now = self.T * t
        dt_internal = torch.as_tensor(self.T * dt, device=v_t.device, dtype=v_t.dtype)
        t_next = (t_now - dt_internal).clamp_min(self.eps)

        out_v = self.reconstruct_full_reverse_velocity_score(
            t=t,
            v_t=v_t,
            pred_v=pred_v,
            index=index,
        )

        # Closed-form Gaussian reverse coefficients for the velocity marginal.
        # This matches the appendix line:
        #
        #   r = alpha_v(n) / alpha_v(n - 1)
        #   c = (r sigma_v(n - 1) - sigma_v(n)) sigma_v(n - 1)
        r = self.gaussian_velocity_mean(t_next) / self.gaussian_velocity_mean(t_now)
        sigma_now = self.vel_scale * self.gaussian_velocity_sigma(t_now)
        sigma_next = self.vel_scale * self.gaussian_velocity_sigma(t_next)

        r = self.match_dims(r, v_t)
        sigma_now = self.match_dims(sigma_now, v_t)
        sigma_next = self.match_dims(sigma_next, v_t)

        c = (r * sigma_now - sigma_next) * sigma_now

        v_pred = r * v_t + c * out_v
        f_pred = self.wrap_displacements(f_t - dt_internal * v_pred)
        return f_pred, v_pred

    @torch.no_grad()
    def reverse_step_corrector(
        self,
        t: torch.Tensor,
        f_t: torch.Tensor,
        v_t: torch.Tensor,
        pred_v: torch.Tensor,
        dt: float,
        index: torch.Tensor,
        tau: float = 0.25,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Corrector step for Algorithm 4.

        In the appendix, after the predictor gives (f_n^pred, v_n^pred), the
        network is evaluated again at the new time level t_n and a single
        Langevin correction is applied:

            delta = tau * dim(out_v) / ||out_v||^2
            v_n = v_n^pred + delta * out_v + sqrt(2 delta) eps_v
            f_n = w(f_n^pred - v_n dt)

        Since mean(score^2) = ||score||^2 / dim, the implementation
        `tau / mean(score^2)` is exactly the same quantity.

        Important detail:
        the appendix uses a fresh Langevin noise

            eps_v ~ N_v(0, I)

        for the corrector step. In our scaled unit chart, the velocity variable
        itself already lives in the `vel_scale` coordinates, so the Langevin
        noise must be sampled in that same chart as well.
        """
        dt_internal = torch.as_tensor(self.T * dt, device=v_t.device, dtype=v_t.dtype)
        out_v = self.reconstruct_full_reverse_velocity_score(
            t=t,
            v_t=v_t,
            pred_v=pred_v,
            index=index,
        )

        num_graphs = int(index.max().item()) + 1

        score_power = torch.zeros((num_graphs, 1), device=v_t.device, dtype=v_t.dtype)
        counts = torch.zeros_like(score_power)

        score_power = score_power.index_add(
            0,
            index,
            out_v.square().mean(dim=-1, keepdim=True),
        )
        counts = counts.index_add(
            0,
            index,
            torch.ones((v_t.shape[0], 1), device=v_t.device, dtype=v_t.dtype),
        )

        # KLDM:
        #   delta = tau * dim(out_v) / ||out_v||^2
        #
        # Since score_power is mean(out_v^2), this is equivalent to:
        #   delta = tau / mean(out_v^2)
        score_power = (score_power / counts.clamp_min(1.0)).clamp_min(self.eps)
        delta = torch.as_tensor(tau, device=v_t.device, dtype=v_t.dtype) / score_power[index]

        eps_v = self.sample_velocity_noise(v_t, index=index)

        v_new = v_t + delta * out_v + torch.sqrt(2.0 * delta) * eps_v
        f_new = self.wrap_displacements(f_t - dt_internal * v_new)

        return f_new, v_new

    @staticmethod
    def match_dims(coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Unsqueeze coeff until it broadcasts with x."""
        while coeff.ndim < x.ndim:
            coeff = coeff.unsqueeze(-1)
        return coeff
