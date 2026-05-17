from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import sys

import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# Inspired by Yang Song's VP-SDE notation, but intentionally kept simple
# to fit the KLDM appendix.
# The paramization trick is covered in our theises..


class ContinuousDiffusion(nn.Module, ABC):
    """Abstract base class for continuous lattice diffusion helpers."""

    def __init__(
        self,
        *,
        eps: float = 1e-5,
        parameterization: str = "eps",
    ) -> None:
        super().__init__()
        if parameterization not in {"eps", "x0"}:
            raise ValueError("parameterization must be either 'eps' or 'x0'.")

        self.eps = float(eps)
        self.parameterization = parameterization

    @staticmethod
    def _match_dims(coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Expand batch-wise coefficients until they broadcast with `x`."""
        while coeff.ndim < x.ndim:
            coeff = coeff.unsqueeze(-1)
        return coeff

    def training_target(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        noise: torch.Tensor,
        num_atoms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the training target matching the configured parameterization."""
        del t, num_atoms
        if self.parameterization == "eps":
            return noise
        return x0

    def sample_prior(
        self,
        x_like: torch.Tensor,
        num_atoms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw the terminal-time prior used by the reverse sampler."""
        del num_atoms
        return torch.randn_like(x_like)

    @abstractmethod
    def forward_sample(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        noise: torch.Tensor | None = None,
        num_atoms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def reverse_step(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        pred: torch.Tensor,
        dt: float,
        num_atoms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class ContinuousVPDiffusion(ContinuousDiffusion):
    """Small VP diffusion helper for Euclidean KLDM modalities.

    Used for:
    - lattice parameters `l`
    - atom representations `a` in the continuous DNG variant

    We use the VP-SDE

        dx = f(x, t) dt + g(t) dW_t
        f(x, t) = -0.5 beta(t) x
        g(t) = sqrt(beta(t))

    with a linear beta schedule. Its closed-form forward kernel is

        x_t | x_0 ~ N(alpha(t) x_0, sigma(t)^2 I)
        x_t = alpha(t) x_0 + sigma(t) eps, eps ~ N(0, I)
    """

    def __init__(
        self,
        eps: float = 1e-5,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        parameterization: str = "eps",
    ) -> None:
        super().__init__(eps=eps, parameterization=parameterization)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Linear VP-SDE noise schedule beta(t)."""
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Mean coefficient of the forward kernel."""
        beta_integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t.pow(2)
        return torch.exp(-0.5 * beta_integral)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Standard deviation of the forward kernel."""
        alpha_t = self.alpha(t)
        return torch.sqrt(torch.clamp(1.0 - alpha_t.pow(2), min=self.eps))

    def forward_sample(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        noise: torch.Tensor | None = None,
        num_atoms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t from the transition kernel.

        Given x_0 and a time t, we draw

            x_t = alpha(t) x_0 + sigma(t) eps

        where eps ~ N(0, I).
        """
        del num_atoms
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_t = self._match_dims(self.alpha(t), x0)
        sigma_t = self._match_dims(self.sigma(t), x0)
        x_t = alpha_t * x0 + sigma_t * noise
        return x_t, noise

    def reverse_step(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        pred: torch.Tensor,
        dt: float,
        num_atoms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Paper-aligned lattice reverse step.

        The network output is interpreted according to `self.parameterization`:

        - eps head:
              score = -eps_theta / sigma(t)

        - x0 head:
              score = (alpha(t) x0_theta - x_t) / sigma(t)^2

        Then we apply reverse Euler-Maruyama for the VP-SDE:

            x_{t-dt} = x_t - [f(x_t,t) - g(t)^2 score] dt
                       + g(t) sqrt(dt) z

        where f(x,t) = -0.5 beta(t)x and g(t)^2 = beta(t). The lattice branch in
        KLDM Appendix Algorithm 3/4 uses this EM update, not a PC update.
        """
        del num_atoms
        dt_t = torch.as_tensor(dt, device=x_t.device, dtype=x_t.dtype)
        beta_t = self._match_dims(self.beta(t), x_t)
        sigma_t = self._match_dims(self.sigma(t), x_t)

        if self.parameterization == "eps":
            score_x = -pred / sigma_t.clamp_min(self.eps)
        else:
            alpha_t = self._match_dims(self.alpha(t), x_t)
            score_x = (alpha_t * pred - x_t) / sigma_t.pow(2).clamp_min(self.eps)

        noise = torch.randn_like(x_t)
        forward_drift = -0.5 * beta_t * x_t
        reverse_drift = forward_drift - beta_t * score_x
        x_prev = x_t - reverse_drift * dt_t
        x_prev = x_prev + torch.sqrt(beta_t * dt_t) * noise
        return x_prev

    @torch.no_grad()
    def reverse_step_predictor(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        pred: torch.Tensor,
        dt: float,
        num_atoms: torch.Tensor | None = None,
        **_,
    ) -> torch.Tensor:
        """
        FacitKLDM-style VP predictor step.

        This is used only for facit ablation samplers, where the continuous
        lattice branch participates in a predictor/corrector loop instead of
        the Appendix-H EM-only lattice update.
        """
        del num_atoms
        dt_t = torch.as_tensor(dt, device=x_t.device, dtype=x_t.dtype)
        alpha_curr = self._match_dims(self.alpha(t), x_t)
        sigma_curr = self._match_dims(self.sigma(t), x_t)
        alpha_next = self._match_dims(self.alpha(t - dt_t), x_t)
        sigma_next = self._match_dims(self.sigma(t - dt_t), x_t)

        if self.parameterization == "eps":
            score = -pred / sigma_curr.clamp_min(self.eps)
        else:
            score = (alpha_curr * pred - x_t) / sigma_curr.pow(2).clamp_min(self.eps)

        alpha_ratio = alpha_next / alpha_curr.clamp_min(self.eps)
        score_coeff = (alpha_ratio * sigma_curr - sigma_next) * sigma_curr
        return alpha_ratio * x_t + score_coeff * score

    @torch.no_grad()
    def reverse_step_corrector(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        pred: torch.Tensor,
        tau: float,
        index: torch.Tensor | None = None,
        num_atoms: torch.Tensor | None = None,
        **_,
    ) -> torch.Tensor:
        """
        FacitKLDM-style VP corrector step.
        """
        del num_atoms
        sigma_t = self._match_dims(self.sigma(t), x_t)
        if self.parameterization == "eps":
            score = -pred / sigma_t.clamp_min(self.eps)
        else:
            alpha_t = self._match_dims(self.alpha(t), x_t)
            score = (alpha_t * pred - x_t) / sigma_t.pow(2).clamp_min(self.eps)

        if index is None:
            denominator = score.square().mean(dim=-1, keepdim=True)
            delta = tau / denominator.clamp_min(self.eps)
        else:
            from torch_scatter import scatter_mean

            denominator = scatter_mean(
                score.square().mean(dim=-1, keepdim=True),
                dim=0,
                index=index,
            )
            delta = tau / denominator[index].clamp_min(self.eps)

        eps = torch.randn_like(x_t)
        return x_t + delta * score + torch.sqrt(2.0 * delta) * eps


class ContinuousMattergenVPDiffusion(ContinuousVPDiffusion):
    """MatterGen-specific VP diffusion entrypoint for lattice experiments.

    Important attribution note:
    this class is a KLDM-side port of the MatterGen lattice prior formulation.
    It is not labeled here as a line-for-line copy of a specific
    `microsoft/mattergen` source file.

    This class is intentionally narrow:
    - epsilon-parameterization only
    - Euler-Maruyama reverse sampling only

    It follows the MatterGen-style atom-count-aware prior:

        x_t = alpha(t) x_0 + (1-alpha(t)) mu(n) I
              + sqrt(1-alpha(t)^2) sigma(n) eps

    where
        mu(n)    = (n c)^(1/3)
        sigma(n) = (n nu)^(1/3)

    In the official MatterGen lattice SDE this corresponds to
        c  = 1 / limit_density
        nu = limit_var_scaling_constant^(3/2)

    KLDM-specific note:
    the official model diffuses a symmetric 3x3 matrix directly. This port
    keeps the same mean/variance formulas but applies them to the KLDM 6D
    vectorization of the symmetric lattice.
    """

    def __init__(
        self,
        *,
        c: float,
        nu: float,
        eps: float = 1e-5,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        parameterization: str = "eps",
    ) -> None:
        if parameterization != "eps":
            raise ValueError("ContinuousMattergenVPDiffusion only supports parameterization='eps'.")
        super().__init__(
            eps=eps,
            beta_min=beta_min,
            beta_max=beta_max,
            parameterization=parameterization,
        )
        self.register_buffer("c", torch.as_tensor(float(c), dtype=torch.get_default_dtype()))
        self.register_buffer("nu", torch.as_tensor(float(nu), dtype=torch.get_default_dtype()))
        self.register_buffer(
            "identity_vec6",
            torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.get_default_dtype()),
        )

    def training_target(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        noise: torch.Tensor,
        num_atoms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del t, x0, num_atoms
        return noise

    def sigma_base(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma(t)

    def mu_sigma_n(
        self,
        *,
        num_atoms: torch.Tensor,
        ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = num_atoms.to(device=ref.device, dtype=ref.dtype).reshape(-1).clamp_min(1.0)
        c = self.c.to(device=ref.device, dtype=ref.dtype).clamp_min(self.eps)
        nu = self.nu.to(device=ref.device, dtype=ref.dtype).clamp_min(self.eps)
        mu_n = torch.pow(n * c, 1.0 / 3.0)
        sigma_n = torch.pow(n * nu, 1.0 / 3.0)
        return mu_n, sigma_n

    def prior_mean(self, *, num_atoms: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        mu_n, _ = self.mu_sigma_n(num_atoms=num_atoms, ref=ref)
        identity = self.identity_vec6.to(device=ref.device, dtype=ref.dtype)
        return mu_n[:, None] * identity[None, :]

    def sample_prior(
        self,
        x_like: torch.Tensor,
        num_atoms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_atoms is None:
            raise ValueError("num_atoms is required for MatterGen lattice prior sampling.")
        # Code segment inspired from mattergen
        # (mattergen/common/diffusion/corruption.py:168-182).
        mu_vec = self.prior_mean(num_atoms=num_atoms, ref=x_like)
        _, sigma_n = self.mu_sigma_n(num_atoms=num_atoms, ref=x_like)
        sigma_n = self._match_dims(sigma_n, x_like)
        return mu_vec + sigma_n * torch.randn_like(x_like)

    def forward_sample(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        noise: torch.Tensor | None = None,
        num_atoms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Code segment inspired from mattergen
        # (mattergen/common/diffusion/corruption.py:80-96,
        #  mattergen/common/diffusion/corruption.py:110-152).
        # KLDM port of the MatterGen lattice forward-kernel equations.
        # This is intentionally marked as a port, not as a 1:1 repo copy.
        if num_atoms is None:
            raise ValueError("num_atoms is required for MatterGen lattice forward sampling.")
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_t = self._match_dims(self.alpha(t), x0)
        sigma_base_t = self._match_dims(self.sigma_base(t), x0)
        mu_vec = self.prior_mean(num_atoms=num_atoms, ref=x0)
        _, sigma_n = self.mu_sigma_n(num_atoms=num_atoms, ref=x0)
        sigma_n = self._match_dims(sigma_n, x0)
        x_t = alpha_t * x0 + (1.0 - alpha_t) * mu_vec + sigma_base_t * sigma_n * noise
        return x_t, noise

    def reverse_step(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        pred: torch.Tensor,
        dt: float,
        num_atoms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Code segment inspired from mattergen
        # (mattergen/common/diffusion/corruption.py:184-212,
        #  mattergen/common/diffusion/predictors_correctors.py:40-52).
        # KLDM port of the MatterGen lattice reverse-step equations.
        # This is intentionally marked as a port, not as a 1:1 repo copy.
        if num_atoms is None:
            raise ValueError("num_atoms is required for MatterGen lattice reverse sampling.")
        dt_t = torch.as_tensor(dt, device=x_t.device, dtype=x_t.dtype)
        beta_t = self._match_dims(self.beta(t), x_t)
        alpha_t = self._match_dims(self.alpha(t), x_t)
        sigma_base_t = self._match_dims(self.sigma_base(t), x_t)
        mu_vec = self.prior_mean(num_atoms=num_atoms, ref=x_t)
        _, sigma_n = self.mu_sigma_n(num_atoms=num_atoms, ref=x_t)
        sigma_n = self._match_dims(sigma_n, x_t)
        score_x = -pred / (sigma_base_t * sigma_n).clamp_min(self.eps)

        noise = torch.randn_like(x_t)
        forward_drift = -0.5 * beta_t * (x_t - mu_vec)
        reverse_drift = forward_drift - beta_t * sigma_n.pow(2) * score_x
        return x_t - reverse_drift * dt_t + sigma_n * torch.sqrt(beta_t * dt_t).clamp_min(self.eps) * noise

    def reverse_step_ancestral(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        pred: torch.Tensor,
        dt: float,
        num_atoms: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Code segment inspired from mattergen
        # (mattergen/common/diffusion/predictors_correctors.py:16-53,
        #  mattergen/diffusion/sampling/predictors.py:122-165).
        #
        # MatterGen samples the lattice with an ancestral predictor, not the
        # generic Euler-Maruyama reverse-SDE step. `pred` is our eps target
        # (+raw noise), so convert it to a score with the corresponding sign.
        if num_atoms is None:
            raise ValueError("num_atoms is required for MatterGen lattice ancestral sampling.")

        dt_t = torch.as_tensor(dt, device=x_t.device, dtype=x_t.dtype)
        t_ref = t.to(device=x_t.device, dtype=x_t.dtype)
        s = (t_ref - dt_t).clamp_min(0.0)

        alpha_t = self._match_dims(self.alpha(t_ref), x_t)
        alpha_s = self._match_dims(self.alpha(s), x_t)
        sigma_base_t = self._match_dims(self.sigma_base(t_ref), x_t)
        sigma_base_s = self._match_dims(self.sigma_base(s), x_t)

        mu_vec = self.prior_mean(num_atoms=num_atoms, ref=x_t)
        _, sigma_n = self.mu_sigma_n(num_atoms=num_atoms, ref=x_t)
        sigma_n = self._match_dims(sigma_n, x_t)

        sigma_t = sigma_base_t * sigma_n
        sigma_s = sigma_base_s * sigma_n
        score_x = -pred / sigma_t.clamp_min(self.eps)

        alpha_t_given_s = (alpha_t / alpha_s.clamp_min(self.eps)).clamp_min(1e-3)
        sigma2_t_given_s = sigma_t.pow(2) - sigma_s.pow(2) * alpha_t.pow(2) / alpha_s.pow(2).clamp_min(self.eps)
        sigma2_t_given_s = sigma2_t_given_s.clamp_min(0.0)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        x_coeff = 1.0 / alpha_t_given_s
        score_coeff = sigma2_t_given_s / alpha_t_given_s
        mean_coeff = 1.0 - x_coeff
        mean = x_coeff * x_t + score_coeff * score_x + mean_coeff * mu_vec

        std = sigma_t_given_s * sigma_s / sigma_t.clamp_min(self.eps)
        if noise is None:
            noise = torch.randn_like(x_t)
        return mean + std * noise
