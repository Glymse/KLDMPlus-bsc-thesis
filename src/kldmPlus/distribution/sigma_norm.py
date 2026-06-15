from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterator, Literal

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from kldmPlus.distribution.wrapped_normal import d_log_wrapped_normal
else:
    from .wrapped_normal import d_log_wrapped_normal


def chunk_ranges(total_size: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, total_size, chunk_size):
        yield start, min(start + chunk_size, total_size)


def wrap_signed_unit(x: torch.Tensor) -> torch.Tensor:
    # Unit-period wrap to [-0.5, 0.5).
    return torch.remainder(x + 0.5, 1.0) - 0.5


def sample_wrapped_zero_mean(
    sigma: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    # epsilon ~ N(0, 1), r = wrap(sigma * epsilon), so r ~ WN(0, sigma^2).
    sigma = sigma.reshape(-1)
    epsilon = torch.randn(
        (num_samples, sigma.numel()),
        device=sigma.device,
        dtype=sigma.dtype,
    )
    return wrap_signed_unit(sigma.unsqueeze(0) * epsilon)


def wrapped_normal_zero_mean_score(
    r: torch.Tensor,
    sigma: torch.Tensor,
    *,
    K: int,
    eps: float,
) -> torch.Tensor:
    # s_K(r; 0, sigma) = ∇_mu log WN_K(r | mu=0, sigma^2).
    sigma = sigma.reshape(-1).clamp_min(eps)
    sigma_batch = sigma.unsqueeze(0).expand_as(r)
    return d_log_wrapped_normal(
        r_t=r,
        mu_r_t=torch.zeros_like(r),
        sigma_r_t=sigma_batch,
        K=K,
        eps=eps,
    )


def estimate_score_square_for_sigma_batch(
    sigma: torch.Tensor,
    *,
    num_samples: int,
    K: int,
    eps: float,
) -> torch.Tensor:
    # Monte Carlo estimate of E_{r ~ WN(0, sigma^2)}[s_K(r; 0, sigma)^2].
    sigma = sigma.reshape(-1).clamp_min(eps)
    r = sample_wrapped_zero_mean(sigma=sigma, num_samples=num_samples)
    score = wrapped_normal_zero_mean_score(r=r, sigma=sigma, K=K, eps=eps)
    return score.square().mean(dim=0)


@dataclass
class WrappedNormalSigmaNormMC:
    K: int = 3
    num_monte_carlo_samples: int = 20000
    sigma_batch_size: int = 128
    sample_batch_size: int = 2048
    eps: float = 1e-8

    def __call__(self, sigma_grid: torch.Tensor) -> torch.Tensor:
        sigma_grid = sigma_grid.reshape(-1).clamp_min(self.eps)
        score_norm_grid = torch.empty_like(sigma_grid)

        with torch.no_grad():
            for sigma_start, sigma_end in chunk_ranges(
                sigma_grid.numel(),
                self.sigma_batch_size,
            ):
                sigma = sigma_grid[sigma_start:sigma_end]
                score_square_sum = torch.zeros_like(sigma)
                num_used_samples = 0

                for sample_start, sample_end in chunk_ranges(
                    self.num_monte_carlo_samples,
                    self.sample_batch_size,
                ):
                    num_samples = sample_end - sample_start
                    score_square_mean = estimate_score_square_for_sigma_batch(
                        sigma=sigma,
                        num_samples=num_samples,
                        K=self.K,
                        eps=self.eps,
                    )
                    score_square_sum += score_square_mean * float(num_samples)
                    num_used_samples += num_samples

                score_norm_grid[sigma_start:sigma_end] = (
                    score_square_sum / float(num_used_samples)
                )

        return score_norm_grid


@dataclass
class WrappedNormalSigmaNormQuadrature:
    K_score: int = 3
    K_density: int = 40
    num_grid_points: int = 8193
    sigma_batch_size: int = 512
    # Small-sigma switch tuned from the Gaussian-limit quadrature error:
    # once sigma is about one grid spacing, plain trapz is already accurate.
    sigma_switch_factor: float = 1.0
    eps: float = 1e-8

    def __call__(self, sigma_grid: torch.Tensor) -> torch.Tensor:
        sigma_grid = sigma_grid.reshape(-1).clamp_min(self.eps)
        score_norm_grid = torch.empty_like(sigma_grid)

        r_grid = torch.linspace(
            -0.5,
            0.5,
            self.num_grid_points,
            device=sigma_grid.device,
            dtype=sigma_grid.dtype,
        )
        r = r_grid[:, None]
        dr = (r_grid[1:] - r_grid[:-1]).clamp_min(self.eps)
        sigma_switch = self.sigma_switch_factor * dr[0]
        k_density = torch.arange(
            -self.K_density,
            self.K_density + 1,
            device=sigma_grid.device,
            dtype=sigma_grid.dtype,
        )

        with torch.no_grad():
            for sigma_start, sigma_end in chunk_ranges(
                sigma_grid.numel(),
                self.sigma_batch_size,
            ):
                sigma = sigma_grid[sigma_start:sigma_end]
                sigma_b = sigma[None, :]
                sigma2_b = sigma_b.square().clamp_min(self.eps)

                r_plus_k_density = r.unsqueeze(-1) + k_density
                exp_density = torch.exp(
                    -r_plus_k_density.square() / (2.0 * sigma2_b.unsqueeze(-1))
                )
                q_density = exp_density.sum(dim=-1).clamp_min(self.eps)

                score = d_log_wrapped_normal(
                    r_t=r,
                    mu_r_t=torch.zeros_like(r),
                    sigma_r_t=sigma_b,
                    K=self.K_score,
                    eps=self.eps,
                )

                normalizer = torch.trapz(q_density, r_grid, dim=0).clamp_min(self.eps)
                weighted_score_square = torch.trapz(
                    score.square() * q_density,
                    r_grid,
                    dim=0,
                )
                quad_value = weighted_score_square / normalizer
                asymptotic_value = 1.0 / sigma.square().clamp_min(self.eps)
                use_asymptotic = sigma < sigma_switch
                score_norm_grid[sigma_start:sigma_end] = torch.where(
                    use_asymptotic,
                    asymptotic_value,
                    quad_value,
                )

        return score_norm_grid


@dataclass
class WrappedNormalSigmaNorm:
    # Backwards-compatible wrapper used by KLDM+ TDM code.
    K: int = 3
    estimator: Literal["quadrature", "mc"] = "quadrature"
    K_density: int | None = None
    num_grid_points: int = 8193
    sigma_batch_size: int = 512
    num_monte_carlo_samples: int = 20000
    sample_batch_size: int = 2048
    sigma_switch_factor: float = 1.0
    eps: float = 1e-8

    def __call__(self, sigma_grid: torch.Tensor) -> torch.Tensor:
        if self.estimator == "mc":
            return WrappedNormalSigmaNormMC(
                K=self.K,
                num_monte_carlo_samples=self.num_monte_carlo_samples,
                sigma_batch_size=self.sigma_batch_size,
                sample_batch_size=self.sample_batch_size,
                eps=self.eps,
            )(sigma_grid)

        K_density = max(self.K_density or 0, self.K, 40)
        return WrappedNormalSigmaNormQuadrature(
            K_score=self.K,
            K_density=K_density,
            num_grid_points=self.num_grid_points,
            sigma_batch_size=self.sigma_batch_size,
            sigma_switch_factor=self.sigma_switch_factor,
            eps=self.eps,
        )(sigma_grid)


@torch.no_grad()
def compare_sigma_norm_estimators(
    sigma_grid: torch.Tensor,
    *,
    K_score: int = 3,
    K_density: int = 40,
    mc_samples: int = 20000,
    grid_points: int = 8193,
    sigma_switch_factor: float = 1.0,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    mc = WrappedNormalSigmaNormMC(
        K=K_score,
        num_monte_carlo_samples=mc_samples,
        eps=eps,
    )(sigma_grid)

    quad = WrappedNormalSigmaNormQuadrature(
        K_score=K_score,
        K_density=K_density,
        num_grid_points=grid_points,
        sigma_switch_factor=sigma_switch_factor,
        eps=eps,
    )(sigma_grid)

    rel_error = (quad - mc).abs() / mc.clamp_min(eps)
    return {
        "mc": mc,
        "quad": quad,
        "absolute_error": (quad - mc).abs(),
        "relative_error": rel_error,
        "relative_error_mean": rel_error.mean(),
        "relative_error_max": rel_error.max(),
    }


def run_sigma_norm_experiment(
    *,
    K_score: int = 3,
    K_density: int = 40,
    mc_samples: int = 20000,
    grid_points: int = 8193,
    sigma_switch_factor: float = 1.0,
    num_sigmas: int = 12,
    sigma_min: float = 1e-4,
    sigma_max: float = 0.25,
    seed: int = 0,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    # Runs a small MC-vs-quadrature comparison and prints a compact summary.
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    sigma_grid = torch.linspace(
        sigma_min,
        sigma_max,
        num_sigmas,
        device=resolved_device,
        dtype=dtype,
    )

    stats = compare_sigma_norm_estimators(
        sigma_grid=sigma_grid,
        K_score=K_score,
        K_density=K_density,
        mc_samples=mc_samples,
        grid_points=grid_points,
        sigma_switch_factor=sigma_switch_factor,
        eps=eps,
    )

    print("sigma_norm experiment")
    print(
        f"device={resolved_device} dtype={dtype} "
        f"K_score={K_score} K_density={K_density} "
        f"mc_samples={mc_samples} grid_points={grid_points}"
    )
    r_grid = torch.linspace(-0.5, 0.5, grid_points, device=resolved_device, dtype=dtype)
    dr = torch.diff(r_grid).clamp_min(eps)
    print(
        f"grid_spacing={float(dr[0]):.6e} "
        f"sigma_switch={sigma_switch_factor * float(dr[0]):.6e}"
    )
    print(
        "mean_relative_error="
        f"{float(stats['relative_error_mean'].item()):.6e} "
        "max_relative_error="
        f"{float(stats['relative_error_max'].item()):.6e}"
    )
    print("")
    print(
        f"{'sigma':>10} {'mc':>14} {'quad':>14} "
        f"{'abs_err':>14} {'rel_err':>14}"
    )

    sigma_cpu = sigma_grid.detach().cpu()
    mc_cpu = stats["mc"].detach().cpu()
    quad_cpu = stats["quad"].detach().cpu()
    abs_err_cpu = stats["absolute_error"].detach().cpu()
    rel_err_cpu = stats["relative_error"].detach().cpu()

    for sigma, mc_value, quad_value, abs_err, rel_err in zip(
        sigma_cpu,
        mc_cpu,
        quad_cpu,
        abs_err_cpu,
        rel_err_cpu,
    ):
        print(
            f"{float(sigma):10.6f} "
            f"{float(mc_value):14.6e} "
            f"{float(quad_value):14.6e} "
            f"{float(abs_err):14.6e} "
            f"{float(rel_err):14.6e}"
        )

    return stats


if __name__ == "__main__":
    run_sigma_norm_experiment()
