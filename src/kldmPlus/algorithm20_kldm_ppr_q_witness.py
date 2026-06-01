from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from kldmPlus.algorithm19_kldm_ppr_diffcsppp import (
    Algorithm19State,
    _get_wyckoff_dof_chart,
    c_w_ops,
    graph_mean_norm,
    kldm_clean_fractional_denoiser_Df,
    kldm_ppr_noise_chart,
    kldm_renoise_from_f0,
    map_model_to_payload_reference_chart,
    map_payload_reference_chart_to_model_frame,
    torus_rmse,
    wrap01,
    wrapdiff,
)
from kldmPlus.symmetry import DiffCSPPPSymmetryPayload


ALGORITHM20_MODE = "kldm_ppr_q_witness"
ALGORITHM20_SHORT_NAME = "Algorithm20-KLDM-PPR-QWitness"
ALGORITHM20_DESCRIPTION = (
    "KLDM-PPR with joint optimization over KLDM noise variables and Wyckoff "
    "free variables q, using a smooth torus witness loss against the fixed "
    "template decoder Phi_T(q)."
)


@dataclass(frozen=True)
class Algorithm20Config:
    M: int = 2
    proj_steps: int = 100
    lr: float = 1.0e-2
    lambda_noise: float = 1.0e-4
    lambda_floor: float = 1.0e-6
    grad_clip: float = 10.0
    anchor_mode: str = "soft"
    denoiser_variant: str = "minus"
    coordinate_score_mode: str = "direct"
    soft_anchor_tol: float = 1.0e-5
    q_init_mode: str = "random"
    q_only_steps: int = 100


@dataclass(frozen=True)
class Algorithm20ProjectResult:
    f_star: torch.Tensor
    v_star: torch.Tensor
    f0_star: torch.Tensor
    q_star: torch.Tensor
    z_proj_payload: torch.Tensor
    witness_sin: float
    witness_rmse_payload: float
    lambda_eff: float
    logs: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Algorithm20KernelResult:
    state: Algorithm19State
    q_star: torch.Tensor | None
    logs: tuple[dict[str, Any], ...]


def witness_torus_sin_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = wrapdiff(source, target)
    return torch.sin(torch.pi * diff).square().mean()


def _tdm_lambda_eff(
    *,
    model,
    t_nodes: torch.Tensor,
    ref_f: torch.Tensor,
    ref_v: torch.Tensor,
    lambda0: float,
    lambda_floor: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    tau = model.tdm.T * t_nodes
    sigma_r = model.tdm.match_dims(model.tdm.wrapped_gaussian_sigma_r_t(tau), ref_f)
    sigma_v = model.tdm.match_dims(model.tdm.vel_scale * model.tdm.gaussian_velocity_sigma(tau), ref_v)
    sigma_r_rms = torch.sqrt(sigma_r.square().mean())
    sigma_v_rms = torch.sqrt(sigma_v.square().mean())
    sigma_eff = torch.sqrt(0.5 * (sigma_r_rms.square() + sigma_v_rms.square()))
    t_mean = t_nodes.float().mean()
    lambda_eff = float(lambda0) * (t_mean.square() / (4.0 * sigma_eff.square() + 4.0))
    lambda_eff = torch.clamp(lambda_eff, min=float(lambda_floor))
    return lambda_eff, {
        "t_mean": float(t_mean.detach().item()),
        "sigma_r_rms": float(sigma_r_rms.detach().item()),
        "sigma_v_rms": float(sigma_v_rms.detach().item()),
        "sigma_eff": float(sigma_eff.detach().item()),
    }


def _initialize_q_raw(
    *,
    chart,
    device: torch.device,
    dtype: torch.dtype,
    q_init: torch.Tensor | None,
    q_init_mode: str,
) -> torch.Tensor:
    total_dof = int(chart.total_dof)
    if total_dof == 0:
        q_raw = torch.empty((0,), device=device, dtype=dtype)
        q_raw.requires_grad_(True)
        return q_raw
    if q_init is not None:
        q_raw = q_init.detach().clone().to(device=device, dtype=dtype).reshape(-1)
        q_raw.requires_grad_(True)
        return q_raw

    mode = str(q_init_mode).strip().lower()
    if mode in {"random", "rand"}:
        return torch.rand(total_dof, device=device, dtype=dtype, requires_grad=True)
    if mode in {"zero", "zeros"}:
        q_raw = torch.zeros(total_dof, device=device, dtype=dtype)
        q_raw.requires_grad_(True)
        return q_raw
    if mode in {"chart_q_ref", "bootstrap", "oracle_structure"}:
        q_raw = torch.as_tensor(chart.q_ref, device=device, dtype=dtype).reshape(-1)
        q_raw.requires_grad_(True)
        return q_raw
    raise ValueError(f"Unsupported q_init_mode={q_init_mode!r}.")


def q_only_witness_fit(
    *,
    z_payload: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
    q_init: torch.Tensor | None = None,
    q_init_mode: str = "random",
    steps: int = 100,
    lr: float = 1.0e-2,
    grad_clip: float = 10.0,
) -> dict[str, Any]:
    z_payload = z_payload.detach().clone()
    chart = _get_wyckoff_dof_chart(payload)
    q_raw = _initialize_q_raw(
        chart=chart,
        device=z_payload.device,
        dtype=z_payload.dtype,
        q_init=q_init,
        q_init_mode=q_init_mode,
    )
    optimizer = torch.optim.LBFGS(
        [q_raw],
        lr=1.0,
        max_iter=1,
        history_size=10,
        line_search_fn="strong_wolfe",
    )
    logs: list[dict[str, Any]] = []

    for step_idx in range(max(int(steps), 1)):
        loss_holder: dict[str, torch.Tensor] = {}

        def closure():
            optimizer.zero_grad(set_to_none=True)
            q = torch.remainder(q_raw, 1.0)
            z_sym = chart.expand_q(q, device=z_payload.device, dtype=z_payload.dtype)
            loss = witness_torus_sin_loss(z_payload, z_sym)
            loss.backward()
            loss_holder["loss"] = loss.detach()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            q_now = torch.remainder(q_raw, 1.0)
            z_now = chart.expand_q(q_now, device=z_payload.device, dtype=z_payload.dtype)
            q_grad_norm = float(torch.linalg.norm(q_raw.grad.detach()).item()) if q_raw.grad is not None else 0.0
            logs.append(
                {
                    "step": int(step_idx),
                    "q_only_witness_loss": float(loss_holder["loss"].item()),
                    "q_only_witness_rmse": float(torus_rmse(z_payload, z_now).detach().item()),
                    "q_grad_norm": q_grad_norm,
                    "q_norm": float(torch.linalg.norm(q_now).detach().item()) if q_now.numel() else 0.0,
                }
            )

    with torch.no_grad():
        q_star = torch.remainder(q_raw, 1.0).detach().clone()
        z_proj = chart.expand_q(q_star, device=z_payload.device, dtype=z_payload.dtype)
        return {
            "q_star": q_star,
            "z_proj_payload": z_proj.detach().clone(),
            "witness_sin": float(witness_torus_sin_loss(z_payload, z_proj).detach().item()),
            "witness_rmse_payload": float(torus_rmse(z_payload, z_proj).detach().item()),
            "logs": tuple(logs),
        }


def ppr_project_step_q_witness(
    *,
    state: Algorithm19State,
    payload: DiffCSPPPSymmetryPayload,
    model,
    config: Algorithm20Config,
    q_init: torch.Tensor | None = None,
) -> Algorithm20ProjectResult:
    chart = _get_wyckoff_dof_chart(payload)
    xi_r = torch.zeros_like(state.f, requires_grad=True)
    xi_v = torch.zeros_like(state.v, requires_grad=True)
    q_raw = _initialize_q_raw(
        chart=chart,
        device=state.f.device,
        dtype=state.f.dtype,
        q_init=q_init,
        q_init_mode=config.q_init_mode,
    )

    optimizer = torch.optim.LBFGS(
        [xi_r, xi_v, q_raw],
        lr=1.0,
        max_iter=1,
        history_size=10,
        line_search_fn="strong_wolfe",
    )
    logs: list[dict[str, Any]] = []

    params = list(model.parameters()) if hasattr(model, "parameters") else []
    old_requires_grad = [bool(p.requires_grad) for p in params]
    was_training = bool(model.training)
    model.eval()
    for p in params:
        p.requires_grad_(False)

    try:
        for step_idx in range(max(int(config.proj_steps), 1)):
            step_cache: dict[str, Any] = {}

            def closure():
                optimizer.zero_grad(set_to_none=True)
                q = torch.remainder(q_raw, 1.0)
                step_cache["q_before"] = q.detach()
                f_var, v_var = kldm_ppr_noise_chart(
                    model=model,
                    f_t=state.f,
                    v_t=state.v,
                    xi_r=xi_r,
                    xi_v=xi_v,
                    t_nodes=state.t_nodes,
                    node_index=state.node_index,
                )
                f0_hat = kldm_clean_fractional_denoiser_Df(
                    model=model,
                    f=f_var,
                    v=v_var,
                    l=state.l,
                    atom_types=state.atom_types,
                    t_graph=state.t_graph,
                    t_nodes=state.t_nodes,
                    node_index=state.node_index,
                    edge_index=state.edge_node_index,
                    variant=config.denoiser_variant,
                    coordinate_score_mode=config.coordinate_score_mode,
                )

                z_payload = map_model_to_payload_reference_chart(f0_hat, payload)
                z_sym = chart.expand_q(q, device=state.f.device, dtype=state.f.dtype)
                c_witness = witness_torus_sin_loss(z_payload, z_sym)
                lambda_eff, lambda_stats = _tdm_lambda_eff(
                    model=model,
                    t_nodes=state.t_nodes,
                    ref_f=state.f,
                    ref_v=state.v,
                    lambda0=float(config.lambda_noise),
                    lambda_floor=float(config.lambda_floor),
                )
                prox = xi_r.square().mean() + xi_v.square().mean()
                loss = c_witness + lambda_eff * prox
                loss.backward()
                step_cache["loss"] = loss.detach()
                step_cache["c_witness"] = c_witness.detach()
                step_cache["prox"] = prox.detach()
                step_cache["lambda_eff"] = lambda_eff.detach()
                step_cache["lambda_stats"] = lambda_stats
                step_cache["f_var"] = f_var.detach()
                step_cache["v_var"] = v_var.detach()
                step_cache["z_payload"] = z_payload.detach()
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                q_now = torch.remainder(q_raw, 1.0)
                z_sym_now = chart.expand_q(q_now, device=state.f.device, dtype=state.f.dtype)
                q_grad_norm = float(torch.linalg.norm(q_raw.grad.detach()).item()) if q_raw.grad is not None else 0.0
                logs.append(
                    {
                        "step": int(step_idx),
                        "loss": float(step_cache["loss"].item()),
                        "c_before_witness_sin": float(step_cache["c_witness"].item()),
                        "witness_rmse_payload": float(torus_rmse(step_cache["z_payload"], z_sym_now).detach().item()),
                        "prox": float(step_cache["prox"].item()),
                        "lambda_eff": float(step_cache["lambda_eff"].item()),
                        "t_mean": float(step_cache["lambda_stats"]["t_mean"]),
                        "sigma_eff": float(step_cache["lambda_stats"]["sigma_eff"]),
                        "xi_r_norm": float(torch.sqrt(xi_r.detach().square().mean()).item()),
                        "xi_v_norm": float(torch.sqrt(xi_v.detach().square().mean()).item()),
                        "q_norm": float(torch.linalg.norm(q_now).detach().item()) if q_now.numel() else 0.0,
                        "q_step_norm": float(torch.linalg.norm(wrapdiff(q_now, step_cache["q_before"])).detach().item()) if q_now.numel() else 0.0,
                        "q_grad_norm": q_grad_norm,
                        "velocity_mean_norm": graph_mean_norm(step_cache["v_var"], state.node_index),
                    }
                )

        with torch.no_grad():
            q_star = torch.remainder(q_raw, 1.0).detach().clone()
            f_star, v_star = kldm_ppr_noise_chart(
                model=model,
                f_t=state.f,
                v_t=state.v,
                xi_r=xi_r,
                xi_v=xi_v,
                t_nodes=state.t_nodes,
                node_index=state.node_index,
            )
            f0_star = kldm_clean_fractional_denoiser_Df(
                model=model,
                f=f_star,
                v=v_star,
                l=state.l,
                atom_types=state.atom_types,
                t_graph=state.t_graph,
                t_nodes=state.t_nodes,
                node_index=state.node_index,
                edge_index=state.edge_node_index,
                variant=config.denoiser_variant,
                coordinate_score_mode=config.coordinate_score_mode,
            )
            z_payload_star = map_model_to_payload_reference_chart(f0_star, payload)
            z_proj_payload = chart.expand_q(q_star, device=state.f.device, dtype=state.f.dtype)
            witness_sin = float(witness_torus_sin_loss(z_payload_star, z_proj_payload).detach().item())
            witness_rmse_payload = float(torus_rmse(z_payload_star, z_proj_payload).detach().item())
            lambda_eff_final, _lambda_stats = _tdm_lambda_eff(
                model=model,
                t_nodes=state.t_nodes,
                ref_f=state.f,
                ref_v=state.v,
                lambda0=float(config.lambda_noise),
                lambda_floor=float(config.lambda_floor),
            )
    finally:
        for p, req in zip(params, old_requires_grad):
            p.requires_grad_(req)
        if was_training:
            model.train()

    return Algorithm20ProjectResult(
        f_star=f_star.detach().clone(),
        v_star=v_star.detach().clone(),
        f0_star=f0_star.detach().clone(),
        q_star=q_star,
        z_proj_payload=z_proj_payload.detach().clone(),
        witness_sin=witness_sin,
        witness_rmse_payload=witness_rmse_payload,
        lambda_eff=float(lambda_eff_final.detach().item()),
        logs=tuple(logs),
    )


def ppr_kernel_q_witness(
    *,
    state: Algorithm19State,
    payload: DiffCSPPPSymmetryPayload,
    model,
    config: Algorithm20Config,
    q_init: torch.Tensor | None = None,
    epsilon_sequence: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
) -> Algorithm20KernelResult:
    current = Algorithm19State(
        f=state.f.detach().clone(),
        v=state.v.detach().clone(),
        l=state.l.detach().clone(),
        atom_types=state.atom_types.detach().clone(),
        node_index=state.node_index.detach().clone(),
        edge_node_index=state.edge_node_index.detach().clone(),
        t_graph=state.t_graph.detach().clone(),
        t_nodes=state.t_nodes.detach().clone(),
    )
    all_logs: list[dict[str, Any]] = []
    q_live = None if q_init is None else q_init.detach().clone()

    for repeat_idx in range(max(int(config.M), 1)):
        q_prev = None if q_live is None else q_live.detach().clone()
        project = ppr_project_step_q_witness(
            state=current,
            payload=payload,
            model=model,
            config=config,
            q_init=q_live,
        )

        z_payload_star = map_model_to_payload_reference_chart(project.f0_star, payload)
        witness_rmse = float(torus_rmse(z_payload_star, project.z_proj_payload).detach().item())
        witness_sin = float(witness_torus_sin_loss(z_payload_star, project.z_proj_payload).detach().item())
        soft_anchor_feasible = bool(witness_sin < float(config.soft_anchor_tol))

        if str(config.anchor_mode).strip().lower() == "soft":
            f0_anchor = project.f0_star
        elif str(config.anchor_mode).strip().lower() == "hard":
            f0_anchor = map_payload_reference_chart_to_model_frame(project.z_proj_payload, payload)
        else:
            raise ValueError(f"Unsupported anchor_mode={config.anchor_mode!r}.")

        eps_v = eps_r = None
        if epsilon_sequence is not None and repeat_idx < len(epsilon_sequence):
            epsilon_pair = epsilon_sequence[repeat_idx]
            if epsilon_pair is not None:
                eps_v, eps_r = epsilon_pair

        if eps_v is None and eps_r is None:
            f_new, v_new, epsilon_v, epsilon_r, r_t = kldm_renoise_from_f0(
                model=model,
                f0_star=f0_anchor,
                t_nodes=current.t_nodes,
                node_index=current.node_index,
            )
        else:
            f_new, v_new, epsilon_v, epsilon_r, r_t = model.tdm.sample_noisy_state(
                t=current.t_nodes,
                f0=f0_anchor,
                index=current.node_index,
                epsilon_v=eps_v,
                epsilon_r=eps_r,
            )
        current = replace(
            current,
            f=f_new.detach().clone(),
            v=v_new.detach().clone(),
            l=current.l.detach().clone(),
        )
        q_live = project.q_star.detach().clone()
        all_logs.append(
            {
                "repeat_idx": int(repeat_idx),
                "c_after_witness_sin": witness_sin,
                "witness_rmse_payload": witness_rmse,
                "soft_anchor_feasible": soft_anchor_feasible,
                "accepted": soft_anchor_feasible,
                "q_updated": True,
                "q_norm": float(torch.linalg.norm(project.q_star).detach().item()) if project.q_star.numel() else 0.0,
                "q_step_norm": (
                    float(torch.linalg.norm(wrapdiff(project.q_star, q_prev)).detach().item())
                    if (q_prev is not None and project.q_star.numel()) else 0.0
                ),
                "ppr_faithfulness": (
                    "soft_ppr_feasible" if soft_anchor_feasible else "soft_ppr_not_yet_feasible"
                ),
                "project_logs": list(project.logs),
                "epsilon_v_rms": float(torch.sqrt(epsilon_v.detach().square().mean()).item()),
                "epsilon_r_rms": float(torch.sqrt(epsilon_r.detach().square().mean()).item()),
                "r_t_rms": float(torch.sqrt(r_t.detach().square().mean()).item()),
            }
        )
        q_init = project.q_star.detach().clone()

    return Algorithm20KernelResult(
        state=current,
        q_star=q_live,
        logs=tuple(all_logs),
    )


__all__ = [
    "ALGORITHM20_MODE",
    "ALGORITHM20_SHORT_NAME",
    "ALGORITHM20_DESCRIPTION",
    "Algorithm20Config",
    "Algorithm20ProjectResult",
    "Algorithm20KernelResult",
    "q_only_witness_fit",
    "witness_torus_sin_loss",
    "ppr_project_step_q_witness",
    "ppr_kernel_q_witness",
]
