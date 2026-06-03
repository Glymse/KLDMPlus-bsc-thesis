from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from kldmPlus.kldm import ModelKLDM
from kldmPlus.utils.ema import EMA


#Read the config files.
def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected config['{name}'] to be a mapping.")
    return value


def build_model(config: dict[str, Any], device: torch.device) -> ModelKLDM:
    cfg = _section(config, "model")
    dataset_cfg = _section(config, "dataset")
    score_network = _section(cfg, "score_network")
    if not score_network:
        raise ValueError("Config must explicitly define model.score_network.")

    conv_sg_aux = cfg.get("conv_sg_aux", {}) or {}
    if not isinstance(conv_sg_aux, dict):
        raise ValueError("Expected model.conv_sg_aux to be a mapping.")
    conv_sg_enabled = bool(conv_sg_aux.get("enabled", float(cfg.get("lambda_conv_sg", 0.0)) > 0.0))
    lambda_conv_sg = float(conv_sg_aux.get("lambda", cfg.get("lambda_conv_sg", 0.0))) if conv_sg_enabled else 0.0
    conv_sg_time_weight = str(conv_sg_aux.get("time_weight", cfg.get("conv_sg_time_weight", "alpha_squared")))
    conv_sg_require_valid_transform = bool(
        conv_sg_aux.get("require_valid_transform", cfg.get("conv_sg_require_valid_transform", True))
    )

    n_sigmas = cfg.get("tdm_n_sigmas")
    if n_sigmas is None:
        n_sigmas = 2000 if device.type == "cuda" else 512

    return ModelKLDM(
        device=device,
        eps=float(cfg.get("eps", 1e-6)),
        wrapped_normal_K=int(cfg.get("wrapped_normal_K", 13)),
        tdm_n_sigmas=int(n_sigmas),
        tdm_compute_sigma_norm=bool(cfg.get("tdm_compute_sigma_norm", True)),
        tdm_velocity_scale=cfg.get("tdm_velocity_scale"),
        tdm_sigma_norm_estimator=str(cfg.get("tdm_sigma_norm_estimator", "quadrature")),
        tdm_sigma_norm_density_K=cfg.get("tdm_sigma_norm_density_K"),
        tdm_sigma_norm_grid_points=int(cfg.get("tdm_sigma_norm_grid_points", 8193)),
        tdm_sigma_norm_mc_samples=int(cfg.get("tdm_sigma_norm_mc_samples", 20000)),
        lattice_parameterization=str(cfg.get("lattice_parameterization", "eps")),
        lattice_diffusion_type=str(cfg.get("lattice_diffusion_type", "VP")),
        lattice_representation=str(cfg.get("lattice_representation", dataset_cfg.get("lattice_representation", "kldm"))),
        lambda_l=float(cfg.get("lambda_l", 1.0)),
        lattice_sg_lambda=float(cfg.get("lattice_sg_lambda", 0.0)),
        lattice_sg_normalize=bool(cfg.get("lattice_sg_normalize", True)),
        lattice_sg_time_weight=str(cfg.get("lattice_sg_time_weight", "quadratic_late")),
        lambda_conv_sg=lambda_conv_sg,
        conv_sg_time_weight=conv_sg_time_weight,
        conv_sg_require_valid_transform=conv_sg_require_valid_transform,
        lattice_debug=bool(cfg.get("lattice_debug", False)),
        lattice_orbit_metric_max_candidates=cfg.get("lattice_orbit_metric_max_candidates", 512),
        score_network_kwargs=score_network,
    ).to(device)


def configure_trainable_parameters(model: ModelKLDM, config: dict[str, Any]) -> None:
    cfg = _section(config, "finetune")
    if not cfg:
        return
    freeze_base = bool(cfg.get("freeze_base", False))
    if not freeze_base:
        return
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    raise ValueError("finetune.freeze_base is no longer supported without explicit trainable parameter overrides.")


def build_optimizer(model: ModelKLDM, config: dict[str, Any]) -> torch.optim.Optimizer:
    cfg = _section(config, "optimizer")
    foreach = cfg.get("foreach", model.device.type == "cuda")
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found when building optimizer.")
    return torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-12)),
        amsgrad=bool(cfg.get("amsgrad", True)),
        foreach=bool(foreach),
    )


def build_ema(model: ModelKLDM, config: dict[str, Any]) -> EMA | None:
    cfg = _section(config, "ema")
    if not bool(cfg.get("enabled", True)):
        return None
    ema_type = str(cfg.get("type", "power" if "gamma" in cfg else "fixed"))

    if ema_type == "power":
        return EMA(
            model=model,
            gamma=float(cfg.get("gamma", 6.94)),
        )

    if ema_type == "fixed":
        return EMA(
            model=model,
            decay=float(cfg.get("decay", 0.999)),
            start_epoch=int(cfg.get("start_epoch", 500)),
        )

    raise ValueError(f"Unknown ema.type={ema_type!r}")


def build_training_components(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[ModelKLDM, torch.optim.Optimizer, EMA | None]:
    model = build_model(config=config, device=device)
    configure_trainable_parameters(model=model, config=config)
    return (
        model,
        build_optimizer(model=model, config=config),
        build_ema(model=model, config=config),
    )


def _ema_model_state(ema_state: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
    if ema_state is None:
        return None
    return {
        key.removeprefix("ema_model.module."): value
        for key, value in ema_state.items()
        if key.startswith("ema_model.module.")
    } or None


def load_checkpoint(
    *,
    checkpoint_path: str | Path,
    model: ModelKLDM,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    ema: EMA | None = None,
    prefer_ema_weights: bool = False,
) -> dict[str, Any]:
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
    )

    model_state = checkpoint["model_state_dict"]
    if prefer_ema_weights:
        model_state = _ema_model_state(checkpoint.get("ema_state_dict")) or model_state
    model.load_state_dict(model_state, strict=False)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if ema is not None and checkpoint.get("ema_state_dict") is not None:
        ema.load_state_dict(checkpoint["ema_state_dict"], strict=False)

    rng_state = checkpoint.get("rng_state")
    if isinstance(rng_state, dict):
        if rng_state.get("python") is not None:
            random.setstate(rng_state["python"])
        if rng_state.get("numpy") is not None:
            np.random.set_state(rng_state["numpy"])
        if rng_state.get("torch") is not None:
            torch.random.set_rng_state(rng_state["torch"].detach().cpu())
        cuda_state = rng_state.get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda_state])

    return checkpoint


def save_checkpoint(
    *,
    model: ModelKLDM,
    optimizer: torch.optim.Optimizer,
    ema: EMA | None,
    time_sampler=None,
    output_path: Path,
    config: dict[str, Any],
    epoch: int,
    metrics: Mapping[str, float | int | None],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": None if ema is None else ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "time_sampler_state_dict": None if time_sampler is None else time_sampler.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "config": config,
            "metrics": metrics,
        },
        output_path,
    )
