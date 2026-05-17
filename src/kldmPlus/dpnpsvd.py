from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from kldmPlus.data.transform import (
    FullyConnectedGraph,
    KLDMContinuousIntervalLattice,
    MatterGenContinuousIntervalLattice,
    lattice_feature_components,
    mattergen_lattice_feature_vector,
)
from kldmPlus.sample_evaluation import evaluate_csp_reconstruction
from kldmPlus.diffusionModels.continuous import ContinuousMattergenVPDiffusion
from kldmPlus.symmetry.frame_bridge import (
    build_symmetry_frame_bridge,
    map_standardized_structure_to_vanilla_frame,
    standardize_structure,
)
from kldmPlus.symmetry.k_basis import (
    cell_to_k,
    free_vars_to_k,
    k_to_cell_matrix,
    k_to_free_vars,
    space_group_k_constraint,
)
from kldmPlus.symmetry.pcs_projection import (
    PCSTemplateState,
    _build_structure_from_standardized_projection,
    _build_vanilla_structure,
    _cell_volume,
    _collapse_centering_equivalent_structure,
    _centering_translations,
    _periodic_pairwise_distances,
    _raw_target_in_requested_conventional_frame,
    _requested_centering_symbol,
    _species_assignment_indices,
    _species_orbit_mismatch_count,
    _standardized_target_tensors,
    _structure_to_primitive_centering_basis,
    _structure_species_orbit_signature_with_source,
    _target_representation_from_name,
    _template_species_orbit_signature,
    _volume_ratio_loss,
    validate_requested_space_group,
    vanilla_structure_to_model_tensors,
)
from kldmPlus.symmetry.template_cache import get_cache_entry, load_template_cache
from kldmPlus.symmetry.template_ranker import load_template_ranker, score_templates
from kldmPlus.symmetry.wyckoff_templates import (
    WyckoffTemplate,
    expand_wyckoff_template_torch,
    extract_wyckoff_templates,
    requested_conventional_atomic_numbers,
    sample_random_free_vars,
)
from kldmPlus.symmetry.template_prior import TemplatePrior, template_prior_score

try:
    from pymatgen.core import Lattice, Structure
except ImportError:  # pragma: no cover
    Lattice = Structure = None


@dataclass(frozen=True)
class DPnPSVDConfig:
    outer_steps: int = 2
    eta_start: float = 0.03
    eta_end: float = 0.012
    faithful_dpnp: bool = False
    max_templates: int = 512
    template_nmax: int = 20000
    quick_templates: bool = False
    template_cache_path: str | None = None
    template_cache_required: bool = False
    template_ranker_path: str | None = None
    template_selection_temperature: float = 0.0
    template_proposal_temperature: float = 1.0
    template_proposal_epsilon: float = 0.05
    template_move_probability: float = 0.10
    optimization_steps: int = 150
    learning_rate: float = 5.0e-2
    pcs_mh_steps: int = 24
    final_pcs_mh_steps: int = 24
    final_fixed_template_refine: bool = True
    svd_step_size: float = 1.0e-3
    svd_damping: float = 1.0e-2
    coord_weight: float = 2.0
    lattice_weight: float = 2.5
    residual_volume_weight: float = 1.0
    steric_weight: float = 40.0
    pair_distance_weight: float = 0.0
    min_distance: float = 1.0
    outer_hard_reject_distance_ratio: float = 1.0
    volume_weight: float = 5.0
    volume_ratio_min: float = 0.0
    volume_ratio_max: float = 0.0
    theta_proposal_free_std: float = 0.05
    theta_proposal_lattice_std: float = 0.05
    oracle_template_orbit_rerank: bool = False
    oracle_template_orbit_filter: bool = False
    oracle_mismatch_penalty: float = 1000.0
    ambient_dds_steps: int = 0
    ambient_dds_t_final: float = 1.0e-3
    ambient_dds_velocity_steps: int = 4
    ambient_dds_velocity_step_size: float = 1.0e-4
    sg_conditioned_dds: bool = False
    sg_guidance_scale: float = 1.0
    chart_dds_steps: int = 0
    chart_dds_projection_steps: int = 8
    chart_dds_kldm_steps: int = 1
    chart_dds_step_size: float = 5.0e-7
    chart_dds_damping: float = 5.0e-1
    chart_dds_coord_weight: float = 1.0
    chart_dds_lattice_weight: float = 0.2
    chart_dds_frac_blend: float = 0.03
    chart_dds_lattice_blend: float = 0.01
    chart_dds_prior_sigma: float = 0.05
    chart_dds_anchor_eta: float = 0.02
    chart_dds_destructive_delta: float = 0.0
    chart_dds_reject_invalid: bool = True
    chart_dds_t_final: float = 5.0e-4
    steric_softplus_tau: float = 0.1
    template_init_restarts: int = 4
    template_prior_mode: str = "dataset"
    template_prior_weight: float = 1.0
    oracle_template_prior_success_prob: float = 0.95
    oracle_template_prior_penalty: float = 1000.0
    symprec: float = 1.0e-2
    angle_tolerance: float = 5.0
    debug_oracle_step_metrics: bool = True
    debug_best_phase_metrics: bool = True
    debug_fixed_template_multistart_restarts: int = 0
    debug_fixed_template_multistart_steps: int = 0
    debug_fixed_template_multistart_eta: float = 0.0
    debug: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "DPnPSVDConfig":
        if not payload:
            return cls()
        fields = cls.__dataclass_fields__
        unknown = sorted(key for key in payload if key not in fields)
        if unknown:
            print(
                f"kldm_dpnpsvd_config_ignore unknown_keys={unknown}",
                flush=True,
            )
        values = {key: payload[key] for key in fields if key in payload}
        return cls(**values)


@dataclass(frozen=True)
class _RankedTemplate:
    template: WyckoffTemplate
    template_idx: int
    ranker_score: float
    template_prior_count: int
    orbit_mismatch: int
    proposal_logit: float


@dataclass(frozen=True)
class _TemplateRankingDebug:
    template_prior_mode: str
    oracle_surrogate_applied: bool
    oracle_surrogate_hit: bool
    oracle_surrogate_match_count: int


@dataclass(frozen=True)
class _PCSChartState:
    template: WyckoffTemplate
    constraint: Any
    bridge: Any
    free_vars: torch.Tensor
    lattice_free_vars: torch.Tensor
    template_rank: int
    candidate_count: int
    objective: float
    target_centering_symbol: str | None
    target_centering_translations: torch.Tensor | None
    target_representation_name: str
    anchor_frac: torch.Tensor
    anchor_atomic_numbers: torch.Tensor
    anchor_cell: torch.Tensor
    anchor_k: torch.Tensor
    anchor_assignment: torch.Tensor
    anchor_lattice_free_vars: torch.Tensor
    reference_volume: float | None


@dataclass
class _ProjectionView:
    frac_coords: torch.Tensor
    projected_cell: torch.Tensor
    residual: torch.Tensor
    min_pair_distance: float
    steric_loss: torch.Tensor
    pair_distance_loss: torch.Tensor
    volume_loss: torch.Tensor


@dataclass
class _ChartAmbientView:
    frac_coords: torch.Tensor
    lattice_features: torch.Tensor
    projected_cell: torch.Tensor
    atomic_numbers: torch.Tensor


@dataclass(frozen=True)
class _ThetaDebugStats:
    residual_norm: float
    coord_residual_norm: float
    lattice_residual_norm: float
    volume_residual: float
    min_pair_distance: float
    steric_loss: float
    pair_distance_loss: float
    volume_loss: float
    cell_volume: float
    max_lattice_length: float
    free_norm: float
    lattice_free_norm: float


@dataclass(frozen=True)
class _ChartDDSPriorStats:
    attempted_steps: int = 0
    available_steps: int = 0
    compat_batch_steps: int = 0
    last_reason: str = "not_run"


_RANKER_CACHE: dict[tuple[str, str], torch.nn.Module] = {}
_DISK_TEMPLATE_CACHE: dict[str, dict[str, Any]] = {}
_LIVE_TEMPLATE_CACHE: dict[tuple[int, tuple[int, ...], int, int, bool], list[WyckoffTemplate]] = {}


def _single_graph_batch(batch: Any, graph_idx: int) -> Any:
    if hasattr(batch, "get_example"):
        try:
            from torch_geometric.data import Batch as PyGBatch

            example = batch.get_example(int(graph_idx))
            return PyGBatch.from_data_list([example])
        except Exception:
            pass
    if hasattr(batch, "index_select"):
        try:
            subset = batch.index_select([int(graph_idx)])
            if hasattr(subset, "batch"):
                return subset
        except Exception:
            pass
    if hasattr(batch, "__getitem__"):
        try:
            subset = batch[int(graph_idx)]
            if hasattr(subset, "batch"):
                return subset
        except Exception:
            pass
    raise RuntimeError("Unable to extract a single-graph batch for DPnPSVD DDS repair.")


def _build_chart_compatible_batch(
    *,
    reference_batch: Any,
    pos: torch.Tensor,
    l: torch.Tensor,
    atomic_numbers: torch.Tensor,
) -> Any:
    try:
        from torch_geometric.data import Batch as PyGBatch
        from torch_geometric.data import Data as PyGData
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("torch_geometric is required to rebuild a chart-compatible KLDM batch.") from exc

    device = pos.device

    if hasattr(reference_batch, "space_group"):
        space_group = torch.as_tensor(reference_batch.space_group, device=device, dtype=torch.long).reshape(-1)[:1].clone()
    else:
        space_group = torch.zeros((1,), device=device, dtype=torch.long)

    if hasattr(reference_batch, "num_atoms"):
        num_atoms_dtype = torch.as_tensor(reference_batch.num_atoms).dtype
    else:
        num_atoms_dtype = torch.long
    num_nodes = int(pos.shape[0])
    num_atoms = torch.tensor([num_nodes], device=device, dtype=num_atoms_dtype)

    data = PyGData(
        pos=pos.detach().clone(),
        l=l.detach().clone(),
        atomic_numbers=atomic_numbers.detach().clone(),
        num_atoms=num_atoms,
        space_group=space_group,
    )
    # Reproduce the same fully connected directed edge rule used by the CSP
    # preprocessing path, but assign directly on a PyG Data object.
    edge_builder = FullyConnectedGraph()
    n = len(getattr(data, edge_builder.len_from))
    adjacency = torch.ones(n, n, device=data.pos.device)
    adjacency = adjacency - torch.eye(n, device=data.pos.device)
    try:
        from torch_geometric.utils import dense_to_sparse
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("torch_geometric is required to rebuild a chart-compatible KLDM batch.") from exc
    edge_index, _ = dense_to_sparse(adjacency)
    setattr(data, edge_builder.key, edge_index)
    return PyGBatch.from_data_list([data])


def _center_per_graph(values: torch.Tensor, *, index: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    num_graphs = int(index.max().item()) + 1 if index.numel() > 0 else 0
    if num_graphs <= 0:
        return values
    sums = torch.zeros((num_graphs, values.shape[-1]), device=values.device, dtype=values.dtype)
    sums = sums.index_add(0, index, values)
    counts = torch.bincount(index, minlength=num_graphs).to(device=values.device, dtype=values.dtype).clamp_min(1.0)
    means = sums / counts[:, None]
    return values - means[index]


def _maybe_load_template_ranker(
    *,
    path: str | None,
    device: torch.device,
) -> torch.nn.Module | None:
    if path is None or str(path).strip() == "":
        return None
    resolved = str(Path(path).expanduser().resolve())
    key = (resolved, str(device))
    cached = _RANKER_CACHE.get(key)
    if cached is None:
        cached = load_template_ranker(resolved, device=device)
        _RANKER_CACHE[key] = cached
        print(f"kldm_dpnpsvd_ranker_load path={resolved}", flush=True)
    return cached


def _maybe_load_disk_template_cache(path: str | None) -> dict[str, Any] | None:
    if path is None or str(path).strip() == "":
        return None
    resolved = str(Path(path).expanduser().resolve())
    cached = _DISK_TEMPLATE_CACHE.get(resolved)
    if cached is None:
        cached = load_template_cache(resolved)
        _DISK_TEMPLATE_CACHE[resolved] = cached
        print(
            f"kldm_dpnpsvd_template_cache_load path={resolved} entries={len(cached.get('entries', {}))}",
            flush=True,
        )
    return cached


def _cached_templates(
    *,
    space_group_number: int,
    atomic_numbers: torch.Tensor,
    max_templates: int,
    template_nmax: int,
    quick: bool,
    template_cache: dict[str, Any] | None,
    template_cache_required: bool,
) -> list[WyckoffTemplate]:
    atomic_key = tuple(int(v) for v in atomic_numbers.detach().cpu().reshape(-1).tolist())
    entry = get_cache_entry(
        template_cache,
        space_group_number=int(space_group_number),
        atomic_numbers=list(atomic_key),
    )
    if entry is not None:
        return list(entry.get("templates", []))[: int(max_templates)]
    if template_cache is not None and bool(template_cache_required):
        return []
    key = (
        int(space_group_number),
        atomic_key,
        int(max_templates),
        int(template_nmax),
        bool(quick),
    )
    cached = _LIVE_TEMPLATE_CACHE.get(key)
    if cached is None:
        if template_cache is not None:
            print(
                "kldm_dpnpsvd_template_cache_miss "
                f"sg={int(space_group_number)} composition={list(atomic_key)} "
                "fallback=live_enumeration",
                flush=True,
            )
        cached = extract_wyckoff_templates(
            space_group_number=int(space_group_number),
            atomic_numbers=list(atomic_key),
            max_templates=int(max_templates),
            quick=bool(quick),
            num_wp=(None, None),
            nmax=int(template_nmax),
        )
        _LIVE_TEMPLATE_CACHE[key] = cached
    return cached


def _lengths_angles_to_cell_matrix(lengths: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    a, b, c = lengths.unbind(dim=-1)
    alpha, beta, gamma = angles.unbind(dim=-1)
    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma).clamp_min(1.0e-8)
    zeros = torch.zeros_like(a)
    row_a = torch.stack([a, zeros, zeros], dim=-1)
    row_b = torch.stack([b * cos_gamma, b * sin_gamma, zeros], dim=-1)
    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz = torch.sqrt((c.square() - cx.square() - cy.square()).clamp_min(1.0e-8))
    row_c = torch.stack([cx, cy, cz], dim=-1)
    return torch.stack([row_a, row_b, row_c], dim=-2)


def _decode_lattice_matrix(
    *,
    l: torch.Tensor,
    num_atoms: int,
    lattice_transform: Any,
) -> torch.Tensor:
    if lattice_transform is not None and hasattr(lattice_transform, "invert_to_matrix"):
        return lattice_transform.invert_to_matrix(l.view(1, -1), num_atoms=num_atoms).reshape(3, 3)
    if isinstance(lattice_transform, MatterGenContinuousIntervalLattice):
        return lattice_transform.invert_to_matrix(l.view(1, -1), num_atoms=num_atoms).reshape(3, 3)
    if isinstance(lattice_transform, KLDMContinuousIntervalLattice):
        lengths, angles = lattice_transform.invert_to_lengths_angles(l.view(1, -1), num_atoms=num_atoms)
        return _lengths_angles_to_cell_matrix(lengths.squeeze(0), angles.squeeze(0))
    log_lengths, angle_features = l.view(1, -1)[..., :3], l.view(1, -1)[..., 3:]
    lengths = torch.exp(log_lengths)
    angles = torch.atan(angle_features) + torch.pi / 2.0
    return _lengths_angles_to_cell_matrix(lengths.squeeze(0), angles.squeeze(0))


def _encode_lattice_features(
    *,
    cell_matrix: torch.Tensor,
    num_atoms: int,
    lattice_transform: Any,
) -> torch.Tensor:
    if isinstance(lattice_transform, MatterGenContinuousIntervalLattice):
        return mattergen_lattice_feature_vector(cell_matrix).view(1, 6)
    if isinstance(lattice_transform, KLDMContinuousIntervalLattice):
        log_lengths, angle_features = lattice_feature_components(cell_matrix, eps=lattice_transform.eps)
        if lattice_transform.standardize and lattice_transform.lengths_loc_scale is not None:
            log_lengths, angle_features = lattice_transform._encode_x0_parts(
                log_lengths=log_lengths,
                angle_features=angle_features,
                num_atoms=int(num_atoms),
            )
            return torch.cat([log_lengths, angle_features], dim=0).view(1, 6)
        features = torch.cat([log_lengths, angle_features], dim=0)
        return lattice_transform.standardize_value(features).view(1, 6)
    log_lengths, angle_features = lattice_feature_components(cell_matrix)
    return torch.cat([log_lengths, angle_features], dim=0).view(1, 6)


def _eta_schedule(cfg: DPnPSVDConfig, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if int(cfg.outer_steps) < 1:
        raise ValueError("outer_steps must be >= 1.")
    if int(cfg.outer_steps) == 1:
        return torch.tensor([float(cfg.eta_start)], device=device, dtype=dtype)
    return torch.linspace(
        float(cfg.eta_start),
        float(cfg.eta_end),
        int(cfg.outer_steps),
        device=device,
        dtype=dtype,
    )


def _structure_from_tensors(frac: torch.Tensor, atomic_numbers: torch.Tensor, cell: torch.Tensor):
    if Structure is None or Lattice is None:
        raise ImportError("DPnPSVD oracle diagnostics require pymatgen.")
    return Structure(
        lattice=Lattice(cell.detach().cpu().numpy()),
        species=[int(v) for v in atomic_numbers.detach().cpu().tolist()],
        coords=np.asarray(frac.detach().cpu().numpy(), dtype=float),
        coords_are_cartesian=False,
        to_unit_cell=True,
    )


def _oracle_target_orbit_signature(
    *,
    target_frac: torch.Tensor,
    target_l: torch.Tensor,
    target_species: torch.Tensor,
    lattice_transform: Any,
    cfg: DPnPSVDConfig,
) -> tuple[tuple[tuple[int, str], ...], str]:
    structure = _structure_from_tensors(
        target_frac,
        target_species,
        _decode_lattice_matrix(
            l=target_l,
            num_atoms=int(target_frac.shape[0]),
            lattice_transform=lattice_transform,
        ).to(device=target_frac.device, dtype=target_frac.dtype),
    )
    _analyzer, standardized = standardize_structure(
        structure,
        standardization="conventional",
        symprec=float(cfg.symprec),
        angle_tolerance=float(cfg.angle_tolerance),
    )
    del _analyzer
    signature, source = _structure_species_orbit_signature_with_source(
        structure=standardized,
        symprec=float(cfg.symprec),
        angle_tolerance=float(cfg.angle_tolerance),
    )
    return signature, source


def _template_signature_labels(template: WyckoffTemplate) -> list[str]:
    return [f"{int(site.atomic_number)}@{str(site.label)}" for site in template.site_templates]


def _template_species_counts(template: WyckoffTemplate) -> dict[int, int]:
    counts: dict[int, int] = {}
    for site in template.site_templates:
        atomic_number = int(site.atomic_number)
        counts[atomic_number] = counts.get(atomic_number, 0) + int(site.multiplicity)
    return counts


def _rank_templates(
    *,
    templates: list[WyckoffTemplate],
    requested_sg: int,
    template_atomic_numbers: torch.Tensor,
    template_prior: TemplatePrior | None,
    ranker: torch.nn.Module | None,
    device: torch.device,
    oracle_target_signature: tuple[tuple[int, str], ...],
    cfg: DPnPSVDConfig,
) -> tuple[list[_RankedTemplate], _TemplateRankingDebug]:
    scores = score_templates(
        ranker=ranker,
        templates=templates,
        requested_sg=int(requested_sg),
        device=device,
    )
    unique_species, species_counts = torch.unique(
        template_atomic_numbers.to(device="cpu", dtype=torch.long),
        sorted=True,
        return_counts=True,
    )
    prior_key = (
        int(requested_sg),
        tuple(int(v) for v in unique_species.tolist()),
        tuple(int(v) for v in species_counts.tolist()),
    )
    ranked: list[_RankedTemplate] = []
    template_prior_mode = str(getattr(cfg, "template_prior_mode", "dataset")).strip().lower() or "dataset"
    for template_idx, template in enumerate(templates):
        orbit_mismatch = int(
            _species_orbit_mismatch_count(
                template_signature=_template_species_orbit_signature(template),
                target_signature=oracle_target_signature,
            )
        )
        prior_count = int(
            template_prior_score(
                prior=template_prior,
                key=prior_key,
                signature=tuple((int(site.atomic_number), str(site.label)) for site in template.site_templates),
            )
        )
        prior_bonus = float(cfg.template_prior_weight) * math.log1p(max(prior_count, 0))
        proposal_logit = float(scores[template_idx])
        proposal_logit += prior_bonus
        if bool(cfg.oracle_template_orbit_rerank):
            proposal_logit -= float(cfg.oracle_mismatch_penalty) * float(orbit_mismatch)
        ranked.append(
            _RankedTemplate(
                template=template,
                template_idx=template_idx,
                ranker_score=float(scores[template_idx]),
                template_prior_count=prior_count,
                orbit_mismatch=orbit_mismatch,
                proposal_logit=proposal_logit,
            )
        )
    oracle_surrogate_applied = False
    oracle_surrogate_hit = False
    oracle_surrogate_match_count = 0
    if template_prior_mode == "oracle_surrogate" and oracle_target_signature and ranked:
        match_count = sum(1 for item in ranked if int(item.orbit_mismatch) == 0)
        oracle_surrogate_match_count = int(match_count)
        if match_count > 0:
            oracle_surrogate_applied = True
            success_prob = min(
                max(float(cfg.oracle_template_prior_success_prob), 0.0),
                1.0,
            )
            oracle_surrogate_hit = bool(torch.rand((), device=device).item() < success_prob)
            if oracle_surrogate_hit:
                penalty = float(cfg.oracle_template_prior_penalty)
                ranked = [
                    replace(
                        item,
                        proposal_logit=(
                            float(item.proposal_logit) - penalty * float(item.orbit_mismatch)
                        ),
                    )
                    for item in ranked
                ]
    ranked.sort(
        key=lambda item: (
            -float(item.proposal_logit),
            -float(item.ranker_score),
            float(item.template.total_free_dims),
            float(item.template.total_sites),
            float(item.template.total_atoms),
            float(item.template_idx),
        )
    )
    return ranked, _TemplateRankingDebug(
        template_prior_mode=template_prior_mode,
        oracle_surrogate_applied=bool(oracle_surrogate_applied),
        oracle_surrogate_hit=bool(oracle_surrogate_hit),
        oracle_surrogate_match_count=int(oracle_surrogate_match_count),
    )


def _effective_faithful_cfg(cfg: DPnPSVDConfig) -> DPnPSVDConfig:
    if not bool(cfg.faithful_dpnp):
        return cfg
    return replace(
        cfg,
        pair_distance_weight=0.0,
        final_fixed_template_refine=False,
        oracle_template_orbit_rerank=False,
        oracle_template_orbit_filter=False,
    )


def _proposal_template_log_probs(
    *,
    ranked_templates: list[_RankedTemplate],
    cfg: DPnPSVDConfig,
    device: torch.device,
) -> torch.Tensor:
    if not ranked_templates:
        raise RuntimeError("Cannot build template proposal over an empty catalog.")
    logits = torch.tensor(
        [item.proposal_logit for item in ranked_templates],
        device=device,
        dtype=torch.float64,
    )
    tau = float(cfg.template_proposal_temperature)
    if tau <= 0.0:
        tau = 1.0
    base = torch.softmax(logits / tau, dim=0)
    eps = float(cfg.template_proposal_epsilon)
    probs = (1.0 - eps) * base + eps / float(len(ranked_templates))
    probs = probs / probs.sum().clamp_min(1.0e-12)
    return torch.log(probs.clamp_min(1.0e-12))


def _initial_template_position(
    *,
    ranked_templates: list[_RankedTemplate],
    cfg: DPnPSVDConfig,
    device: torch.device,
) -> int:
    if not ranked_templates:
        raise RuntimeError("Cannot initialize DPnPSVD from an empty template list.")
    if float(cfg.template_selection_temperature) <= 0.0:
        return 0
    logits = torch.tensor(
        [item.proposal_logit for item in ranked_templates],
        device=device,
        dtype=torch.float64,
    ) / max(float(cfg.template_selection_temperature), 1.0e-8)
    probs = torch.softmax(logits, dim=0)
    return int(torch.multinomial(probs, 1).item())


def _anchor_representation(
    *,
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell_matrix: torch.Tensor,
    requested_sg: int,
    cfg: DPnPSVDConfig,
    template_total_atoms: int | None = None,
) -> dict[str, Any]:
    device = frac_coords.device
    dtype = frac_coords.dtype
    vanilla_structure = _build_vanilla_structure(
        frac_coords=frac_coords,
        atomic_numbers=atomic_numbers,
        cell_matrix=cell_matrix,
    )
    bridge = build_symmetry_frame_bridge(
        vanilla_structure=vanilla_structure,
        standardization="conventional",
        symprec=float(cfg.symprec),
        angle_tolerance=float(cfg.angle_tolerance),
    )
    standardized_frac, standardized_atomic_numbers, standardized_cell, _ = _standardized_target_tensors(
        bridge,
        device=device,
        dtype=dtype,
    )

    target_centering = _requested_centering_symbol(int(requested_sg))
    target_centering_translations = _centering_translations(target_centering, device=device, dtype=dtype)
    raw_requested_frac, raw_requested_cell = _raw_target_in_requested_conventional_frame(
        frac_coords=torch.remainder(frac_coords, 1.0),
        cell_matrix=cell_matrix,
        centering_symbol=target_centering,
    )
    standardized_target = _target_representation_from_name(
        target_name="standardized",
        raw_requested_frac=raw_requested_frac,
        raw_requested_atomic_numbers=atomic_numbers.to(device=device, dtype=torch.long),
        raw_requested_cell=raw_requested_cell,
        standardized_frac=standardized_frac,
        standardized_atomic_numbers=standardized_atomic_numbers,
        standardized_cell=standardized_cell,
        centering_translations=None,
    )
    raw_requested_expanded_target = _target_representation_from_name(
        target_name="raw_requested_expanded",
        raw_requested_frac=raw_requested_frac,
        raw_requested_atomic_numbers=atomic_numbers.to(device=device, dtype=torch.long),
        raw_requested_cell=raw_requested_cell,
        standardized_frac=standardized_frac,
        standardized_atomic_numbers=standardized_atomic_numbers,
        standardized_cell=standardized_cell,
        centering_translations=target_centering_translations,
    )
    standardized_count = int(standardized_target[1].shape[0])
    raw_requested_expanded_count = int(raw_requested_expanded_target[1].shape[0])
    use_raw_requested_expanded = (
        template_total_atoms is not None
        and int(template_total_atoms) == raw_requested_expanded_count
        and int(template_total_atoms) != standardized_count
    )
    if use_raw_requested_expanded:
        target_name = "raw_requested_expanded"
        target_frac, target_atomic_numbers, target_cell, target_k = raw_requested_expanded_target
        resolved_target_centering_symbol = target_centering
        resolved_target_centering_translations = target_centering_translations
    else:
        target_name = "standardized"
        target_frac, target_atomic_numbers, target_cell, target_k = standardized_target
        resolved_target_centering_symbol = None
        resolved_target_centering_translations = None
    return {
        "bridge": bridge,
        "target_centering_symbol": resolved_target_centering_symbol,
        "target_centering_translations": resolved_target_centering_translations,
        "target_frac": target_frac.detach().clone(),
        "target_atomic_numbers": target_atomic_numbers.detach().clone(),
        "target_cell": target_cell.detach().clone(),
        "target_k": target_k.detach().clone(),
        "target_representation_name": str(target_name),
    }


def _wrap_delta(delta: torch.Tensor) -> torch.Tensor:
    return delta - torch.round(delta)


def _soft_steric_overlap_loss(
    *,
    distances: torch.Tensor,
    min_distance: float,
    tau: float,
) -> torch.Tensor:
    if distances.numel() == 0:
        return distances.new_zeros(())
    scale = max(float(tau), 1.0e-6)
    gap = (float(min_distance) - distances) / scale
    penalties = F.softplus(gap)
    return penalties.square().mean()


def _cell_sanity_status(
    *,
    cell_matrix: np.ndarray,
    frac_coords: np.ndarray,
) -> tuple[bool, str, float | None, float | None]:
    if cell_matrix.shape != (3, 3):
        return False, "bad_cell_shape", None, None
    if frac_coords.ndim != 2 or frac_coords.shape[-1] != 3:
        return False, "bad_frac_shape", None, None
    if not np.isfinite(cell_matrix).all():
        return False, "nonfinite_cell", None, None
    if not np.isfinite(frac_coords).all():
        return False, "nonfinite_frac", None, None

    lengths = np.linalg.norm(cell_matrix, axis=1)
    if not np.isfinite(lengths).all():
        return False, "nonfinite_lengths", None, None
    max_lattice_length = float(np.max(lengths)) if lengths.size > 0 else 0.0
    min_lattice_length = float(np.min(lengths)) if lengths.size > 0 else 0.0
    if min_lattice_length < 1.0e-4:
        return False, "tiny_lattice", None, max_lattice_length
    if max_lattice_length > 40.0:
        return False, "huge_lattice", None, max_lattice_length

    try:
        volume = float(abs(np.linalg.det(cell_matrix)))
    except Exception:
        return False, "det_failed", None, max_lattice_length
    if not math.isfinite(volume):
        return False, "nonfinite_volume", None, max_lattice_length
    if volume < 0.1:
        return False, "tiny_volume", volume, max_lattice_length
    return True, "ok", volume, max_lattice_length


def _target_representation_for_state(
    *,
    state: _PCSChartState,
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell_matrix: torch.Tensor,
    cfg: DPnPSVDConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = frac_coords.device
    dtype = frac_coords.dtype
    target_name = state.target_representation_name or "standardized"
    if target_name == "raw_requested_expanded":
        anchor_frac = torch.remainder(frac_coords, 1.0).detach().clone()
        anchor_atomic_numbers = atomic_numbers.to(device=device, dtype=torch.long).detach().clone()
        anchor_cell = cell_matrix.to(device=device, dtype=dtype).detach().clone()
        anchor_k = cell_to_k(anchor_cell, eps=1.0e-8)
        expected_template_atoms = int(state.template.total_atoms)
        if int(anchor_atomic_numbers.shape[0]) != expected_template_atoms:
            raise RuntimeError(
                "Expanded target refresh atom count does not match template atom count: "
                f"target_repr={target_name!r}, "
                f"template_atoms={expected_template_atoms}, "
                f"anchor_atoms={int(anchor_atomic_numbers.shape[0])}."
            )
        return anchor_frac, anchor_atomic_numbers, anchor_cell, anchor_k

    vanilla_structure = _build_vanilla_structure(
        frac_coords=frac_coords,
        atomic_numbers=atomic_numbers,
        cell_matrix=cell_matrix,
    )
    bridge = build_symmetry_frame_bridge(
        vanilla_structure=vanilla_structure,
        standardization="conventional",
        symprec=float(cfg.symprec),
        angle_tolerance=float(cfg.angle_tolerance),
    )
    standardized_frac, standardized_atomic_numbers, standardized_cell, _ = _standardized_target_tensors(
        bridge,
        device=device,
        dtype=dtype,
    )
    centering_symbol = state.target_centering_symbol or _requested_centering_symbol(int(state.constraint.space_group))
    centering_translations = state.target_centering_translations
    if centering_translations is not None:
        centering_translations = centering_translations.to(device=device, dtype=dtype)
    raw_requested_frac, raw_requested_cell = _raw_target_in_requested_conventional_frame(
        frac_coords=torch.remainder(frac_coords, 1.0),
        cell_matrix=cell_matrix,
        centering_symbol=centering_symbol,
    )
    return _target_representation_from_name(
        target_name=target_name,
        raw_requested_frac=raw_requested_frac,
        raw_requested_atomic_numbers=atomic_numbers.to(device=device, dtype=torch.long),
        raw_requested_cell=raw_requested_cell,
        standardized_frac=standardized_frac,
        standardized_atomic_numbers=standardized_atomic_numbers,
        standardized_cell=standardized_cell,
        centering_translations=centering_translations,
    )


def _local_template_fit(
    *,
    template: WyckoffTemplate,
    constraint: Any,
    target_frac: torch.Tensor,
    target_atomic_numbers: torch.Tensor,
    target_k: torch.Tensor,
    cfg: DPnPSVDConfig,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    device = target_frac.device
    dtype = target_frac.dtype
    anchor_lattice_free = k_to_free_vars(target_k, constraint).detach().clone()
    best_free = torch.zeros((int(template.total_free_dims),), device=device, dtype=dtype)
    best_lattice = anchor_lattice_free.detach().clone()
    best_loss = float("inf")
    reference_volume = float(_cell_volume(k_to_cell_matrix(target_k)).detach().item())

    for _restart in range(max(1, int(cfg.template_init_restarts))):
        restart_idx = _restart + 1
        free_init = sample_random_free_vars(
            template,
            device=device,
            dtype=dtype,
        ).reshape(-1)
        if free_init.numel() == 0:
            free_param = torch.zeros((0,), device=device, dtype=dtype, requires_grad=True)
        else:
            free_param = free_init.detach().clone().requires_grad_(True)
        lattice_param = anchor_lattice_free.detach().clone().requires_grad_(True)

        optimizer = torch.optim.Adam([free_param, lattice_param], lr=float(cfg.learning_rate))
        restart_best_free = free_param.detach().clone()
        restart_best_lattice = lattice_param.detach().clone()
        restart_best_loss = float("inf")

        for _ in range(int(cfg.optimization_steps)):
            optimizer.zero_grad(set_to_none=True)
            expansion = expand_wyckoff_template_torch(template=template, free_vars=free_param)
            if not torch.isfinite(expansion.frac_coords).all():
                if bool(cfg.debug):
                    print(
                        "kldm_dpnpsvd_template_fit_skip reason=nonfinite_expansion",
                        flush=True,
                    )
                break
            with torch.no_grad():
                try:
                    assignment = _species_assignment_indices(
                        source_frac=expansion.frac_coords,
                        source_atomic_numbers=expansion.atomic_numbers,
                        target_frac=target_frac,
                        target_atomic_numbers=target_atomic_numbers,
                    )
                except RuntimeError as exc:
                    if "invalid numeric entries" not in str(exc):
                        raise
                    if bool(cfg.debug):
                        print(
                            "kldm_dpnpsvd_template_fit_skip reason=invalid_assignment_matrix",
                            flush=True,
                        )
                    break
            matched_target = target_frac[assignment]
            coord_residual = _wrap_delta(expansion.frac_coords - matched_target).reshape(-1)
            lattice_residual = lattice_param - anchor_lattice_free
            projected_cell = k_to_cell_matrix(free_vars_to_k(lattice_param, constraint))
            if not torch.isfinite(projected_cell).all():
                if bool(cfg.debug):
                    print(
                        "kldm_dpnpsvd_template_fit_skip reason=nonfinite_projected_cell",
                        flush=True,
                    )
                break
            pair_distances = _periodic_pairwise_distances(
                frac_coords=expansion.frac_coords,
                cell_matrix=projected_cell,
            )
            coord_term = coord_residual.square().mean() if coord_residual.numel() > 0 else free_param.new_zeros(())
            lattice_term = lattice_residual.square().mean() if lattice_residual.numel() > 0 else lattice_param.new_zeros(())
            steric_term = _soft_steric_overlap_loss(
                distances=pair_distances,
                min_distance=float(cfg.min_distance),
                tau=float(cfg.steric_softplus_tau),
            )
            volume_term = _volume_ratio_loss(
                projected_cell=projected_cell,
                reference_volume=reference_volume,
                min_ratio=float(cfg.volume_ratio_min),
                max_ratio=float(cfg.volume_ratio_max),
            )
            loss = (
                float(cfg.coord_weight) * coord_term
                + float(cfg.lattice_weight) * lattice_term
                + float(cfg.steric_weight) * steric_term
                + float(cfg.volume_weight) * volume_term
            )
            if not torch.isfinite(loss):
                if bool(cfg.debug):
                    print(
                        "kldm_dpnpsvd_template_fit_skip reason=nonfinite_loss",
                        flush=True,
                    )
                break
            loss_value = float(loss.detach().item())
            free_before = free_param.detach().clone()
            lattice_before = lattice_param.detach().clone()
            if loss_value < restart_best_loss:
                restart_best_loss = loss_value
                restart_best_free = free_before
                restart_best_lattice = lattice_before
            loss.backward()
            optimizer.step()
            if free_param.numel() > 0:
                with torch.no_grad():
                    free_param.remainder_(1.0)
            if not torch.isfinite(free_param).all():
                if bool(cfg.debug):
                    print(
                        "kldm_dpnpsvd_template_fit_skip reason=nonfinite_free_after_step",
                        flush=True,
                    )
                break
            if not torch.isfinite(lattice_param).all():
                if bool(cfg.debug):
                    print(
                        "kldm_dpnpsvd_template_fit_skip reason=nonfinite_lattice_after_step",
                        flush=True,
                    )
                break
            projected_cell_after = k_to_cell_matrix(free_vars_to_k(lattice_param, constraint))
            if not torch.isfinite(projected_cell_after).all():
                if bool(cfg.debug):
                    print(
                        "kldm_dpnpsvd_template_fit_skip reason=nonfinite_projected_cell_after_step",
                        flush=True,
                    )
                break

        if restart_best_loss < best_loss:
            best_loss = restart_best_loss
            best_free = restart_best_free
            best_lattice = restart_best_lattice
        if bool(cfg.debug):
            print(
                f"kldm_dpnpsvd_template_fit_restart restart={restart_idx}/{max(1, int(cfg.template_init_restarts))} "
                f"best_loss={restart_best_loss:.6f}",
                flush=True,
            )

    if not math.isfinite(best_loss):
        raise RuntimeError("DPnPSVD could not initialize template state because every local fit restart failed.")
    oracle_metrics_prev: dict[str, float | None] | None = None
    oracle_metrics_prev: dict[str, float | None] | None = None
    if bool(cfg.debug):
        best_expansion = expand_wyckoff_template_torch(template=template, free_vars=best_free)
        best_assignment = _species_assignment_indices(
            source_frac=best_expansion.frac_coords,
            source_atomic_numbers=best_expansion.atomic_numbers,
            target_frac=target_frac,
            target_atomic_numbers=target_atomic_numbers,
        )
        fitted_theta = torch.cat([best_free.reshape(-1), best_lattice.reshape(-1)], dim=0)
        init_state = SimpleNamespace(
            template=template,
            constraint=constraint,
            anchor_frac=target_frac.detach().clone(),
            anchor_assignment=best_assignment.detach().clone(),
            anchor_lattice_free_vars=anchor_lattice_free.detach().clone(),
            anchor_cell=k_to_cell_matrix(target_k).detach().clone(),
            reference_volume=reference_volume,
        )
        stats = _theta_debug_stats(
            state=init_state,  # type: ignore[arg-type]
            theta=fitted_theta,
            cfg=cfg,
        )
        print(
            f"kldm_dpnpsvd_template_fit_best best_loss={best_loss:.6f} "
            f"residual_norm={stats.residual_norm:.6f} coord_residual_norm={stats.coord_residual_norm:.6f} "
            f"lattice_residual_norm={stats.lattice_residual_norm:.6f} min_pair_distance={stats.min_pair_distance:.4f} "
            f"cell_volume={stats.cell_volume:.6f} max_lattice_length={stats.max_lattice_length:.6f}",
            flush=True,
        )
    return best_free, best_lattice, best_loss


def _cell_abc_angles(cell_matrix: torch.Tensor) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    cell = cell_matrix.detach().to(dtype=torch.float64)
    a_vec, b_vec, c_vec = cell[0], cell[1], cell[2]
    a = float(torch.linalg.norm(a_vec).item())
    b = float(torch.linalg.norm(b_vec).item())
    c = float(torch.linalg.norm(c_vec).item())

    def _angle(u: torch.Tensor, v: torch.Tensor) -> float:
        denom = torch.linalg.norm(u) * torch.linalg.norm(v)
        if float(denom.item()) <= 1.0e-12:
            return float("nan")
        cosine = torch.clamp(torch.dot(u, v) / denom, min=-1.0, max=1.0)
        return float(torch.rad2deg(torch.acos(cosine)).item())

    alpha = _angle(b_vec, c_vec)
    beta = _angle(a_vec, c_vec)
    gamma = _angle(a_vec, b_vec)
    return (a, b, c), (alpha, beta, gamma)


def _initialize_state_for_template(
    *,
    template: WyckoffTemplate,
    template_rank: int,
    candidate_count: int,
    anchor_frac: torch.Tensor,
    anchor_species: torch.Tensor,
    anchor_l: torch.Tensor,
    requested_sg: int,
    lattice_transform: Any,
    cfg: DPnPSVDConfig,
    target_representation_name: str | None = None,
    target_centering_symbol: str | None = None,
    target_centering_translations: torch.Tensor | None = None,
) -> _PCSChartState:
    device = anchor_frac.device
    dtype = anchor_frac.dtype
    cell_matrix = _decode_lattice_matrix(
        l=anchor_l,
        num_atoms=int(anchor_frac.shape[0]),
        lattice_transform=lattice_transform,
    ).to(device=device, dtype=dtype)
    if target_representation_name is None:
        anchor_view = _anchor_representation(
            frac_coords=anchor_frac,
            atomic_numbers=anchor_species,
            cell_matrix=cell_matrix,
            requested_sg=int(requested_sg),
            cfg=cfg,
            template_total_atoms=int(template.total_atoms),
        )
        bridge = anchor_view["bridge"]
        resolved_target_centering_symbol = anchor_view["target_centering_symbol"]
        resolved_target_centering_translations = anchor_view["target_centering_translations"]
        resolved_target_representation_name = str(anchor_view["target_representation_name"])
        target_frac = anchor_view["target_frac"]
        target_atomic_numbers = anchor_view["target_atomic_numbers"]
        target_cell = anchor_view["target_cell"]
        target_k = anchor_view["target_k"]
    else:
        vanilla_structure = _build_vanilla_structure(
            frac_coords=anchor_frac,
            atomic_numbers=anchor_species,
            cell_matrix=cell_matrix,
        )
        bridge = build_symmetry_frame_bridge(
            vanilla_structure=vanilla_structure,
            standardization="conventional",
            symprec=float(cfg.symprec),
            angle_tolerance=float(cfg.angle_tolerance),
        )
        standardized_frac, standardized_atomic_numbers, standardized_cell, _ = _standardized_target_tensors(
            bridge,
            device=device,
            dtype=dtype,
        )
        centering_symbol = target_centering_symbol or _requested_centering_symbol(int(requested_sg))
        centering_translations = target_centering_translations
        if centering_translations is not None:
            centering_translations = centering_translations.to(device=device, dtype=dtype)
        raw_requested_frac, raw_requested_cell = _raw_target_in_requested_conventional_frame(
            frac_coords=torch.remainder(anchor_frac, 1.0),
            cell_matrix=cell_matrix,
            centering_symbol=centering_symbol,
        )
        target_frac, target_atomic_numbers, target_cell, target_k = _target_representation_from_name(
            target_name=str(target_representation_name),
            raw_requested_frac=raw_requested_frac,
            raw_requested_atomic_numbers=anchor_species.to(device=device, dtype=torch.long),
            raw_requested_cell=raw_requested_cell,
            standardized_frac=standardized_frac,
            standardized_atomic_numbers=standardized_atomic_numbers,
            standardized_cell=standardized_cell,
            centering_translations=centering_translations,
        )
        resolved_target_centering_symbol = target_centering_symbol
        resolved_target_centering_translations = (
            centering_translations.detach().clone() if centering_translations is not None else None
        )
        resolved_target_representation_name = str(target_representation_name)
    constraint = space_group_k_constraint(
        space_group_number=int(requested_sg),
        device=device,
        dtype=dtype,
    )
    fitted_free, fitted_lattice, objective = _local_template_fit(
        template=template,
        constraint=constraint,
        target_frac=target_frac,
        target_atomic_numbers=target_atomic_numbers,
        target_k=target_k,
        cfg=cfg,
    )
    expansion = expand_wyckoff_template_torch(template=template, free_vars=fitted_free)
    anchor_assignment = _species_assignment_indices(
        source_frac=expansion.frac_coords,
        source_atomic_numbers=expansion.atomic_numbers,
        target_frac=target_frac,
        target_atomic_numbers=target_atomic_numbers,
    )
    if bool(cfg.debug):
        target_abc, target_angles = _cell_abc_angles(target_cell)
        anchor_abc, anchor_angles = _cell_abc_angles(cell_matrix)
        print(
            f"kldm_dpnpsvd_init_repr sg={int(requested_sg)} "
            f"representation={resolved_target_representation_name} "
            f"free_dim={int(template.total_free_dims)} lattice_dim={int(fitted_lattice.numel())} "
            f"target_abc={[round(v, 4) for v in target_abc]} "
            f"target_angles={[round(v, 4) for v in target_angles]} "
            f"anchor_abc={[round(v, 4) for v in anchor_abc]} "
            f"anchor_angles={[round(v, 4) for v in anchor_angles]}",
            flush=True,
        )
    return _PCSChartState(
        template=template,
        constraint=constraint,
        bridge=bridge,
        free_vars=fitted_free.detach().clone(),
        lattice_free_vars=fitted_lattice.detach().clone(),
        template_rank=int(template_rank),
        candidate_count=int(candidate_count),
        objective=float(objective),
        target_centering_symbol=resolved_target_centering_symbol,
        target_centering_translations=resolved_target_centering_translations,
        target_representation_name=resolved_target_representation_name,
        anchor_frac=target_frac.detach().clone(),
        anchor_atomic_numbers=target_atomic_numbers.detach().clone(),
        anchor_cell=target_cell.detach().clone(),
        anchor_k=target_k.detach().clone(),
        anchor_assignment=anchor_assignment.detach().clone(),
        anchor_lattice_free_vars=k_to_free_vars(target_k, constraint).detach().clone(),
        reference_volume=float(_cell_volume(target_cell).detach().item()),
    )


def _retarget_state(
    *,
    state: _PCSChartState,
    theta: torch.Tensor,
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell_matrix: torch.Tensor,
    cfg: DPnPSVDConfig,
) -> _PCSChartState:
    device = frac_coords.device
    dtype = frac_coords.dtype
    vanilla_structure = _build_vanilla_structure(
        frac_coords=frac_coords,
        atomic_numbers=atomic_numbers,
        cell_matrix=cell_matrix,
    )
    bridge = build_symmetry_frame_bridge(
        vanilla_structure=vanilla_structure,
        standardization="conventional",
        symprec=float(cfg.symprec),
        angle_tolerance=float(cfg.angle_tolerance),
    )
    standardized_frac, standardized_atomic_numbers, standardized_cell, _ = _standardized_target_tensors(
        bridge,
        device=device,
        dtype=dtype,
    )
    centering_symbol = state.target_centering_symbol or _requested_centering_symbol(int(state.constraint.space_group))
    centering_translations = state.target_centering_translations
    if centering_translations is not None:
        centering_translations = centering_translations.to(device=device, dtype=dtype)
    raw_requested_frac, raw_requested_cell = _raw_target_in_requested_conventional_frame(
        frac_coords=torch.remainder(frac_coords, 1.0),
        cell_matrix=cell_matrix,
        centering_symbol=centering_symbol,
    )
    target_name = state.target_representation_name or "standardized"
    anchor_frac, anchor_atomic_numbers, anchor_cell, anchor_k = _target_representation_from_name(
        target_name=target_name,
        raw_requested_frac=raw_requested_frac,
        raw_requested_atomic_numbers=atomic_numbers.to(device=device, dtype=torch.long),
        raw_requested_cell=raw_requested_cell,
        standardized_frac=standardized_frac,
        standardized_atomic_numbers=standardized_atomic_numbers,
        standardized_cell=standardized_cell,
        centering_translations=centering_translations,
    )
    free_dim = int(state.template.total_free_dims)
    free_vars = theta[:free_dim].detach().clone()
    lattice_free = theta[free_dim:].detach().clone()
    expansion = expand_wyckoff_template_torch(template=state.template, free_vars=free_vars)

    expected_template_atoms = int(expansion.frac_coords.shape[0])
    refreshed_anchor_atoms = int(anchor_atomic_numbers.shape[0])
    if refreshed_anchor_atoms != expected_template_atoms:
        can_keep_existing_anchor = (
            state.anchor_frac is not None
            and state.anchor_atomic_numbers is not None
            and state.anchor_cell is not None
            and state.anchor_k is not None
            and state.anchor_assignment is not None
            and int(state.anchor_atomic_numbers.shape[0]) == expected_template_atoms
            and int(state.anchor_assignment.numel()) == expected_template_atoms
        )
        if can_keep_existing_anchor:
            return replace(
                state,
                free_vars=free_vars,
                lattice_free_vars=lattice_free,
            )
        raise RuntimeError(
            "DPnPSVD retargeted anchor atom count does not match the current template atom count: "
            f"target_repr={target_name!r}, "
            f"template_atoms={expected_template_atoms}, "
            f"refreshed_anchor_atoms={refreshed_anchor_atoms}."
        )

    try:
        anchor_assignment = _species_assignment_indices(
            source_frac=expansion.frac_coords,
            source_atomic_numbers=expansion.atomic_numbers,
            target_frac=anchor_frac,
            target_atomic_numbers=anchor_atomic_numbers,
        )
    except RuntimeError:
        can_keep_existing_anchor = (
            state.anchor_frac is not None
            and state.anchor_atomic_numbers is not None
            and state.anchor_cell is not None
            and state.anchor_k is not None
            and state.anchor_assignment is not None
            and int(state.anchor_atomic_numbers.shape[0]) == expected_template_atoms
            and int(state.anchor_assignment.numel()) == expected_template_atoms
        )
        if can_keep_existing_anchor:
            return replace(
                state,
                free_vars=free_vars,
                lattice_free_vars=lattice_free,
            )
        raise

    return replace(
        state,
        free_vars=free_vars,
        lattice_free_vars=lattice_free,
        anchor_frac=anchor_frac.detach().clone(),
        anchor_atomic_numbers=anchor_atomic_numbers.detach().clone(),
        anchor_cell=anchor_cell.detach().clone(),
        anchor_k=anchor_k.detach().clone(),
        anchor_assignment=anchor_assignment.detach().clone(),
        anchor_lattice_free_vars=k_to_free_vars(anchor_k, state.constraint).detach().clone(),
        reference_volume=float(_cell_volume(anchor_cell).detach().item()),
    )


def _state_to_theta(state: _PCSChartState) -> torch.Tensor:
    return torch.cat([state.free_vars.reshape(-1), state.lattice_free_vars.reshape(-1)], dim=0)


def _state_with_theta(state: _PCSChartState, theta: torch.Tensor) -> _PCSChartState:
    free_dim = int(state.template.total_free_dims)
    return replace(
        state,
        free_vars=theta[:free_dim].detach().clone(),
        lattice_free_vars=theta[free_dim:].detach().clone(),
    )


def _projection_state(state: _PCSChartState) -> PCSTemplateState:
    return PCSTemplateState(
        template=state.template,
        constraint=state.constraint,
        bridge=state.bridge,
        free_vars=state.free_vars.detach().clone(),
        lattice_free_vars=state.lattice_free_vars.detach().clone(),
        objective=float(state.objective),
        template_rank=int(state.template_rank),
        candidate_count=int(state.candidate_count),
        target_centering_symbol=state.target_centering_symbol,
        target_centering_translations=(
            state.target_centering_translations.detach().clone()
            if state.target_centering_translations is not None
            else None
        ),
    )


def _materialize_state(
    *,
    state: _PCSChartState,
    lattice_transform: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    free_vars = state.free_vars.detach().clone()
    expansion = expand_wyckoff_template_torch(
        template=state.template,
        free_vars=free_vars,
    )
    projected_cell = k_to_cell_matrix(
        free_vars_to_k(state.lattice_free_vars.detach().clone(), state.constraint)
    )
    if bool(getattr(state, "constraint", None) is not None) and int(state.constraint.space_group) == 227:
        atom_count = int(expansion.atomic_numbers.shape[0]) if expansion.atomic_numbers.ndim > 0 else 1
        print(f"sg {int(state.constraint.space_group)}", flush=True)
        print(f"template {_template_signature_labels(state.template)}", flush=True)
        print(f"num_atoms {atom_count}", flush=True)
        print(f"cell {projected_cell.detach().cpu().tolist()}", flush=True)
        print(f"frac finite {bool(torch.isfinite(expansion.frac_coords).all().item())}", flush=True)
        print(f"lattice finite {bool(torch.isfinite(projected_cell).all().item())}", flush=True)
    projected_structure = _build_structure_from_standardized_projection(
        frac_coords=expansion.frac_coords,
        atomic_numbers=expansion.atomic_numbers,
        cell_matrix=projected_cell,
    )
    target_representation_name = state.target_representation_name or "standardized"
    expected_atomic_numbers = np.asarray(state.bridge.vanilla_atomic_numbers, dtype=int)
    projected_structure_for_output = projected_structure
    preserve_expanded_representation = target_representation_name in {"raw_requested_expanded", "expanded"}
    if not preserve_expanded_representation and state.target_centering_translations is not None:
        try:
            projected_structure_for_output = _collapse_centering_equivalent_structure(
                structure=projected_structure_for_output,
                translations=state.target_centering_translations,
                expected_atomic_numbers=expected_atomic_numbers,
            )
        except Exception:
            projected_structure_for_output = projected_structure
    if not preserve_expanded_representation:
        try:
            mapped_structure = map_standardized_structure_to_vanilla_frame(
                standardized_structure=projected_structure_for_output,
                vanilla_reference_structure=state.bridge.vanilla_structure,
                symprec=state.bridge.symprec,
                angle_tolerance=state.bridge.angle_tolerance,
            )
            mapped_atomic_numbers = np.asarray(mapped_structure.atomic_numbers, dtype=int)
            if mapped_atomic_numbers.shape == expected_atomic_numbers.shape and np.array_equal(
                np.sort(mapped_atomic_numbers),
                np.sort(expected_atomic_numbers),
            ):
                projected_structure_for_output = mapped_structure
        except Exception:
            pass
    pos_half, l_half, h_half = vanilla_structure_to_model_tensors(
        structure=projected_structure_for_output,
        lattice_transform=lattice_transform,
        device=device,
        dtype=dtype,
    )
    if preserve_expanded_representation:
        template_n = int(state.template.total_atoms)
        out_n = int(h_half.shape[0])
        if out_n != template_n:
            raise RuntimeError(
                "Expanded representation mismatch: "
                f"template_n={template_n}, out_n={out_n}, "
                f"representation={target_representation_name}."
            )
    return pos_half, l_half, h_half, projected_structure_for_output


def _materialize_state_for_batch_output(
    *,
    state: _PCSChartState,
    lattice_transform: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_out, l_out, h_out, structure_out = _materialize_state(
        state=state,
        lattice_transform=lattice_transform,
        device=device,
        dtype=dtype,
    )
    expected_atomic_numbers = np.asarray(state.bridge.vanilla_atomic_numbers, dtype=int)
    expected_num_atoms = int(expected_atomic_numbers.shape[0])
    if int(h_out.shape[0]) == expected_num_atoms:
        return pos_out, l_out, h_out

    candidate_structures: list[Any] = []
    seen_signatures: set[tuple[tuple[int, ...], tuple[float, ...]]] = set()

    def _push_candidate(structure_candidate: Any) -> None:
        atomic_numbers = np.asarray(structure_candidate.atomic_numbers, dtype=int)
        frac = np.mod(np.asarray(structure_candidate.frac_coords, dtype=float), 1.0)
        signature = (
            tuple(int(v) for v in atomic_numbers.tolist()),
            tuple(float(v) for v in np.round(frac.reshape(-1), 6).tolist()),
        )
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            candidate_structures.append(structure_candidate)

    _push_candidate(structure_out)
    if state.target_centering_translations is not None:
        try:
            _push_candidate(
                _collapse_centering_equivalent_structure(
                    structure=structure_out,
                    translations=state.target_centering_translations,
                    expected_atomic_numbers=expected_atomic_numbers,
                )
            )
        except Exception:
            pass
    if state.target_centering_symbol not in {None, "", "P"}:
        try:
            primitive_centered = _structure_to_primitive_centering_basis(
                structure=structure_out,
                centering_symbol=state.target_centering_symbol,
                expected_atomic_numbers=expected_atomic_numbers,
            )
            if primitive_centered is not None:
                _push_candidate(primitive_centered)
        except Exception:
            pass

    for base_candidate in list(candidate_structures):
        try:
            _push_candidate(
                map_standardized_structure_to_vanilla_frame(
                    standardized_structure=base_candidate,
                    vanilla_reference_structure=state.bridge.vanilla_structure,
                    symprec=state.bridge.symprec,
                    angle_tolerance=state.bridge.angle_tolerance,
                )
            )
        except Exception:
            pass

    expected_atomic_numbers_t = torch.as_tensor(expected_atomic_numbers, device=device, dtype=torch.long)
    target_volume = float(abs(np.linalg.det(np.asarray(state.bridge.vanilla_structure.lattice.matrix, dtype=float))))
    scored_candidates: list[tuple[int, int, float, int, Any]] = []
    for idx, candidate_structure in enumerate(candidate_structures):
        candidate_atomic_numbers = np.asarray(candidate_structure.atomic_numbers, dtype=int)
        if candidate_atomic_numbers.shape != expected_atomic_numbers.shape or not np.array_equal(
            np.sort(candidate_atomic_numbers),
            np.sort(expected_atomic_numbers),
        ):
            continue
        try:
            validation = validate_requested_space_group(
                structure=candidate_structure,
                requested_space_group=int(state.constraint.space_group),
                expected_atomic_numbers=expected_atomic_numbers_t,
                symprec=state.bridge.symprec,
                angle_tolerance=state.bridge.angle_tolerance,
            )
            requested_match = int(bool(validation.requested_space_group_match))
            detected_space_group = int(validation.detected_space_group) if validation.detected_space_group is not None else -1
        except Exception:
            requested_match = 0
            detected_space_group = -1
        volume = float(abs(np.linalg.det(np.asarray(candidate_structure.lattice.matrix, dtype=float))))
        volume_error = abs(volume - target_volume)
        scored_candidates.append((requested_match, detected_space_group, -volume_error, -idx, candidate_structure))

    if not scored_candidates:
        raise RuntimeError(
            "Unable to map DPnPSVD batch output back to the vanilla graph frame: "
            f"expected_atoms={expected_num_atoms}, "
            f"candidate_atoms={[int(np.asarray(candidate.atomic_numbers, dtype=int).shape[0]) for candidate in candidate_structures]}, "
            f"target_representation={state.target_representation_name!r}."
        )

    scored_candidates.sort(reverse=True)
    candidate_structure = scored_candidates[0][-1]

    return vanilla_structure_to_model_tensors(
        structure=candidate_structure,
        lattice_transform=lattice_transform,
        device=device,
        dtype=dtype,
    )


def _invalid_sample_like(
    *,
    pos_ref: torch.Tensor,
    l_ref: torch.Tensor,
    h_ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_invalid = torch.full_like(pos_ref, float("nan"))
    l_invalid = torch.full_like(l_ref.view(1, -1), float("nan"))
    return pos_invalid, l_invalid, h_ref.detach().clone()


def _theta_difference(theta: torch.Tensor, mean: torch.Tensor, free_dim: int) -> torch.Tensor:
    diff = theta - mean
    if free_dim > 0:
        diff = diff.clone()
        diff[:free_dim] = _wrap_delta(diff[:free_dim])
    return diff


def _proposal_view(
    *,
    state: _PCSChartState,
    theta: torch.Tensor,
    cfg: DPnPSVDConfig,
) -> _ProjectionView:
    free_dim = int(state.template.total_free_dims)
    free_vars = theta[:free_dim]
    lattice_free = theta[free_dim:]
    expansion = expand_wyckoff_template_torch(template=state.template, free_vars=free_vars)
    projected_k = free_vars_to_k(lattice_free, state.constraint)
    projected_cell = k_to_cell_matrix(projected_k)

    matched_anchor = state.anchor_frac.to(device=theta.device, dtype=theta.dtype)[
        state.anchor_assignment.to(device=theta.device, dtype=torch.long)
    ]
    coord_residual = _wrap_delta(expansion.frac_coords - matched_anchor).reshape(-1)
    residual_parts: list[torch.Tensor] = []
    if float(cfg.coord_weight) > 0.0 and coord_residual.numel() > 0:
        residual_parts.append(math.sqrt(float(cfg.coord_weight)) * coord_residual)
    if float(cfg.lattice_weight) > 0.0 and lattice_free.numel() > 0:
        anchor_lattice = state.anchor_lattice_free_vars.to(device=theta.device, dtype=theta.dtype)
        residual_parts.append(math.sqrt(float(cfg.lattice_weight)) * (lattice_free - anchor_lattice))
    if float(cfg.residual_volume_weight) > 0.0:
        anchor_volume = torch.abs(
            torch.linalg.det(state.anchor_cell.to(device=theta.device, dtype=theta.dtype))
        ).clamp_min(1.0e-8)
        current_volume = torch.abs(torch.linalg.det(projected_cell)).clamp_min(1.0e-8)
        residual_parts.append(
            math.sqrt(float(cfg.residual_volume_weight))
            * torch.log(current_volume / anchor_volume).reshape(1)
        )
    residual = torch.cat([part.reshape(-1) for part in residual_parts], dim=0) if residual_parts else theta.new_zeros((1,))

    pair_distances = _periodic_pairwise_distances(
        frac_coords=expansion.frac_coords,
        cell_matrix=projected_cell,
    )
    if pair_distances.numel() > 0:
        min_pair = float(pair_distances.min().detach().item())
    else:
        min_pair = float("inf")

    if float(cfg.steric_weight) > 0.0:
        steric_loss = _soft_steric_overlap_loss(
            distances=pair_distances,
            min_distance=float(cfg.min_distance),
            tau=float(cfg.steric_softplus_tau),
        )
    else:
        steric_loss = theta.new_zeros(())

    if float(cfg.pair_distance_weight) > 0.0:
        pair_distance_loss = _pair_distance_consistency_loss(
            frac_coords=expansion.frac_coords,
            cell_matrix=projected_cell,
            anchor_frac_coords=matched_anchor,
            anchor_cell_matrix=state.anchor_cell.to(device=theta.device, dtype=theta.dtype),
        )
    else:
        pair_distance_loss = theta.new_zeros(())

    if float(cfg.volume_weight) > 0.0:
        volume_loss = _volume_ratio_loss(
            projected_cell=projected_cell,
            reference_volume=float(state.reference_volume) if state.reference_volume is not None else None,
            min_ratio=float(cfg.volume_ratio_min),
            max_ratio=float(cfg.volume_ratio_max),
        )
    else:
        volume_loss = theta.new_zeros(())

    return _ProjectionView(
        frac_coords=expansion.frac_coords,
        projected_cell=projected_cell,
        residual=residual,
        min_pair_distance=min_pair,
        steric_loss=steric_loss,
        pair_distance_loss=pair_distance_loss,
        volume_loss=volume_loss,
    )


def _theta_debug_stats(
    *,
    state: _PCSChartState,
    theta: torch.Tensor,
    cfg: DPnPSVDConfig,
) -> _ThetaDebugStats:
    view = _proposal_view(state=state, theta=theta, cfg=cfg)
    free_dim = int(state.template.total_free_dims)
    free_vars = theta[:free_dim]
    lattice_free = theta[free_dim:]
    coord_residual_norm = 0.0
    if int(view.frac_coords.shape[0]) == int(state.anchor_assignment.numel()):
        matched_anchor = state.anchor_frac.to(device=theta.device, dtype=theta.dtype)[
            state.anchor_assignment.to(device=theta.device, dtype=torch.long)
        ]
        coord_residual_norm = float(_wrap_delta(view.frac_coords - matched_anchor).reshape(-1).norm().detach().item())
    anchor_lattice = state.anchor_lattice_free_vars.to(device=theta.device, dtype=theta.dtype)
    lattice_residual = lattice_free - anchor_lattice
    lattice_residual_norm = float(lattice_residual.norm().detach().item()) if lattice_residual.numel() > 0 else 0.0
    anchor_volume = float(
        torch.abs(torch.linalg.det(state.anchor_cell.to(device=theta.device, dtype=theta.dtype))).clamp_min(1.0e-8).item()
    )
    current_volume_tensor = torch.abs(torch.linalg.det(view.projected_cell)).clamp_min(1.0e-8)
    current_volume = float(current_volume_tensor.detach().item())
    max_lattice_length = float(torch.linalg.norm(view.projected_cell, dim=1).max().detach().item())
    volume_residual = float(torch.log(current_volume_tensor / current_volume_tensor.new_tensor(anchor_volume)).detach().item())
    return _ThetaDebugStats(
        residual_norm=float(view.residual.norm().detach().item()) if view.residual.numel() > 0 else 0.0,
        coord_residual_norm=coord_residual_norm,
        lattice_residual_norm=lattice_residual_norm,
        volume_residual=volume_residual,
        min_pair_distance=float(view.min_pair_distance),
        steric_loss=float(view.steric_loss.detach().item()),
        pair_distance_loss=float(view.pair_distance_loss.detach().item()),
        volume_loss=float(view.volume_loss.detach().item()),
        cell_volume=current_volume,
        max_lattice_length=max_lattice_length,
        free_norm=float(free_vars.norm().detach().item()) if free_vars.numel() > 0 else 0.0,
        lattice_free_norm=float(lattice_free.norm().detach().item()) if lattice_free.numel() > 0 else 0.0,
    )


def _pair_distance_consistency_loss(
    *,
    frac_coords: torch.Tensor,
    cell_matrix: torch.Tensor,
    anchor_frac_coords: torch.Tensor,
    anchor_cell_matrix: torch.Tensor,
) -> torch.Tensor:
    pair_distances = _periodic_pairwise_distances(
        frac_coords=frac_coords,
        cell_matrix=cell_matrix,
    ).reshape(-1)
    anchor_pair_distances = _periodic_pairwise_distances(
        frac_coords=anchor_frac_coords,
        cell_matrix=anchor_cell_matrix,
    ).reshape(-1)
    count = min(int(pair_distances.numel()), int(anchor_pair_distances.numel()))
    if count <= 0:
        return frac_coords.new_zeros(())
    pair_distances = torch.sort(pair_distances)[0][:count]
    anchor_pair_distances = torch.sort(anchor_pair_distances)[0][:count]
    return F.mse_loss(pair_distances, anchor_pair_distances)


def _chart_ambient_view(
    *,
    state: _PCSChartState,
    theta: torch.Tensor,
    lattice_transform: Any,
) -> _ChartAmbientView:
    free_dim = int(state.template.total_free_dims)
    free_vars = theta[:free_dim]
    lattice_free = theta[free_dim:]
    expansion = expand_wyckoff_template_torch(template=state.template, free_vars=free_vars)
    projected_k = free_vars_to_k(lattice_free, state.constraint)
    projected_cell = k_to_cell_matrix(projected_k)
    lattice_features = _encode_lattice_features(
        cell_matrix=projected_cell,
        num_atoms=int(state.bridge.vanilla_atomic_numbers.shape[0]),
        lattice_transform=lattice_transform,
    ).reshape(-1)
    return _ChartAmbientView(
        frac_coords=expansion.frac_coords,
        lattice_features=lattice_features,
        projected_cell=projected_cell,
        atomic_numbers=expansion.atomic_numbers,
    )


def _metric_info(
    *,
    state: _PCSChartState,
    theta: torch.Tensor,
    cfg: DPnPSVDConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dim = int(theta.numel())
    eye = torch.eye(dim, device=theta.device, dtype=theta.dtype)
    if dim == 0:
        return eye, eye, theta.new_zeros(()), theta.new_zeros((0,))

    def residual_fn(th: torch.Tensor) -> torch.Tensor:
        return _proposal_view(state=state, theta=th, cfg=cfg).residual

    jac = torch.autograd.functional.jacobian(
        residual_fn,
        theta.detach().clone().requires_grad_(True),
    )
    jac = torch.nan_to_num(jac.reshape(-1, dim), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        _u, singular_values, vh = torch.linalg.svd(jac, full_matrices=False)
    except RuntimeError:
        jac = jac + 1.0e-6 * torch.randn_like(jac)
        _u, singular_values, vh = torch.linalg.svd(jac, full_matrices=False)
    v = vh.transpose(0, 1)
    denom = singular_values.square() + float(cfg.svd_damping)
    precond = v @ torch.diag(1.0 / denom) @ v.transpose(0, 1)
    precond_sqrt = v @ torch.diag(torch.rsqrt(denom)) @ v.transpose(0, 1)
    precond = 0.5 * (precond + precond.transpose(0, 1))
    precond_sqrt = 0.5 * (precond_sqrt + precond_sqrt.transpose(0, 1))
    log_jacobian = 0.5 * torch.log(denom.clamp_min(1.0e-12)).sum()
    return precond, precond_sqrt, log_jacobian, singular_values


def _target_energy(
    *,
    state: _PCSChartState,
    theta: torch.Tensor,
    eta: float,
    cfg: DPnPSVDConfig,
    include_jacobian: bool,
    log_jacobian: torch.Tensor | None = None,
) -> torch.Tensor:
    view = _proposal_view(state=state, theta=theta, cfg=cfg)
    prox_energy = view.residual.square().sum() / (2.0 * max(float(eta) ** 2, 1.0e-8))
    energy = (
        prox_energy
        + float(cfg.steric_weight) * view.steric_loss
        + float(cfg.pair_distance_weight) * view.pair_distance_loss
        + float(cfg.volume_weight) * view.volume_loss
    )
    if include_jacobian:
        if log_jacobian is None:
            _precond, _precond_sqrt, log_jacobian, _singular_values = _metric_info(state=state, theta=theta, cfg=cfg)
        energy = energy - log_jacobian
    if not torch.isfinite(energy):
        return theta.new_tensor(float("inf"))
    return energy


def _proposal_energy_and_grad(
    *,
    state: _PCSChartState,
    theta: torch.Tensor,
    eta: float,
    cfg: DPnPSVDConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta_var = theta.detach().clone().requires_grad_(True)
    view = _proposal_view(state=state, theta=theta_var, cfg=cfg)
    prox_energy = view.residual.square().sum() / (2.0 * max(float(eta) ** 2, 1.0e-8))
    energy = (
        prox_energy
        + float(cfg.steric_weight) * view.steric_loss
        + float(cfg.pair_distance_weight) * view.pair_distance_loss
        + float(cfg.volume_weight) * view.volume_loss
    )
    if not bool(getattr(energy, "requires_grad", False)):
        return theta.new_tensor(float("inf")), torch.zeros_like(theta)
    grad, = torch.autograd.grad(energy, theta_var)
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(energy):
        return theta.new_tensor(float("inf")), torch.zeros_like(theta)
    return energy.detach(), grad.detach()


def _gaussian_log_prob(
    *,
    value: torch.Tensor,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    free_dim: int,
) -> torch.Tensor:
    dim = int(value.numel())
    if dim == 0:
        return value.new_zeros(())
    cov = 0.5 * (covariance + covariance.transpose(0, 1))
    jitter = 1.0e-8
    eye = torch.eye(dim, device=value.device, dtype=value.dtype)
    chol = None
    for _ in range(6):
        try:
            chol = torch.linalg.cholesky(cov + jitter * eye)
            break
        except RuntimeError:
            jitter *= 10.0
    if chol is None:
        raise RuntimeError("Failed to stabilize Gaussian covariance in DPnPSVD.")
    diff = _theta_difference(value, mean, free_dim=free_dim).unsqueeze(-1)
    solved = torch.cholesky_solve(diff, chol)
    quad = torch.matmul(diff.transpose(0, 1), solved).reshape(())
    logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
    return -0.5 * (quad + logdet + dim * value.new_tensor(math.log(2.0 * math.pi)))


def _continuous_mala_step(
    *,
    state: _PCSChartState,
    theta: torch.Tensor,
    eta: float,
    cfg: DPnPSVDConfig,
) -> tuple[torch.Tensor, bool, dict[str, float | str]]:
    free_dim = int(state.template.total_free_dims)
    proposal_energy, grad = _proposal_energy_and_grad(state=state, theta=theta, eta=eta, cfg=cfg)
    if not bool(torch.isfinite(proposal_energy).item()):
        return theta.detach().clone(), False, {"reason": "proposal_energy_nonfinite"}
    precond, precond_sqrt, log_j_current, singular_values_current = _metric_info(state=state, theta=theta, cfg=cfg)
    target_energy_current = _target_energy(
        state=state,
        theta=theta,
        eta=eta,
        cfg=cfg,
        include_jacobian=True,
        log_jacobian=log_j_current.detach(),
    )
    if not bool(torch.isfinite(target_energy_current).item()):
        return theta.detach().clone(), False, {"reason": "target_energy_current_nonfinite"}

    mean_fwd = theta - 0.5 * float(cfg.svd_step_size) * (precond @ grad)
    noise = torch.randn_like(theta)
    theta_prop = mean_fwd + math.sqrt(float(cfg.svd_step_size)) * (precond_sqrt @ noise)
    if free_dim > 0:
        theta_prop = theta_prop.clone()
        theta_prop[:free_dim] = torch.remainder(theta_prop[:free_dim], 1.0)

    proposal_energy_prop, grad_prop = _proposal_energy_and_grad(state=state, theta=theta_prop, eta=eta, cfg=cfg)
    if not bool(torch.isfinite(proposal_energy_prop).item()):
        return theta.detach().clone(), False, {"reason": "proposal_energy_prop_nonfinite"}
    precond_prop, _precond_prop_sqrt, log_j_prop, singular_values_prop = _metric_info(
        state=state,
        theta=theta_prop,
        cfg=cfg,
    )
    target_energy_prop = _target_energy(
        state=state,
        theta=theta_prop,
        eta=eta,
        cfg=cfg,
        include_jacobian=True,
        log_jacobian=log_j_prop.detach(),
    )
    if not bool(torch.isfinite(target_energy_prop).item()):
        return theta.detach().clone(), False, {"reason": "target_energy_prop_nonfinite"}

    mean_rev = theta_prop - 0.5 * float(cfg.svd_step_size) * (precond_prop @ grad_prop)
    log_q_fwd = _gaussian_log_prob(
        value=theta_prop,
        mean=mean_fwd,
        covariance=float(cfg.svd_step_size) * precond,
        free_dim=free_dim,
    )
    log_q_rev = _gaussian_log_prob(
        value=theta,
        mean=mean_rev,
        covariance=float(cfg.svd_step_size) * precond_prop,
        free_dim=free_dim,
    )
    log_alpha = (-target_energy_prop + target_energy_current + log_q_rev - log_q_fwd).detach()
    debug_info: dict[str, float | str] = {
        "reason": "mh_reject",
        "proposal_energy_current": float(proposal_energy.item()),
        "proposal_energy_prop": float(proposal_energy_prop.item()),
        "target_energy_current": float(target_energy_current.item()),
        "target_energy_prop": float(target_energy_prop.item()),
        "log_q_fwd": float(log_q_fwd.item()),
        "log_q_rev": float(log_q_rev.item()),
        "log_alpha": float(log_alpha.item()),
        "grad_norm_current": float(grad.norm().detach().item()) if grad.numel() > 0 else 0.0,
        "grad_norm_prop": float(grad_prop.norm().detach().item()) if grad_prop.numel() > 0 else 0.0,
        "sv_min_current": float(singular_values_current.min().detach().item()) if singular_values_current.numel() > 0 else 0.0,
        "sv_max_current": float(singular_values_current.max().detach().item()) if singular_values_current.numel() > 0 else 0.0,
        "sv_min_prop": float(singular_values_prop.min().detach().item()) if singular_values_prop.numel() > 0 else 0.0,
        "sv_max_prop": float(singular_values_prop.max().detach().item()) if singular_values_prop.numel() > 0 else 0.0,
    }
    if math.log(max(float(torch.rand((), device=theta.device).item()), 1.0e-12)) < float(log_alpha.item()):
        debug_info["reason"] = "accept"
        return theta_prop.detach(), True, debug_info
    return theta.detach().clone(), False, debug_info


def _proposal_center_theta(state: _PCSChartState) -> torch.Tensor:
    return _state_to_theta(state)


def _sample_theta_proposal(
    *,
    center: torch.Tensor,
    free_dim: int,
    cfg: DPnPSVDConfig,
) -> torch.Tensor:
    theta = center.detach().clone()
    if free_dim > 0 and float(cfg.theta_proposal_free_std) > 0.0:
        theta[:free_dim] = torch.remainder(
            theta[:free_dim] + float(cfg.theta_proposal_free_std) * torch.randn_like(theta[:free_dim]),
            1.0,
        )
    if center.numel() > free_dim and float(cfg.theta_proposal_lattice_std) > 0.0:
        theta[free_dim:] = theta[free_dim:] + float(cfg.theta_proposal_lattice_std) * torch.randn_like(theta[free_dim:])
    return theta


def _theta_proposal_log_prob(
    *,
    theta: torch.Tensor,
    center: torch.Tensor,
    free_dim: int,
    cfg: DPnPSVDConfig,
) -> torch.Tensor:
    diff = _theta_difference(theta, center, free_dim=free_dim)
    log_prob = theta.new_zeros(())
    if free_dim > 0:
        sigma = max(float(cfg.theta_proposal_free_std), 1.0e-8)
        part = diff[:free_dim]
        log_prob = log_prob - 0.5 * (
            part.square().sum() / (sigma * sigma)
            + free_dim * math.log(2.0 * math.pi * sigma * sigma)
        )
    lattice_dim = int(theta.numel()) - int(free_dim)
    if lattice_dim > 0:
        sigma = max(float(cfg.theta_proposal_lattice_std), 1.0e-8)
        part = diff[free_dim:]
        log_prob = log_prob - 0.5 * (
            part.square().sum() / (sigma * sigma)
            + lattice_dim * math.log(2.0 * math.pi * sigma * sigma)
        )
    return log_prob


def _map_eta_to_kldm_time(
    *,
    model: Any,
    eta: float,
    num_atoms: torch.Tensor | None = None,
    ref_l: torch.Tensor | None = None,
) -> float:
    eta_clamped = float(max(1.0e-5, min(float(eta), 0.999)))
    device = next(model.parameters()).device
    dtype = torch.get_default_dtype()
    t_internal_grid = torch.linspace(0.0, float(model.tdm.T), 1024, device=device, dtype=dtype)
    sigma_r_grid = model.tdm.wrapped_gaussian_sigma_r_t(t_internal_grid)
    target = torch.tensor(eta_clamped, device=device, dtype=dtype)

    t_unit_grid = t_internal_grid / float(model.tdm.T)
    alpha_grid = model.diffusion_l.alpha(t_unit_grid)
    sigma_grid = model.diffusion_l.sigma(t_unit_grid)
    lattice_width_grid = sigma_grid / alpha_grid.clamp_min(1.0e-6)
    if hasattr(model.diffusion_l, "mu_sigma_n") and num_atoms is not None:
        ref = ref_l if ref_l is not None else torch.zeros((int(num_atoms.shape[0]), 6), device=device, dtype=dtype)
        _, sigma_n = model.diffusion_l.mu_sigma_n(
            num_atoms=num_atoms.to(device=device),
            ref=ref.to(device=device, dtype=dtype),
        )
        lattice_scale = torch.median(sigma_n.to(device=device, dtype=dtype)).clamp_min(1.0e-6)
        lattice_width_grid = lattice_width_grid * lattice_scale

    combined_error = (sigma_r_grid - target).square() + (lattice_width_grid - target).square()
    idx = int(torch.argmin(combined_error).item())
    t_internal = float(t_internal_grid[idx].item())
    return max(1.0e-3, min(t_internal / float(model.tdm.T), 1.0))


def _map_eta_to_vp_time(
    *,
    model: Any,
    eta: float,
) -> float:
    eta_clamped = float(max(1.0e-5, min(float(eta), 0.999)))
    device = next(model.parameters()).device
    dtype = torch.get_default_dtype()
    t_grid = torch.linspace(1.0e-4, 1.0, 4096, device=device, dtype=dtype)
    alpha_grid = model.diffusion_l.alpha(t_grid)
    sigma_grid = model.diffusion_l.sigma(t_grid)
    eta_grid = sigma_grid / alpha_grid.clamp_min(1.0e-6)
    idx = int(torch.argmin((eta_grid - eta_clamped).square()).item())
    return float(t_grid[idx].item())


def _sample_conditional_velocity(
    *,
    model: Any,
    batch: Any,
    f_t: torch.Tensor,
    l_t: torch.Tensor,
    h_t: torch.Tensor,
    t_graph: float,
    cfg: DPnPSVDConfig,
) -> torch.Tensor:
    node_index = batch.batch
    times = model._prepare_csp_sampling(
        batch=batch,
        n_steps=1,
        t_start=max(float(t_graph), float(cfg.ambient_dds_t_final) + 1.0e-6),
        t_final=float(cfg.ambient_dds_t_final),
        space_group=(
            torch.as_tensor(batch.space_group, device=f_t.device, dtype=torch.long).reshape(-1)
            if bool(cfg.sg_conditioned_dds) and hasattr(batch, "space_group")
            else None
        ),
        sg_guidance_scale=float(cfg.sg_guidance_scale),
    )
    score_network = times["score_network"]
    restore_training = bool(times["restore_training"])
    v_t = model.tdm.sample_velocity_noise(f_t, index=node_index)
    try:
        with torch.no_grad():
            batch_times = times["sampling_time_grid"]
            del batch_times
            graph_times = torch.full((int(batch.num_graphs), 1), float(t_graph), device=f_t.device, dtype=f_t.dtype)
            node_times = graph_times[node_index].squeeze(-1)
            for _ in range(max(int(cfg.ambient_dds_velocity_steps), 0)):
                preds = model._score_network_forward(
                    t=graph_times,
                    pos=f_t,
                    v=v_t,
                    h=h_t,
                    l=l_t,
                    node_index=node_index,
                    edge_node_index=batch.edge_node_index,
                    space_group=times.get("space_group"),
                    sg_guidance_scale=float(times.get("sg_guidance_scale", 1.0)),
                )
                score_v = model.tdm.reconstruct_full_reverse_velocity_score(
                    t=node_times,
                    v_t=v_t,
                    pred_v=preds["v"],
                    index=node_index,
                )
                noise_v = model.tdm.sample_velocity_noise(v_t, index=node_index)
                v_t = (
                    v_t
                    + float(cfg.ambient_dds_velocity_step_size) * score_v
                    + math.sqrt(2.0 * float(cfg.ambient_dds_velocity_step_size)) * noise_v
                )
                v_t = _center_per_graph(v_t, index=node_index)
    finally:
        if restore_training:
            score_network.train()
    return v_t


def _lattice_score_from_prediction(
    *,
    model: Any,
    t_graph: torch.Tensor,
    lattice_features: torch.Tensor,
    pred_l: torch.Tensor,
    num_atoms: torch.Tensor,
) -> torch.Tensor:
    diffusion_l = model.diffusion_l
    if isinstance(diffusion_l, ContinuousMattergenVPDiffusion):
        sigma_base_t = diffusion_l._match_dims(diffusion_l.sigma_base(t_graph), lattice_features)
        _mu_n, sigma_n = diffusion_l.mu_sigma_n(num_atoms=num_atoms, ref=lattice_features)
        sigma_n = diffusion_l._match_dims(sigma_n, lattice_features)
        return -pred_l / (sigma_base_t * sigma_n).clamp_min(diffusion_l.eps)

    sigma_t = diffusion_l._match_dims(diffusion_l.sigma(t_graph), lattice_features)
    if getattr(diffusion_l, "parameterization", "eps") == "eps":
        return -pred_l / sigma_t.clamp_min(diffusion_l.eps)
    alpha_t = diffusion_l._match_dims(diffusion_l.alpha(t_graph), lattice_features)
    return (alpha_t * pred_l - lattice_features) / sigma_t.pow(2).clamp_min(diffusion_l.eps)


def _chart_dds_metric_info(
    *,
    state: _PCSChartState,
    theta: torch.Tensor,
    lattice_transform: Any,
    cfg: DPnPSVDConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = int(theta.numel())
    eye = torch.eye(dim, device=theta.device, dtype=theta.dtype)
    if dim == 0:
        return eye, eye

    def residual_fn(th: torch.Tensor) -> torch.Tensor:
        view = _chart_ambient_view(state=state, theta=th, lattice_transform=lattice_transform)
        parts: list[torch.Tensor] = []
        if float(cfg.chart_dds_coord_weight) > 0.0 and view.frac_coords.numel() > 0:
            parts.append(math.sqrt(float(cfg.chart_dds_coord_weight)) * view.frac_coords.reshape(-1))
        if float(cfg.chart_dds_lattice_weight) > 0.0 and view.lattice_features.numel() > 0:
            parts.append(math.sqrt(float(cfg.chart_dds_lattice_weight)) * view.lattice_features.reshape(-1))
        if not parts:
            return th.new_zeros((1,))
        return torch.cat(parts, dim=0)

    jac = torch.autograd.functional.jacobian(
        residual_fn,
        theta.detach().clone().requires_grad_(True),
    )
    jac = torch.nan_to_num(jac.reshape(-1, dim), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        _u, singular_values, vh = torch.linalg.svd(jac, full_matrices=False)
    except RuntimeError:
        jac = jac + 1.0e-6 * torch.randn_like(jac)
        _u, singular_values, vh = torch.linalg.svd(jac, full_matrices=False)
    v = vh.transpose(0, 1)
    denom = singular_values.square() + float(cfg.chart_dds_damping)
    precond = v @ torch.diag(1.0 / denom) @ v.transpose(0, 1)
    precond_sqrt = v @ torch.diag(torch.rsqrt(denom)) @ v.transpose(0, 1)
    precond = 0.5 * (precond + precond.transpose(0, 1))
    precond_sqrt = 0.5 * (precond_sqrt + precond_sqrt.transpose(0, 1))
    return precond, precond_sqrt


def _dds_kldm_ambient_kernel(
    *,
    model: Any,
    graph_batch: Any,
    pos_half: torch.Tensor,
    l_half: torch.Tensor,
    h_half: torch.Tensor,
    cfg: DPnPSVDConfig,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if int(cfg.ambient_dds_steps) <= 0:
        return pos_half, l_half, h_half, {
            "prior_available": False,
            "prior_attempted_steps": 0,
            "prior_used_steps": 0,
            "compat_batch_steps": 0,
            "prior_last_reason": "ambient_dds_disabled",
        }

    prior_batch = _build_chart_compatible_batch(
        reference_batch=graph_batch,
        pos=pos_half,
        l=l_half,
        atomic_numbers=h_half,
    )
    t_k = _map_eta_to_kldm_time(
        model=model,
        eta=float(eta),
        num_atoms=prior_batch.num_atoms,
        ref_l=l_half.reshape(1, -1),
    )
    t_k = max(float(cfg.ambient_dds_t_final) + 1.0e-6, min(t_k, 1.0))
    t_tensor = torch.tensor([t_k], device=l_half.device, dtype=l_half.dtype)
    alpha_t = model.diffusion_l.alpha(t_tensor).reshape(1, 1)
    l_anchor = alpha_t * l_half.reshape(1, -1)
    f_anchor = pos_half
    v_anchor = _sample_conditional_velocity(
        model=model,
        batch=prior_batch,
        f_t=f_anchor,
        l_t=l_anchor,
        h_t=h_half,
        t_graph=t_k,
        cfg=cfg,
    )
    if bool(cfg.debug):
        print(
            f"kldm_dpnpsvd_ambient_dds_time eta={float(eta):.5f} t_k={t_k:.6f}",
            flush=True,
        )

    state = model._prepare_csp_sampling(
        batch=prior_batch,
        n_steps=max(1, int(cfg.ambient_dds_steps)),
        t_start=t_k,
        t_final=float(cfg.ambient_dds_t_final),
        space_group=(
            torch.as_tensor(prior_batch.space_group, device=l_half.device, dtype=torch.long).reshape(-1)
            if bool(cfg.sg_conditioned_dds) and hasattr(prior_batch, "space_group")
            else None
        ),
        sg_guidance_scale=float(cfg.sg_guidance_scale),
    )
    state["f_t"] = f_anchor.to(device=state["device"], dtype=state["dtype"])
    state["v_t"] = v_anchor.to(device=state["device"], dtype=state["dtype"])
    state["l_t"] = l_anchor.to(device=state["device"], dtype=state["batch"].l.dtype)
    state["a_t"] = h_half.to(device=state["device"], dtype=state["batch"].atomic_numbers.dtype)
    state = model._run_csp_em_reverse_chain(state)
    if state["restore_training"]:
        state["score_network"].train()

    pos_out = state["f_t"]
    l_out = state["l_t"]
    h_out = state["a_t"]
    if (
        not torch.isfinite(pos_out).all().item()
        or not torch.isfinite(l_out).all().item()
        or not torch.isfinite(h_out).all().item()
    ):
        return pos_half, l_half, h_half, {
            "prior_available": False,
            "prior_attempted_steps": 1,
            "prior_used_steps": 0,
            "compat_batch_steps": 1,
            "prior_last_reason": "ambient_dds_nonfinite",
        }

    return pos_out, l_out, h_out, {
        "prior_available": True,
        "prior_attempted_steps": 1,
        "prior_used_steps": 1,
        "compat_batch_steps": 1,
        "prior_last_reason": "ambient_reverse_ok",
    }


def _log_oracle_batch_metrics(
    *,
    graph_idx: int,
    phase: str,
    step_idx: int | None,
    total_steps: int | None,
    pred_f: torch.Tensor,
    pred_l: torch.Tensor,
    pred_a: torch.Tensor,
    requested_sg: int,
    oracle_target_frac: torch.Tensor | None,
    oracle_target_l: torch.Tensor | None,
    oracle_target_species: torch.Tensor | None,
    cfg: DPnPSVDConfig,
    lattice_transform: Any,
    previous_metrics: dict[str, float | None] | None = None,
) -> dict[str, float | None] | None:
    if not bool(cfg.debug) or not bool(cfg.debug_oracle_step_metrics):
        return None
    if oracle_target_frac is None or oracle_target_l is None or oracle_target_species is None:
        return None

    try:
        result = evaluate_csp_reconstruction(
            pred_f=pred_f,
            pred_l=pred_l.reshape(-1),
            pred_a=pred_a,
            target_f=oracle_target_frac,
            target_l=oracle_target_l.reshape(-1),
            target_a=oracle_target_species,
            lattice_transform=lattice_transform,
            requested_space_group=int(requested_sg),
            sg_symprec=float(cfg.symprec),
            sg_angle_tolerance=float(cfg.angle_tolerance),
            validity_cutoff=float(cfg.min_distance),
        )
    except Exception as exc:
        suffix = (
            ""
            if step_idx is None or total_steps is None
            else f" step={int(step_idx)}/{int(total_steps)}"
        )
        print(
            f"kldm_dpnpsvd_oracle_step graph={graph_idx + 1} phase={phase}{suffix} "
            f"status=failed reason={type(exc).__name__} detail={exc}",
            flush=True,
        )
        return None

    suffix = (
        ""
        if step_idx is None or total_steps is None
        else f" step={int(step_idx)}/{int(total_steps)}"
    )
    def _fmt(value: float | None, spec: str) -> str:
        return "na" if value is None else format(float(value), spec)
    diagnostics = result.matcher_diagnostics
    standardized_frac_rmse = (
        None if diagnostics is None else diagnostics.standardized_frac_rmse
    )
    metrics: dict[str, float | None] = {
        "frac_rmse": result.frac_rmse,
        "standardized_frac_rmse": standardized_frac_rmse,
        "min_pair_distance": result.min_pair_distance,
        "lattice_lengths_mae": result.lattice_lengths_mae,
        "lattice_angles_mae": result.lattice_angles_mae,
        "valid": float(int(result.valid)),
        "match": float(int(result.match)),
        "composition_match": None if result.composition_match is None else float(int(bool(result.composition_match))),
        "requested_sg_match": None
        if result.requested_space_group_match is None
        else float(int(bool(result.requested_space_group_match))),
    }
    print(
        f"kldm_dpnpsvd_oracle_step graph={graph_idx + 1} phase={phase}{suffix} "
        f"valid={int(result.valid)} match={int(result.match)} "
        f"composition_match={int(bool(result.composition_match)) if result.composition_match is not None else 'na'} "
        f"requested_sg_match={int(bool(result.requested_space_group_match)) if result.requested_space_group_match is not None else 'na'} "
        f"rmse={_fmt(result.rmse, '.6f')} "
        f"frac_rmse={_fmt(result.frac_rmse, '.6f')} "
        f"frac_status={result.frac_rmse_status or 'na'} "
        f"lengths_mae={_fmt(result.lattice_lengths_mae, '.6f')} "
        f"angles_mae={_fmt(result.lattice_angles_mae, '.6f')} "
        f"min_pair_distance={_fmt(result.min_pair_distance, '.4f')} "
        f"validity_reason={result.validity_reason or 'na'}",
        flush=True,
    )
    if previous_metrics is not None:
        def _delta(name: str) -> float | None:
            current = metrics.get(name)
            previous = previous_metrics.get(name)
            if current is None or previous is None:
                return None
            return float(current) - float(previous)

        def _trend(name: str, *, better_when: str) -> str:
            delta = _delta(name)
            if delta is None or not math.isfinite(delta):
                return "na"
            if abs(delta) < 1.0e-9:
                return "flat"
            if better_when == "lower":
                return "improved" if delta < 0.0 else "worsened"
            return "improved" if delta > 0.0 else "worsened"

        print(
            f"kldm_dpnpsvd_oracle_delta graph={graph_idx + 1} phase={phase}{suffix} "
            f"oracle_delta_frac_rmse={_fmt(_delta('frac_rmse'), '.6f')} "
            f"oracle_delta_std_frac_rmse={_fmt(_delta('standardized_frac_rmse'), '.6f')} "
            f"oracle_delta_min_pair={_fmt(_delta('min_pair_distance'), '.4f')} "
            f"oracle_delta_lengths_mae={_fmt(_delta('lattice_lengths_mae'), '.6f')} "
            f"oracle_delta_angles_mae={_fmt(_delta('lattice_angles_mae'), '.6f')}",
            flush=True,
        )
        print(
            f"kldm_dpnpsvd_oracle_effect graph={graph_idx + 1} phase={phase}{suffix} "
            f"valid={int(result.valid)} "
            f"match={int(result.match)} "
            f"requested_sg_match={int(bool(result.requested_space_group_match)) if result.requested_space_group_match is not None else 'na'} "
            f"composition_match={int(bool(result.composition_match)) if result.composition_match is not None else 'na'} "
            f"frac_effect={_trend('frac_rmse', better_when='lower')} "
            f"std_frac_effect={_trend('standardized_frac_rmse', better_when='lower')} "
            f"min_pair_effect={_trend('min_pair_distance', better_when='higher')} "
            f"lengths_effect={_trend('lattice_lengths_mae', better_when='lower')} "
            f"angles_effect={_trend('lattice_angles_mae', better_when='lower')}",
            flush=True,
        )
    if diagnostics is not None:
        print(
            f"kldm_dpnpsvd_oracle_matcher graph={graph_idx + 1} phase={phase}{suffix} "
            f"diagnosis={diagnostics.diagnosis} "
            f"standardized_match={int(diagnostics.conventional_match)} "
            f"primitive_match={int(diagnostics.primitive_match)} "
            f"standardized_frac_rmse={_fmt(diagnostics.standardized_frac_rmse, '.6f')} "
            f"standardized_frac_status={diagnostics.standardized_frac_status or 'na'} "
            f"std_pred_sg={diagnostics.standardized_predicted_space_group if diagnostics.standardized_predicted_space_group is not None else 'na'} "
            f"std_target_sg={diagnostics.standardized_target_space_group if diagnostics.standardized_target_space_group is not None else 'na'}",
            flush=True,
        )
    return metrics


def _log_oracle_step_metrics(
    *,
    graph_idx: int,
    phase: str,
    step_idx: int | None,
    total_steps: int | None,
    state: _PCSChartState,
    requested_sg: int,
    oracle_target_frac: torch.Tensor | None,
    oracle_target_l: torch.Tensor | None,
    oracle_target_species: torch.Tensor | None,
    lattice_transform: Any,
    cfg: DPnPSVDConfig,
    device: torch.device,
    dtype: torch.dtype,
    previous_metrics: dict[str, float | None] | None = None,
) -> dict[str, float | None] | None:
    pred_f, pred_l, pred_a = _materialize_state_for_batch_output(
        state=state,
        lattice_transform=lattice_transform,
        device=device,
        dtype=dtype,
    )
    return _log_oracle_batch_metrics(
        graph_idx=graph_idx,
        phase=phase,
        step_idx=step_idx,
        total_steps=total_steps,
        pred_f=pred_f,
        pred_l=pred_l,
        pred_a=pred_a,
        requested_sg=requested_sg,
        oracle_target_frac=oracle_target_frac,
        oracle_target_l=oracle_target_l,
        oracle_target_species=oracle_target_species,
        cfg=cfg,
        lattice_transform=lattice_transform,
        previous_metrics=previous_metrics,
    )


def _chart_dds_time_grid(
    *,
    t_start: float,
    t_final: float,
    n_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if n_steps <= 1 or not math.isfinite(t_start) or t_start <= 0.0:
        return torch.tensor([max(float(t_start), 1.0e-3)], device=device, dtype=dtype)
    t_floor = max(1.0e-6, min(float(t_final), float(t_start)))
    return torch.linspace(float(t_start), float(t_floor), steps=int(n_steps), device=device, dtype=dtype)


def _chart_dds_prior_direction(
    *,
    model: Any,
    graph_batch: Any,
    pos_current: torch.Tensor,
    l_current: torch.Tensor,
    h_current: torch.Tensor,
    cfg: DPnPSVDConfig,
    t_now: float,
) -> tuple[torch.Tensor, torch.Tensor, bool, str, bool]:
    zero_frac = torch.zeros_like(pos_current)
    zero_l = torch.zeros_like(l_current)
    t_start = float(max(1.0e-3, t_now))
    t_final = min(float(cfg.chart_dds_t_final), 0.5 * t_start)
    if t_start <= t_final:
        return zero_frac, zero_l, False, "time_too_small", False

    use_compat_batch = False
    compat_batch_reason = "native_batch"
    prior_batch = graph_batch
    if (
        tuple(pos_current.shape) != tuple(graph_batch.pos.shape)
        or tuple(l_current.shape) != tuple(graph_batch.l.shape)
        or tuple(h_current.shape) != tuple(graph_batch.atomic_numbers.shape)
    ):
        try:
            prior_batch = _build_chart_compatible_batch(
                reference_batch=graph_batch,
                pos=pos_current,
                l=l_current,
                atomic_numbers=h_current,
            )
            use_compat_batch = True
            compat_batch_reason = "compat_batch_rebuilt"
        except Exception as exc:
            return zero_frac, zero_l, False, f"compat_batch_failed:{type(exc).__name__}", False

    try:
        pos_prop, _v_prop, l_prop, h_prop = _kldm_dds_step(
            model=model,
            batch=prior_batch,
            pos_clean=pos_current,
            l_clean=l_current,
            h_clean=h_current,
            n_steps=max(1, int(cfg.chart_dds_kldm_steps)),
            t_start=t_start,
            t_final=t_final,
            space_group=(
                torch.as_tensor(prior_batch.space_group, device=pos_current.device, dtype=torch.long).reshape(-1)
                if bool(cfg.sg_conditioned_dds) and hasattr(prior_batch, "space_group")
                else None
            ),
            sg_guidance_scale=float(cfg.sg_guidance_scale),
        )
    except ValueError as exc:
        if "initialize_from_clean_state received" not in str(exc):
            raise
        return zero_frac, zero_l, False, f"{type(exc).__name__}", use_compat_batch

    species_match = (
        h_prop.shape == h_current.shape
        and bool(
            torch.equal(
                torch.sort(h_prop.detach().to(device="cpu", dtype=torch.long)).values,
                torch.sort(h_current.detach().to(device="cpu", dtype=torch.long)).values,
            )
        )
    )
    if (
        tuple(pos_prop.shape) != tuple(pos_current.shape)
        or tuple(l_prop.shape) != tuple(l_current.shape)
        or not species_match
    ):
        return zero_frac, zero_l, False, "proposal_representation_mismatch", use_compat_batch

    prior_sigma = max(float(cfg.chart_dds_prior_sigma), 1.0e-8)
    frac_dir = _wrap_delta(pos_prop - pos_current) / (prior_sigma * prior_sigma)
    lattice_dir = (l_prop - l_current) / (prior_sigma * prior_sigma)
    frac_dir = torch.nan_to_num(frac_dir, nan=0.0, posinf=0.0, neginf=0.0)
    lattice_dir = torch.nan_to_num(lattice_dir, nan=0.0, posinf=0.0, neginf=0.0)
    return frac_dir, lattice_dir, True, compat_batch_reason, use_compat_batch


def _chart_dds_step(
    *,
    model: Any,
    graph_batch: Any,
    state: _PCSChartState,
    theta_h: torch.Tensor,
    requested_sg: int,
    lattice_transform: Any,
    cfg: DPnPSVDConfig,
    eta: float,
    expected_atomic_numbers: torch.Tensor,
) -> tuple[torch.Tensor, _ChartDDSPriorStats]:
    # This is a chart-preserving pulled-back update: we combine an anchor term
    # with a KLDM-derived ambient prior direction, then pull that direction back
    # into chart coordinates and take a stochastic preconditioned step.
    #
    # When the current chart state cannot be represented by the ambient KLDM
    # batch shape (for example SG-227 expanded charts), the KLDM prior term
    # falls back to zero and the step becomes anchor-only inside the chart.
    if int(cfg.chart_dds_steps) <= 0:
        return theta_h.detach().clone(), _ChartDDSPriorStats()
    projection_steps = max(1, int(cfg.chart_dds_steps))
    if bool(cfg.debug):
        print(
            f"kldm_dpnpsvd_chart_dds_enter sg={int(requested_sg)} "
            f"steps={projection_steps} kldm_steps={int(cfg.chart_dds_kldm_steps)} "
            f"eta={float(eta):.5f} variant=pulled_back_anchored_chart",
            flush=True,
        )

    theta = theta_h.detach().clone()
    pcs_state = _state_with_theta(state, theta_h.detach())
    if bool(cfg.debug):
        print("kldm_dpnpsvd_chart_dds_materialize_anchor phase=start", flush=True)
    pos_half, l_half, h_half, structure_half = _materialize_state(
        state=pcs_state,
        lattice_transform=lattice_transform,
        device=theta.device,
        dtype=theta.dtype,
    )
    if bool(cfg.debug):
        print("kldm_dpnpsvd_chart_dds_materialize_anchor phase=done", flush=True)
    cell_half = torch.tensor(
        np.asarray(structure_half.lattice.matrix, dtype=float).copy(),
        device=theta.device,
        dtype=theta.dtype,
    )
    if bool(cfg.debug):
        print("kldm_dpnpsvd_chart_dds_anchor_repr phase=start", flush=True)
    anchor_frac_target, anchor_atomic_target, _anchor_cell_target, _anchor_k_target = _target_representation_for_state(
        state=state,
        frac_coords=pos_half,
        atomic_numbers=h_half,
        cell_matrix=cell_half,
        cfg=cfg,
    )
    if bool(cfg.debug):
        print("kldm_dpnpsvd_chart_dds_anchor_repr phase=done", flush=True)
    expansion_h = expand_wyckoff_template_torch(template=state.template, free_vars=pcs_state.free_vars)
    if bool(cfg.debug):
        print("kldm_dpnpsvd_chart_dds_anchor_assignment phase=start", flush=True)
    anchor_assignment = _species_assignment_indices(
        source_frac=expansion_h.frac_coords,
        source_atomic_numbers=expansion_h.atomic_numbers,
        target_frac=anchor_frac_target,
        target_atomic_numbers=anchor_atomic_target,
    )
    if bool(cfg.debug):
        print("kldm_dpnpsvd_chart_dds_anchor_assignment phase=done", flush=True)

    t_start = _map_eta_to_kldm_time(
        model=model,
        eta=float(eta),
        num_atoms=graph_batch.num_atoms,
        ref_l=l_half,
    )
    time_grid = _chart_dds_time_grid(
        t_start=t_start,
        t_final=float(cfg.chart_dds_t_final),
        n_steps=projection_steps,
        device=theta.device,
        dtype=theta.dtype,
    )
    anchor_eta = max(float(cfg.chart_dds_anchor_eta), 1.0e-8)
    device = theta.device
    prior_attempted_steps = 0
    prior_available_steps = 0
    compat_batch_steps = 0
    last_prior_reason = "not_run"

    for step_idx, t_now in enumerate(time_grid.tolist(), start=1):
        if bool(cfg.debug):
            print(
                f"kldm_dpnpsvd_chart_dds_project iter={step_idx}/{projection_steps} phase=start",
                flush=True,
            )
        theta_var = theta.detach().clone().requires_grad_(True)
        ambient = _chart_ambient_view(
            state=state,
            theta=theta_var,
            lattice_transform=lattice_transform,
        )
        matched_anchor = anchor_frac_target[anchor_assignment.to(device=theta.device, dtype=torch.long)]
        anchor_frac_score = -_wrap_delta(ambient.frac_coords - matched_anchor) / (anchor_eta * anchor_eta)
        anchor_l_score = -(ambient.lattice_features.reshape_as(l_half) - l_half) / (anchor_eta * anchor_eta)

        prior_frac_score, prior_l_score, prior_available, prior_reason, used_compat_batch = _chart_dds_prior_direction(
            model=model,
            graph_batch=graph_batch,
            pos_current=ambient.frac_coords.detach(),
            l_current=ambient.lattice_features.reshape_as(l_half).detach(),
            h_current=ambient.atomic_numbers.detach(),
            cfg=cfg,
            t_now=float(t_now),
        )
        prior_attempted_steps += 1
        last_prior_reason = prior_reason
        if prior_available:
            prior_available_steps += 1
        if used_compat_batch:
            compat_batch_steps += 1
        if bool(cfg.debug) and not prior_available:
            print(
                f"kldm_dpnpsvd_chart_dds_prior_skip iter={step_idx}/{projection_steps} "
                f"reason={prior_reason}",
                flush=True,
            )

        total_frac_score = (
            float(cfg.chart_dds_coord_weight)
            * (anchor_frac_score + prior_frac_score.to(device=theta.device, dtype=theta.dtype))
        )
        total_l_score = (
            float(cfg.chart_dds_lattice_weight)
            * (anchor_l_score + prior_l_score.to(device=theta.device, dtype=theta.dtype))
        )
        ambient_objective = (
            (ambient.frac_coords * total_frac_score.detach()).sum()
            + (ambient.lattice_features.reshape_as(l_half) * total_l_score.detach()).sum()
        )
        view = _proposal_view(state=state, theta=theta_var, cfg=cfg)
        total_objective = (
            ambient_objective
            - float(cfg.steric_weight) * view.steric_loss
            - float(cfg.volume_weight) * view.volume_loss
        )
        grad, = torch.autograd.grad(total_objective, theta_var)
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        precond, precond_sqrt = _chart_dds_metric_info(
            state=state,
            theta=theta_var.detach(),
            lattice_transform=lattice_transform,
            cfg=cfg,
        )
        noise = torch.randn_like(theta_var)
        theta_prop = (
            theta
            + float(cfg.chart_dds_step_size) * (precond @ grad.detach())
            + math.sqrt(2.0 * float(cfg.chart_dds_step_size)) * (precond_sqrt @ noise)
        )
        free_dim = int(state.template.total_free_dims)
        if free_dim > 0:
            theta_prop = theta_prop.clone()
            theta_prop[:free_dim] = torch.remainder(theta_prop[:free_dim], 1.0)

        if bool(cfg.chart_dds_reject_invalid):
            prop_state = _state_with_theta(state, theta_prop.detach())
            _pos_prop, _l_prop, _h_prop, structure_prop = _materialize_state(
                state=prop_state,
                lattice_transform=lattice_transform,
                device=device,
                dtype=theta.dtype,
            )
            validation_prop, min_pair_prop = _validate_projection(
                structure=structure_prop,
                requested_sg=int(requested_sg),
                expected_atomic_numbers=expected_atomic_numbers,
                cfg=cfg,
            )
            if bool(validation_prop.requested_space_group_match) and min_pair_prop >= float(cfg.min_distance):
                theta = theta_prop.detach()
        else:
            theta = theta_prop.detach()
        if bool(cfg.debug):
            print(
                f"kldm_dpnpsvd_chart_dds_project iter={step_idx}/{projection_steps} phase=done",
                flush=True,
            )

    final_state = _state_with_theta(state, theta.detach())
    _pos_final, _l_final, _h_final, structure_final = _materialize_state(
        state=final_state,
        lattice_transform=lattice_transform,
        device=device,
        dtype=theta.dtype,
    )
    validation_final, min_pair_final = _validate_projection(
        structure=structure_final,
        requested_sg=int(requested_sg),
        expected_atomic_numbers=expected_atomic_numbers,
        cfg=cfg,
    )
    if (
        not bool(validation_final.requested_space_group_match)
        or min_pair_final < float(cfg.min_distance)
    ):
        return theta_h.detach().clone(), _ChartDDSPriorStats(
            attempted_steps=prior_attempted_steps,
            available_steps=prior_available_steps,
            compat_batch_steps=compat_batch_steps,
            last_reason=last_prior_reason,
        )
    return theta.detach().clone(), _ChartDDSPriorStats(
        attempted_steps=prior_attempted_steps,
        available_steps=prior_available_steps,
        compat_batch_steps=compat_batch_steps,
        last_reason=last_prior_reason,
    )


def _kldm_dds_step(
    *,
    model: Any,
    batch: Any,
    pos_clean: torch.Tensor,
    l_clean: torch.Tensor,
    h_clean: torch.Tensor,
    n_steps: int,
    t_start: float,
    t_final: float,
    space_group: torch.Tensor | None = None,
    sg_guidance_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # DPnPSVD uses this as a tiny ambient KLDM prior proposal inside chart-DDS.
    # The chart kernel itself stays in theta-space; this helper only supplies a
    # local prior-side direction in ambient crystal coordinates.
    repair_state = model._prepare_csp_sampling(
        batch=batch,
        n_steps=n_steps,
        t_start=t_start,
        t_final=t_final,
        space_group=space_group,
        sg_guidance_scale=sg_guidance_scale,
        initial_f=pos_clean,
        initial_l=l_clean,
        initial_a=h_clean,
        initialize_from_dds_anchor=True,
    )
    repair_state = model._run_csp_em_reverse_chain(repair_state)
    if repair_state["restore_training"]:
        repair_state["score_network"].train()
    return repair_state["f_t"], repair_state["v_t"], repair_state["l_t"], repair_state["a_t"]


def _validate_projection(
    *,
    structure: Any,
    requested_sg: int,
    expected_atomic_numbers: torch.Tensor,
    cfg: DPnPSVDConfig,
) -> tuple[Any, float]:
    cell_np = np.asarray(structure.lattice.matrix, dtype=float).copy()
    frac_np = np.asarray(structure.frac_coords, dtype=float).copy()
    sane, reason, _volume, _max_lattice_length = _cell_sanity_status(
        cell_matrix=cell_np,
        frac_coords=frac_np,
    )
    if not sane:
        validation = SimpleNamespace(
            composition_match=False,
            requested_space_group=int(requested_sg),
            detected_space_group=None,
            requested_space_group_match=False,
            sanity_reason=str(reason),
        )
        return validation, 0.0
    validation = validate_requested_space_group(
        structure=structure,
        requested_space_group=int(requested_sg),
        expected_atomic_numbers=expected_atomic_numbers,
        symprec=float(cfg.symprec),
        angle_tolerance=float(cfg.angle_tolerance),
    )
    cell_half = torch.tensor(
        cell_np,
        device=expected_atomic_numbers.device,
        dtype=torch.float64,
    )
    frac_half = torch.tensor(
        frac_np,
        device=expected_atomic_numbers.device,
        dtype=torch.float64,
    )
    pair_distances = _periodic_pairwise_distances(frac_coords=frac_half, cell_matrix=cell_half)
    min_pair = float(pair_distances.min().detach().item()) if pair_distances.numel() > 0 else float("inf")
    return validation, min_pair


def _pcs_kernel(
    *,
    ranked_templates: list[_RankedTemplate],
    template_log_probs: torch.Tensor,
    current_idx: int,
    current_state: _PCSChartState,
    anchor_frac: torch.Tensor,
    anchor_l: torch.Tensor,
    anchor_species: torch.Tensor,
    requested_sg: int,
    lattice_transform: Any,
    cfg: DPnPSVDConfig,
    eta: float,
    num_steps: int,
    hard_reject_close_contacts: bool = False,
    hard_reject_distance: float | None = None,
) -> tuple[int, _PCSChartState]:
    device = anchor_frac.device
    dtype = anchor_frac.dtype
    if bool(cfg.debug):
        print(
            f"kldm_dpnpsvd_pcs_enter eta={float(eta):.5f} "
            f"template_idx={int(current_idx) + 1}/{len(ranked_templates)}",
            flush=True,
        )
    anchor_cell = _decode_lattice_matrix(
        l=anchor_l.reshape(-1),
        num_atoms=int(anchor_frac.shape[0]),
        lattice_transform=lattice_transform,
    ).to(device=device, dtype=dtype)
    theta = _state_to_theta(current_state)
    current_state = _retarget_state(
        state=current_state,
        theta=theta,
        frac_coords=anchor_frac,
        atomic_numbers=anchor_species,
        cell_matrix=anchor_cell,
        cfg=cfg,
    )
    theta = _state_to_theta(current_state)
    if bool(cfg.debug):
        stats = _theta_debug_stats(state=current_state, theta=theta, cfg=cfg)
        print(
            f"kldm_dpnpsvd_pcs_state eta={float(eta):.5f} template_idx={int(current_idx) + 1}/{len(ranked_templates)} "
            f"residual_norm={stats.residual_norm:.6f} coord_residual_norm={stats.coord_residual_norm:.6f} "
            f"lattice_residual_norm={stats.lattice_residual_norm:.6f} volume_residual={stats.volume_residual:.6f} "
            f"min_pair_distance={stats.min_pair_distance:.4f} steric_loss={stats.steric_loss:.6f} "
            f"pair_distance_loss={stats.pair_distance_loss:.6f} "
            f"volume_loss={stats.volume_loss:.6f} cell_volume={stats.cell_volume:.6f} "
            f"max_lattice_length={stats.max_lattice_length:.6f} free_norm={stats.free_norm:.6f} "
            f"lattice_free_norm={stats.lattice_free_norm:.6f}",
            flush=True,
        )

    proposal_cache: dict[int, _PCSChartState | None] = {int(current_idx): current_state}
    theta_accept = 0
    w_accept = 0
    w_attempt = 0
    theta_reject_reasons: dict[str, int] = {}
    last_theta_debug: dict[str, float | str] = {}

    anchor_species_counts: dict[int, int] = {}
    for atomic_number in current_state.anchor_atomic_numbers.detach().cpu().tolist():
        atomic_number = int(atomic_number)
        anchor_species_counts[atomic_number] = anchor_species_counts.get(atomic_number, 0) + 1

    def proposal_state(idx: int) -> _PCSChartState | None:
        if int(idx) in proposal_cache:
            return proposal_cache[int(idx)]
        item = ranked_templates[int(idx)]
        if _template_species_counts(item.template) != anchor_species_counts:
            if bool(cfg.debug):
                print(
                    f"kldm_dpnpsvd_skip_template eta={float(eta):.5f} template_idx={int(item.template_idx) + 1} "
                    "reason=species_count_mismatch",
                    flush=True,
                )
            proposal_cache[int(idx)] = None
            return None
        try:
            cached = _initialize_state_for_template(
                template=item.template,
                template_rank=int(item.template_idx + 1),
                candidate_count=int(len(ranked_templates)),
                anchor_frac=anchor_frac,
                anchor_species=anchor_species,
                anchor_l=anchor_l.reshape(-1),
                requested_sg=int(requested_sg),
                lattice_transform=lattice_transform,
                cfg=cfg,
                target_representation_name=current_state.target_representation_name,
                target_centering_symbol=current_state.target_centering_symbol,
                target_centering_translations=current_state.target_centering_translations,
            )
        except RuntimeError as exc:
            if "species counts differ" not in str(exc):
                raise
            if bool(cfg.debug):
                print(
                    f"kldm_dpnpsvd_skip_template eta={float(eta):.5f} template_idx={int(item.template_idx) + 1} "
                    "reason=assignment_init_failed",
                    flush=True,
                )
            proposal_cache[int(idx)] = None
            return None
        proposal_cache[int(idx)] = cached
        return cached

    def _cached_proposal_state(idx: int) -> _PCSChartState | None:
        if int(idx) == int(current_idx):
            return current_state
        return proposal_cache.get(int(idx))

    for _ in range(int(num_steps)):
        theta_prev = theta.detach().clone()
        state_prev = current_state
        theta, accepted, theta_debug = _continuous_mala_step(
            state=current_state,
            theta=theta,
            eta=float(eta),
            cfg=cfg,
        )
        theta_accept += int(bool(accepted))
        reason = str(theta_debug.get("reason", "unknown"))
        theta_reject_reasons[reason] = theta_reject_reasons.get(reason, 0) + 1
        last_theta_debug = theta_debug
        current_state = _state_with_theta(current_state, theta)
        if hard_reject_close_contacts or hard_reject_distance is not None:
            view_after = _proposal_view(state=current_state, theta=theta, cfg=cfg)
            reject_distance = (
                float(cfg.min_distance)
                if hard_reject_close_contacts
                else float(hard_reject_distance)
            )
            if float(view_after.min_pair_distance) < reject_distance:
                theta = theta_prev
                current_state = state_prev
                theta_accept -= int(bool(accepted))
                reject_reason = (
                    "hard_close_contact_reject"
                    if hard_reject_close_contacts
                    else "outer_close_contact_reject"
                )
                theta_reject_reasons[reject_reason] = (
                    theta_reject_reasons.get(reject_reason, 0) + 1
                )
                last_theta_debug = {"reason": reject_reason}
                continue

        if float(cfg.template_move_probability) <= 0.0:
            continue
        if float(torch.rand((), device=device).item()) >= float(cfg.template_move_probability):
            continue

        w_attempt += 1
        prop_idx = int(torch.multinomial(template_log_probs.exp(), 1).item())
        prop_template_state = proposal_state(prop_idx)
        if prop_template_state is None:
            continue
        prop_center = _proposal_center_theta(prop_template_state)
        prop_theta = _sample_theta_proposal(
            center=prop_center,
            free_dim=int(prop_template_state.template.total_free_dims),
            cfg=cfg,
        )
        prop_template_state = _state_with_theta(prop_template_state, prop_theta)

        current_center_state = _cached_proposal_state(current_idx)
        if current_center_state is None:
            current_center_state = current_state
        current_center = _proposal_center_theta(current_center_state)

        current_energy = _target_energy(
            state=current_state,
            theta=theta,
            eta=float(eta),
            cfg=cfg,
            include_jacobian=True,
        )
        proposed_energy = _target_energy(
            state=prop_template_state,
            theta=prop_theta,
            eta=float(eta),
            cfg=cfg,
            include_jacobian=True,
        )
        if not bool(torch.isfinite(proposed_energy).item()):
            continue

        log_q_forward = template_log_probs[prop_idx] + _theta_proposal_log_prob(
            theta=prop_theta,
            center=prop_center,
            free_dim=int(prop_template_state.template.total_free_dims),
            cfg=cfg,
        )
        log_q_reverse = template_log_probs[current_idx] + _theta_proposal_log_prob(
            theta=theta,
            center=current_center,
            free_dim=int(current_state.template.total_free_dims),
            cfg=cfg,
        )
        log_alpha = (-proposed_energy + current_energy + log_q_reverse - log_q_forward).detach()
        if math.log(max(float(torch.rand((), device=device).item()), 1.0e-12)) < float(log_alpha.item()):
            current_idx = int(prop_idx)
            current_state = prop_template_state
            theta = prop_theta.detach().clone()
            w_accept += 1

    if bool(cfg.debug):
        final_stats = _theta_debug_stats(state=current_state, theta=theta, cfg=cfg)
        reason_summary = ",".join(
            f"{key}:{value}"
            for key, value in sorted(theta_reject_reasons.items())
        )
        print(
            f"kldm_dpnpsvd_pcs eta={float(eta):.5f} template_idx={int(current_idx) + 1}/{len(ranked_templates)} "
            f"theta_accept={theta_accept}/{int(num_steps)} w_accept={w_accept}/{w_attempt} "
            f"w_attempt={w_attempt} move_prob={float(cfg.template_move_probability):.3f} "
            f"theta_reasons={reason_summary if reason_summary else 'none'} "
            f"last_log_alpha={float(last_theta_debug.get('log_alpha', float('nan'))):.6f} "
            f"last_target_current={float(last_theta_debug.get('target_energy_current', float('nan'))):.6f} "
            f"last_target_prop={float(last_theta_debug.get('target_energy_prop', float('nan'))):.6f} "
            f"last_sv_min={float(last_theta_debug.get('sv_min_current', float('nan'))):.6e} "
            f"last_sv_max={float(last_theta_debug.get('sv_max_current', float('nan'))):.6e} "
            f"final_residual_norm={final_stats.residual_norm:.6f} "
            f"final_min_pair_distance={final_stats.min_pair_distance:.4f} "
            f"final_cell_volume={final_stats.cell_volume:.6f}",
            flush=True,
        )
        close_contact_rejects = int(
            theta_reject_reasons.get("outer_close_contact_reject", 0)
            + theta_reject_reasons.get("hard_close_contact_reject", 0)
        )
        if close_contact_rejects > 0:
            print(
                f"kldm_dpnpsvd_pcs_alert eta={float(eta):.5f} "
                f"close_contact_rejects={close_contact_rejects}/{int(num_steps)} "
                f"final_min_pair_distance={final_stats.min_pair_distance:.4f}",
                flush=True,
            )
    return current_idx, current_state


def _run_fixed_template_pcs(
    *,
    ranked_item: _RankedTemplate,
    current_state: _PCSChartState,
    anchor_frac: torch.Tensor,
    anchor_l: torch.Tensor,
    anchor_species: torch.Tensor,
    requested_sg: int,
    lattice_transform: Any,
    cfg: DPnPSVDConfig,
    eta: float,
    num_steps: int,
    hard_reject_close_contacts: bool,
) -> _PCSChartState:
    fixed_log_probs = torch.zeros((1,), device=anchor_frac.device, dtype=anchor_frac.dtype)
    _fixed_idx, refined_state = _pcs_kernel(
        ranked_templates=[ranked_item],
        template_log_probs=fixed_log_probs,
        current_idx=0,
        current_state=current_state,
        anchor_frac=anchor_frac,
        anchor_l=anchor_l,
        anchor_species=anchor_species,
        requested_sg=int(requested_sg),
        lattice_transform=lattice_transform,
        cfg=cfg,
        eta=float(eta),
        num_steps=max(1, int(num_steps)),
        hard_reject_close_contacts=bool(hard_reject_close_contacts),
    )
    return refined_state


def _sample_graph(
    *,
    model: Any,
    graph_idx: int,
    graph_batch: Any,
    requested_sg: int,
    pos_prior: torch.Tensor,
    l_prior: torch.Tensor,
    h_prior: torch.Tensor,
    oracle_target_frac: torch.Tensor | None,
    oracle_target_l: torch.Tensor | None,
    oracle_target_species: torch.Tensor | None,
    lattice_transform: Any,
    cfg: DPnPSVDConfig,
    schedule: torch.Tensor,
    ranker: torch.nn.Module | None,
    template_cache: dict[str, Any] | None,
    template_prior: TemplatePrior | None,
    prior_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    device = pos_prior.device
    dtype = pos_prior.dtype
    template_atomic_numbers = requested_conventional_atomic_numbers(
        h_prior,
        space_group_number=int(requested_sg),
    ).to(device=h_prior.device, dtype=torch.long)
    templates = _cached_templates(
        space_group_number=int(requested_sg),
        atomic_numbers=template_atomic_numbers,
        max_templates=int(cfg.max_templates),
        template_nmax=int(cfg.template_nmax),
        quick=bool(cfg.quick_templates),
        template_cache=template_cache,
        template_cache_required=bool(cfg.template_cache_required),
    )
    if not templates:
        if bool(cfg.debug):
            print(
                f"kldm_dpnpsvd_catalog graph={graph_idx + 1} sg={requested_sg} status=no_templates",
                flush=True,
            )
        if bool(cfg.debug):
            print(
                f"kldm_dpnpsvd_fallback graph={graph_idx + 1} reason=no_requested_sg_state source=no_templates",
                flush=True,
            )
        pos_invalid, l_invalid, h_invalid = _invalid_sample_like(pos_ref=pos_prior, l_ref=l_prior, h_ref=h_prior)
        return pos_invalid, l_invalid, h_invalid, {
            "prior_available": False,
            "prior_attempted_steps": 0,
            "prior_used_steps": 0,
            "compat_batch_steps": 0,
            "prior_last_reason": "no_templates",
        }

    oracle_target_signature: tuple[tuple[int, str], ...] = ()
    oracle_source = "disabled"
    if bool(
        cfg.oracle_template_orbit_rerank
        or cfg.oracle_template_orbit_filter
        or str(getattr(cfg, "template_prior_mode", "dataset")).strip().lower() == "oracle_surrogate"
    ):
        if oracle_target_frac is not None and oracle_target_l is not None and oracle_target_species is not None:
            try:
                oracle_target_signature, oracle_source = _oracle_target_orbit_signature(
                    target_frac=oracle_target_frac,
                    target_l=oracle_target_l,
                    target_species=oracle_target_species,
                    lattice_transform=lattice_transform,
                    cfg=cfg,
                )
            except Exception as exc:
                oracle_source = f"failed:{type(exc).__name__}"

    ranked_templates, ranking_debug = _rank_templates(
        templates=templates,
        requested_sg=int(requested_sg),
        template_atomic_numbers=template_atomic_numbers,
        template_prior=template_prior,
        ranker=ranker,
        device=device,
        oracle_target_signature=oracle_target_signature,
        cfg=cfg,
    )
    if bool(cfg.oracle_template_orbit_filter) and oracle_target_signature and ranked_templates:
        best_mismatch = min(int(item.orbit_mismatch) for item in ranked_templates)
        ranked_templates = [
            item
            for item in ranked_templates
            if int(item.orbit_mismatch) == int(best_mismatch)
        ]
    template_log_probs = _proposal_template_log_probs(
        ranked_templates=ranked_templates,
        cfg=cfg,
        device=device,
    )
    current_idx = _initial_template_position(
        ranked_templates=ranked_templates,
        cfg=cfg,
        device=device,
    )
    best_available_state: _PCSChartState | None = None
    best_available_energy: float | None = None
    chosen: _RankedTemplate | None = None
    current_state: _PCSChartState | None = None
    candidate_order = [current_idx] + [idx for idx in range(len(ranked_templates)) if idx != current_idx]
    for idx in candidate_order:
        candidate = ranked_templates[idx]
        try:
            state = _initialize_state_for_template(
                template=candidate.template,
                template_rank=int(candidate.template_idx + 1),
                candidate_count=int(len(ranked_templates)),
                anchor_frac=pos_prior,
                anchor_species=h_prior,
                anchor_l=l_prior,
                requested_sg=int(requested_sg),
                lattice_transform=lattice_transform,
                cfg=cfg,
            )
        except RuntimeError as exc:
            if bool(cfg.debug):
                print(
                    f"kldm_dpnpsvd_skip_initial_template graph={graph_idx + 1} "
                    f"template_idx={int(candidate.template_idx) + 1} reason={type(exc).__name__} "
                    f"detail={exc}",
                    flush=True,
                )
                print(
                    f"kldm_dpnpsvd_skip_initial_template_context graph={graph_idx + 1} "
                    f"sg={int(requested_sg)} signature={_template_signature_labels(candidate.template)} "
                    f"template_atoms={int(candidate.template.total_atoms)} prior_atoms={int(h_prior.shape[0])}",
                    flush=True,
                )
                print(traceback.format_exc(), flush=True)
                if int(requested_sg) == 227:
                    cell_prior = _decode_lattice_matrix(
                        l=l_prior,
                        num_atoms=int(pos_prior.shape[0]),
                        lattice_transform=lattice_transform,
                    ).to(device=device, dtype=dtype)
                    print(f"sg {int(requested_sg)}", flush=True)
                    print(f"template {_template_signature_labels(candidate.template)}", flush=True)
                    print(f"num_atoms {int(h_prior.shape[0])}", flush=True)
                    print(f"cell {cell_prior.detach().cpu().tolist()}", flush=True)
                    print(f"frac finite {bool(torch.isfinite(pos_prior).all().item())}", flush=True)
                    print(f"lattice finite {bool(torch.isfinite(cell_prior).all().item())}", flush=True)
                print(
                    f"kldm_dpnpsvd_skip_initial_template graph={graph_idx + 1} "
                    f"template_idx={int(candidate.template_idx) + 1} status=skipped",
                    flush=True,
                )
            continue
        chosen = candidate
        current_idx = idx
        current_state = state
        init_energy = float(state.objective)
        if best_available_energy is None or init_energy < best_available_energy:
            best_available_state = replace(state)
            best_available_energy = init_energy
        break
    if chosen is None or current_state is None:
        if bool(cfg.debug):
            print(
                f"kldm_dpnpsvd_catalog graph={graph_idx + 1} sg={requested_sg} status=init_failed",
                flush=True,
            )
            print(
                f"kldm_dpnpsvd_fallback graph={graph_idx + 1} reason=no_requested_sg_state source=prior_init_failed",
                flush=True,
            )
        pos_invalid, l_invalid, h_invalid = _invalid_sample_like(pos_ref=pos_prior, l_ref=l_prior, h_ref=h_prior)
        return pos_invalid, l_invalid, h_invalid, {
            "prior_available": False,
            "prior_attempted_steps": 0,
            "prior_used_steps": 0,
            "compat_batch_steps": 0,
            "prior_last_reason": "prior_init_failed",
        }
    best_standardized_phase: dict[str, Any] | None = None
    best_matcher_phase: dict[str, Any] | None = None
    first_match_loss_phase: str | None = None
    previous_match_value: int | None = None

    def _phase_label(phase: str, step_idx: int | None, total_steps: int | None) -> str:
        if step_idx is None or total_steps is None:
            return phase
        return f"{phase}:{int(step_idx)}/{int(total_steps)}"

    def _record_first_match_loss(
        *,
        phase: str,
        step_idx: int | None,
        total_steps: int | None,
        metrics: dict[str, float | None] | None,
    ) -> None:
        nonlocal first_match_loss_phase, previous_match_value
        if not bool(cfg.debug) or metrics is None:
            return
        match_value = metrics.get("match")
        current_match = None if match_value is None else int(match_value)
        if (
            first_match_loss_phase is None
            and previous_match_value == 1
            and current_match == 0
        ):
            first_match_loss_phase = _phase_label(phase, step_idx, total_steps)
            print(
                f"kldm_dpnpsvd_match_transition graph={graph_idx + 1} "
                f"event=first_loss_of_match phase={first_match_loss_phase}",
                flush=True,
            )
        if current_match is not None:
            previous_match_value = current_match

    def _record_best_phase(
        *,
        phase: str,
        step_idx: int | None,
        total_steps: int | None,
        metrics: dict[str, float | None] | None,
    ) -> None:
        nonlocal best_standardized_phase, best_matcher_phase
        if not bool(cfg.debug) or not bool(cfg.debug_best_phase_metrics) or metrics is None:
            return
        phase_name = _phase_label(phase, step_idx, total_steps)
        std_value = metrics.get("standardized_frac_rmse")
        frac_value = metrics.get("frac_rmse")
        if std_value is not None and math.isfinite(float(std_value)):
            if (
                best_standardized_phase is None
                or float(std_value) < float(best_standardized_phase["standardized_frac_rmse"])
            ):
                best_standardized_phase = {
                    "phase": phase_name,
                    "standardized_frac_rmse": float(std_value),
                    "frac_rmse": None if frac_value is None else float(frac_value),
                }
        match_value = metrics.get("match")
        if match_value is not None and int(match_value) == 1:
            matcher_score = std_value if std_value is not None else frac_value
            if matcher_score is None or not math.isfinite(float(matcher_score)):
                matcher_score = float("inf")
            if (
                best_matcher_phase is None
                or float(matcher_score) < float(best_matcher_phase["score"])
            ):
                best_matcher_phase = {
                    "phase": phase_name,
                    "score": float(matcher_score),
                    "standardized_frac_rmse": None if std_value is None else float(std_value),
                }
    if bool(cfg.debug):
        init_stats = _theta_debug_stats(state=current_state, theta=_state_to_theta(current_state), cfg=cfg)
        print(
            f"kldm_dpnpsvd_catalog graph={graph_idx + 1} sg={requested_sg} templates={len(ranked_templates)} "
            f"selected_idx={int(chosen.template_idx) + 1} selected_ranker={float(chosen.ranker_score):.6f} "
            f"selected_prior_count={int(chosen.template_prior_count)} "
            f"template_prior_mode={ranking_debug.template_prior_mode} "
            f"oracle_surrogate_applied={int(ranking_debug.oracle_surrogate_applied)} "
            f"oracle_surrogate_hit={int(ranking_debug.oracle_surrogate_hit)} "
            f"oracle_surrogate_match_count={int(ranking_debug.oracle_surrogate_match_count)} "
            f"orbit_mismatch={int(chosen.orbit_mismatch)} oracle_source={oracle_source} "
            f"signature={_template_signature_labels(chosen.template)} "
            f"init_objective={float(current_state.objective):.6f} "
            f"init_residual_norm={init_stats.residual_norm:.6f} "
            f"init_min_pair_distance={init_stats.min_pair_distance:.4f} "
            f"init_cell_volume={init_stats.cell_volume:.6f}",
            flush=True,
        )
        oracle_metrics_prev = _log_oracle_step_metrics(
            graph_idx=graph_idx,
            phase="init",
            step_idx=None,
            total_steps=None,
            state=current_state,
            requested_sg=int(requested_sg),
            oracle_target_frac=oracle_target_frac,
            oracle_target_l=oracle_target_l,
            oracle_target_species=oracle_target_species,
            lattice_transform=lattice_transform,
            cfg=cfg,
            device=device,
            dtype=dtype,
            previous_metrics=None,
        )
        _record_best_phase(
            phase="init",
            step_idx=None,
            total_steps=None,
            metrics=oracle_metrics_prev,
        )
        _record_first_match_loss(
            phase="init",
            step_idx=None,
            total_steps=None,
            metrics=oracle_metrics_prev,
        )

    current_pos = pos_prior
    current_l = l_prior.view(1, -1)
    current_h = h_prior
    best_valid_state: _PCSChartState | None = None
    best_valid_energy: float | None = None
    prior_attempted_total = 0
    prior_available_total = 0
    compat_batch_total = 0
    prior_last_reason = "not_run"

    for step_idx, eta in enumerate(schedule, start=1):
        if bool(cfg.debug):
            print(
                f"kldm_dpnpsvd_outer_enter graph={graph_idx + 1} step={step_idx}/{len(schedule)} "
                f"eta={float(eta.item()):.5f}",
                flush=True,
            )
        current_idx, current_state = _pcs_kernel(
            ranked_templates=ranked_templates,
            template_log_probs=template_log_probs,
            current_idx=current_idx,
            current_state=current_state,
            anchor_frac=current_pos,
            anchor_l=current_l,
            anchor_species=current_h,
            requested_sg=int(requested_sg),
            lattice_transform=lattice_transform,
            cfg=cfg,
            eta=float(eta.item()),
            num_steps=int(cfg.pcs_mh_steps),
            hard_reject_distance=float(cfg.min_distance) * float(cfg.outer_hard_reject_distance_ratio),
        )
        pos_half, l_half, h_half, structure_half = _materialize_state(
            state=current_state,
            lattice_transform=lattice_transform,
            device=device,
            dtype=dtype,
        )
        if bool(cfg.debug):
            cell_half = torch.tensor(
                np.asarray(structure_half.lattice.matrix, dtype=float).copy(),
                device=device,
                dtype=dtype,
            )
            out_abc, out_angles = _cell_abc_angles(cell_half)
            print(
                f"kldm_dpnpsvd_materialize graph={graph_idx + 1} step={step_idx}/{len(schedule)} "
                f"representation={current_state.target_representation_name} "
                f"atoms={int(h_half.shape[0])} abc={[round(v, 4) for v in out_abc]} "
                f"angles={[round(v, 4) for v in out_angles]}",
                flush=True,
            )
        validation, min_pair = _validate_projection(
            structure=structure_half,
            requested_sg=int(requested_sg),
            expected_atomic_numbers=h_prior,
            cfg=cfg,
        )
        if bool(cfg.debug):
            print(
                f"kldm_dpnpsvd_step graph={graph_idx + 1} step={step_idx}/{len(schedule)} "
                f"eta={float(eta.item()):.5f} requested_sg_match={int(bool(validation.requested_space_group_match))} "
                f"min_pair_distance={min_pair:.4f}",
                flush=True,
            )
        if bool(validation.requested_space_group_match) and min_pair >= float(cfg.min_distance):
            pcs_energy = float(
                _target_energy(
                    state=current_state,
                    theta=_state_to_theta(current_state),
                    eta=float(eta.item()),
                    cfg=cfg,
                    include_jacobian=False,
                ).detach().item()
            )
            if best_valid_energy is None or pcs_energy < best_valid_energy:
                best_valid_state = replace(current_state)
                best_valid_energy = pcs_energy
        pcs_energy_any = float(
            _target_energy(
                state=current_state,
                theta=_state_to_theta(current_state),
                eta=float(eta.item()),
                cfg=cfg,
                include_jacobian=False,
            ).detach().item()
        )
        if best_available_energy is None or pcs_energy_any < best_available_energy:
            best_available_state = replace(current_state)
            best_available_energy = pcs_energy_any

        if bool(cfg.debug):
            print(
                f"kldm_dpnpsvd_post_pcs graph={graph_idx + 1} step={step_idx}/{len(schedule)} "
                f"ambient_dds_steps={int(cfg.ambient_dds_steps)}",
                flush=True,
            )
            oracle_metrics_prev = _log_oracle_step_metrics(
                graph_idx=graph_idx,
                phase="pcs",
                step_idx=step_idx,
                total_steps=len(schedule),
                state=current_state,
                requested_sg=int(requested_sg),
                oracle_target_frac=oracle_target_frac,
                oracle_target_l=oracle_target_l,
                oracle_target_species=oracle_target_species,
                lattice_transform=lattice_transform,
                cfg=cfg,
                device=device,
                dtype=dtype,
                previous_metrics=oracle_metrics_prev,
            )
            _record_best_phase(
                phase="pcs",
                step_idx=step_idx,
                total_steps=len(schedule),
                metrics=oracle_metrics_prev,
            )
            _record_first_match_loss(
                phase="pcs",
                step_idx=step_idx,
                total_steps=len(schedule),
                metrics=oracle_metrics_prev,
            )

        if int(cfg.ambient_dds_steps) > 0:
            if bool(cfg.debug):
                print(
                    f"kldm_dpnpsvd_pre_ambient_dds graph={graph_idx + 1} step={step_idx}/{len(schedule)}",
                    flush=True,
                )
            current_pos, current_l, current_h, prior_stats = _dds_kldm_ambient_kernel(
                model=model,
                graph_batch=graph_batch,
                pos_half=pos_half,
                l_half=l_half,
                h_half=h_half,
                cfg=cfg,
                eta=float(eta.item()),
            )
            prior_attempted_total += int(prior_stats.get("prior_attempted_steps", 0))
            prior_available_total += int(prior_stats.get("prior_used_steps", 0))
            compat_batch_total += int(prior_stats.get("compat_batch_steps", 0))
            prior_last_reason = str(prior_stats.get("prior_last_reason", "ambient_dds_unknown"))
            if bool(cfg.debug):
                print(
                    f"kldm_dpnpsvd_post_ambient_dds graph={graph_idx + 1} step={step_idx}/{len(schedule)} "
                    f"prior_available={int(bool(prior_stats.get('prior_available', False)))} "
                    f"prior_used_steps={int(prior_stats.get('prior_used_steps', 0))}/{int(prior_stats.get('prior_attempted_steps', 0))} "
                    f"compat_batch_steps={int(prior_stats.get('compat_batch_steps', 0))} "
                    f"reason={prior_last_reason}",
                    flush=True,
                )
                oracle_metrics_prev = _log_oracle_batch_metrics(
                    graph_idx=graph_idx,
                    phase="dds",
                    step_idx=step_idx,
                    total_steps=len(schedule),
                    pred_f=current_pos,
                    pred_l=current_l,
                    pred_a=current_h,
                    requested_sg=int(requested_sg),
                    oracle_target_frac=oracle_target_frac,
                    oracle_target_l=oracle_target_l,
                    oracle_target_species=oracle_target_species,
                    cfg=cfg,
                    lattice_transform=lattice_transform,
                    previous_metrics=oracle_metrics_prev,
                )
                _record_best_phase(
                    phase="dds",
                    step_idx=step_idx,
                    total_steps=len(schedule),
                    metrics=oracle_metrics_prev,
                )
                _record_first_match_loss(
                    phase="dds",
                    step_idx=step_idx,
                    total_steps=len(schedule),
                    metrics=oracle_metrics_prev,
                )
            if bool(cfg.debug):
                print(
                    f"kldm_dpnpsvd_post_dds_materialize graph={graph_idx + 1} step={step_idx}/{len(schedule)}",
                    flush=True,
                )
        else:
            if bool(validation.requested_space_group_match) and min_pair >= float(cfg.min_distance):
                current_pos, current_l, current_h = pos_half, l_half, h_half
                if bool(cfg.debug):
                    print(
                        f"kldm_dpnpsvd_skip_ambient_dds graph={graph_idx + 1} step={step_idx}/{len(schedule)} "
                        "mode=accept_pcs_anchor",
                        flush=True,
                    )
            else:
                if bool(cfg.debug):
                    print(
                        f"kldm_dpnpsvd_skip_ambient_dds graph={graph_idx + 1} step={step_idx}/{len(schedule)} "
                        "mode=retain_previous_anchor_invalid_pcs",
                        flush=True,
                    )

    final_eta = float(schedule[-1].item())
    if int(cfg.debug_fixed_template_multistart_restarts) > 0 and bool(cfg.debug):
        diag_cfg = replace(cfg, template_move_probability=0.0)
        diag_eta = (
            float(cfg.debug_fixed_template_multistart_eta)
            if float(cfg.debug_fixed_template_multistart_eta) > 0.0
            else final_eta
        )
        diag_steps = (
            int(cfg.debug_fixed_template_multistart_steps)
            if int(cfg.debug_fixed_template_multistart_steps) > 0
            else max(1, int(cfg.final_pcs_mh_steps))
        )
        print(
            f"kldm_dpnpsvd_fixed_template_multistart graph={graph_idx + 1} "
            f"restarts={int(cfg.debug_fixed_template_multistart_restarts)} "
            f"eta={diag_eta:.5f} steps={diag_steps}",
            flush=True,
        )
        diag_best_energy: float | None = None
        diag_best_restart: int | None = None
        diag_center = _proposal_center_theta(current_state)
        for restart_idx in range(max(1, int(cfg.debug_fixed_template_multistart_restarts))):
            if restart_idx == 0:
                diag_state_start = replace(current_state)
            else:
                diag_theta = _sample_theta_proposal(
                    center=diag_center,
                    free_dim=int(current_state.template.total_free_dims),
                    cfg=diag_cfg,
                )
                diag_state_start = _state_with_theta(replace(current_state), diag_theta)
            diag_state = _run_fixed_template_pcs(
                ranked_item=ranked_templates[current_idx],
                current_state=diag_state_start,
                anchor_frac=current_pos,
                anchor_l=current_l,
                anchor_species=current_h,
                requested_sg=int(requested_sg),
                lattice_transform=lattice_transform,
                cfg=diag_cfg,
                eta=diag_eta,
                num_steps=diag_steps,
                hard_reject_close_contacts=True,
            )
            diag_pos, diag_l, diag_h, diag_structure = _materialize_state(
                state=diag_state,
                lattice_transform=lattice_transform,
                device=device,
                dtype=dtype,
            )
            diag_validation, diag_min_pair = _validate_projection(
                structure=diag_structure,
                requested_sg=int(requested_sg),
                expected_atomic_numbers=h_prior,
                cfg=cfg,
            )
            diag_energy = float(
                _target_energy(
                    state=diag_state,
                    theta=_state_to_theta(diag_state),
                    eta=diag_eta,
                    cfg=diag_cfg,
                    include_jacobian=False,
                ).detach().item()
            )
            print(
                f"kldm_dpnpsvd_fixed_template_multistart_restart graph={graph_idx + 1} "
                f"restart={restart_idx + 1}/{int(cfg.debug_fixed_template_multistart_restarts)} "
                f"energy={diag_energy:.6f} "
                f"requested_sg_match={int(bool(diag_validation.requested_space_group_match))} "
                f"min_pair_distance={diag_min_pair:.4f}",
                flush=True,
            )
            diag_metrics = _log_oracle_batch_metrics(
                graph_idx=graph_idx,
                phase="diag_fixed_template",
                step_idx=restart_idx + 1,
                total_steps=int(cfg.debug_fixed_template_multistart_restarts),
                pred_f=diag_pos,
                pred_l=diag_l,
                pred_a=diag_h,
                requested_sg=int(requested_sg),
                oracle_target_frac=oracle_target_frac,
                oracle_target_l=oracle_target_l,
                oracle_target_species=oracle_target_species,
                cfg=cfg,
                lattice_transform=lattice_transform,
                previous_metrics=None,
            )
            _record_best_phase(
                phase="diag_fixed_template",
                step_idx=restart_idx + 1,
                total_steps=int(cfg.debug_fixed_template_multistart_restarts),
                metrics=diag_metrics,
            )
            if (
                bool(diag_validation.requested_space_group_match)
                and diag_min_pair >= float(cfg.min_distance)
                and (diag_best_energy is None or diag_energy < diag_best_energy)
            ):
                diag_best_energy = diag_energy
                diag_best_restart = restart_idx + 1
        print(
            f"kldm_dpnpsvd_fixed_template_multistart_best graph={graph_idx + 1} "
            f"best_restart={diag_best_restart if diag_best_restart is not None else 'na'} "
            f"best_energy={diag_best_energy if diag_best_energy is not None else float('nan'):.6f}",
            flush=True,
        )

    if bool(cfg.final_fixed_template_refine):
        final_cfg = replace(cfg, template_move_probability=0.0)
        if bool(cfg.debug):
            print(
                f"kldm_dpnpsvd_final_refine graph={graph_idx + 1} "
                f"mode=fixed_template eta={final_eta:.5f} steps={max(1, int(cfg.final_pcs_mh_steps))}",
                flush=True,
            )
        current_state = _run_fixed_template_pcs(
            ranked_item=ranked_templates[current_idx],
            current_state=current_state,
            anchor_frac=current_pos,
            anchor_l=current_l,
            anchor_species=current_h,
            requested_sg=int(requested_sg),
            lattice_transform=lattice_transform,
            cfg=final_cfg,
            eta=final_eta,
            num_steps=max(1, int(cfg.final_pcs_mh_steps)),
            hard_reject_close_contacts=True,
        )
    elif bool(cfg.debug):
        print(
            f"kldm_dpnpsvd_final_refine graph={graph_idx + 1} mode=disabled",
            flush=True,
        )
    pos_out, l_out, h_out, structure_out = _materialize_state(
        state=current_state,
        lattice_transform=lattice_transform,
        device=device,
        dtype=dtype,
    )
    validation_out, min_pair_out = _validate_projection(
        structure=structure_out,
        requested_sg=int(requested_sg),
        expected_atomic_numbers=h_prior,
        cfg=cfg,
    )
    if bool(cfg.debug):
        print(
            f"kldm_dpnpsvd_final graph={graph_idx + 1} requested_sg_match={int(bool(validation_out.requested_space_group_match))} "
            f"min_pair_distance={min_pair_out:.4f} "
            f"prior_available={int(prior_available_total > 0)} "
            f"prior_used_steps={prior_available_total}/{prior_attempted_total} "
            f"compat_batch_steps={compat_batch_total} "
            f"prior_last_reason={prior_last_reason}",
            flush=True,
        )
        oracle_metrics_prev = _log_oracle_step_metrics(
            graph_idx=graph_idx,
            phase="final",
            step_idx=None,
            total_steps=None,
            state=current_state,
            requested_sg=int(requested_sg),
            oracle_target_frac=oracle_target_frac,
            oracle_target_l=oracle_target_l,
            oracle_target_species=oracle_target_species,
            lattice_transform=lattice_transform,
            cfg=cfg,
            device=device,
            dtype=dtype,
            previous_metrics=oracle_metrics_prev,
        )
        _record_best_phase(
            phase="final",
            step_idx=None,
            total_steps=None,
            metrics=oracle_metrics_prev,
        )
        _record_first_match_loss(
            phase="final",
            step_idx=None,
            total_steps=None,
            metrics=oracle_metrics_prev,
        )
        if bool(cfg.debug_best_phase_metrics):
            best_std_phase = "na" if best_standardized_phase is None else str(best_standardized_phase["phase"])
            best_std_value = (
                "na"
                if best_standardized_phase is None
                else f"{float(best_standardized_phase['standardized_frac_rmse']):.6f}"
            )
            best_match_phase = "na" if best_matcher_phase is None else str(best_matcher_phase["phase"])
            best_match_std = (
                "na"
                if best_matcher_phase is None or best_matcher_phase["standardized_frac_rmse"] is None
                else f"{float(best_matcher_phase['standardized_frac_rmse']):.6f}"
            )
            print(
                f"kldm_dpnpsvd_best_oracle graph={graph_idx + 1} "
                f"best_phase_by_standardized_frac_rmse={best_std_phase} "
                f"best_standardized_frac_rmse={best_std_value} "
                f"best_phase_by_matcher={best_match_phase} "
                f"best_matcher_standardized_frac_rmse={best_match_std} "
                f"first_loss_of_match={first_match_loss_phase or 'na'}",
                flush=True,
            )
    final_requested_sg_valid = bool(validation_out.requested_space_group_match) and min_pair_out >= float(cfg.min_distance)
    if final_requested_sg_valid:
        pos_final, l_final, h_final = _materialize_state_for_batch_output(
            state=current_state,
            lattice_transform=lattice_transform,
            device=device,
            dtype=dtype,
        )
        return pos_final, l_final, h_final, {
            "prior_available": prior_available_total > 0,
            "prior_attempted_steps": prior_attempted_total,
            "prior_used_steps": prior_available_total,
            "compat_batch_steps": compat_batch_total,
            "prior_last_reason": prior_last_reason,
        }
    if bool(cfg.debug):
        print(
            f"kldm_dpnpsvd_failure graph={graph_idx + 1} reason=final_pcs_invalid "
            f"source=report_failure_not_best_state",
            flush=True,
        )
    pos_invalid, l_invalid, h_invalid = _invalid_sample_like(pos_ref=pos_prior, l_ref=l_prior, h_ref=h_prior)
    return pos_invalid, l_invalid, h_invalid, {
        "prior_available": prior_available_total > 0,
        "prior_attempted_steps": prior_attempted_total,
        "prior_used_steps": prior_available_total,
        "compat_batch_steps": compat_batch_total,
        "prior_last_reason": prior_last_reason,
    }


def sample_kldm_dpnp_svd(
    *,
    model: Any,
    n_steps: int,
    batch: Any,
    lattice_transform: Any,
    t_start: float,
    t_final: float,
    config: DPnPSVDConfig,
    template_prior: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # The PCS path in this file is the main theorem-facing Wyckoff implementation.
    # The DDS path is an ambient KLDM reverse chain initialized from the PCS
    # materialization rather than a chart-local repair step.
    if not hasattr(batch, "space_group"):
        raise ValueError("DPnPSVD requires batch.space_group.")

    config = _effective_faithful_cfg(config)

    started = time.perf_counter()
    if bool(config.debug):
        print(
            "kldm_dpnpsvd_mode "
            f"faithful={int(bool(config.faithful_dpnp))} "
            f"pcs_templates={'union' if float(config.template_move_probability) > 0.0 else 'fixed'} "
            f"template_prior_mode={str(config.template_prior_mode)} "
            f"oracle_orbit_filter={int(bool(config.oracle_template_orbit_filter))} "
            f"ambient_dds={'off' if int(config.ambient_dds_steps) <= 0 else 'kldm_reverse'} "
            f"sg_conditioned_dds={int(bool(config.sg_conditioned_dds))} "
            f"sg_guidance_scale={float(config.sg_guidance_scale):.3f}",
            flush=True,
        )
        if bool(config.faithful_dpnp):
            print(
                "kldm_dpnpsvd_faithful_overrides "
                "pair_distance_weight=0.0 final_fixed_template_refine=0 "
                "oracle_template_orbit_rerank=0 oracle_template_orbit_filter=0",
                flush=True,
            )
    pos_prior, v_prior, l_prior, h_prior = model.sample_CSP_algorithm3(
        n_steps=n_steps,
        batch=batch,
        t_start=t_start,
        t_final=t_final,
        space_group=(
            torch.as_tensor(batch.space_group, device=batch.pos.device, dtype=torch.long).reshape(-1)
            if bool(config.sg_conditioned_dds) and hasattr(batch, "space_group")
            else None
        ),
        sg_guidance_scale=float(config.sg_guidance_scale),
    )
    print(
        f"kldm_dpnpsvd_progress phase=prior graphs={int(batch.num_graphs)} "
        f"elapsed_s={time.perf_counter() - started:.1f}",
        flush=True,
    )
    del v_prior

    ranker = _maybe_load_template_ranker(
        path=config.template_ranker_path,
        device=pos_prior.device,
    )
    template_cache = _maybe_load_disk_template_cache(config.template_cache_path)
    schedule = _eta_schedule(config, device=pos_prior.device, dtype=pos_prior.dtype)
    requested = torch.as_tensor(batch.space_group, device=h_prior.device, dtype=torch.long).reshape(-1)
    ptr = batch.ptr.tolist()

    pos_blocks: list[torch.Tensor] = []
    l_blocks: list[torch.Tensor] = []
    h_blocks: list[torch.Tensor] = []
    batch_index_blocks: list[torch.Tensor] = []
    prior_available_graphs = 0
    prior_attempted_steps_total = 0
    prior_used_steps_total = 0
    compat_batch_steps_total = 0
    for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
        graph_batch = _single_graph_batch(batch, graph_idx)
        graph_pos, graph_l, graph_h, graph_meta = _sample_graph(
            model=model,
            graph_idx=graph_idx,
            graph_batch=graph_batch,
            requested_sg=int(requested[graph_idx].item()),
            pos_prior=pos_prior[start:end],
            l_prior=l_prior[graph_idx],
            h_prior=h_prior[start:end],
            oracle_target_frac=batch.pos[start:end],
            oracle_target_l=batch.l[graph_idx],
            oracle_target_species=batch.atomic_numbers[start:end],
            lattice_transform=lattice_transform,
            cfg=config,
            schedule=schedule,
            ranker=ranker,
            template_cache=template_cache,
            template_prior=template_prior,
            prior_steps=int(n_steps),
        )
        prior_available_graphs += int(bool(graph_meta.get("prior_available", False)))
        prior_attempted_steps_total += int(graph_meta.get("prior_attempted_steps", 0))
        prior_used_steps_total += int(graph_meta.get("prior_used_steps", 0))
        compat_batch_steps_total += int(graph_meta.get("compat_batch_steps", 0))
        pos_blocks.append(graph_pos.to(device=pos_prior.device, dtype=pos_prior.dtype))
        l_blocks.append(graph_l.to(device=l_prior.device, dtype=l_prior.dtype))
        h_blocks.append(graph_h.to(device=h_prior.device, dtype=h_prior.dtype))
        batch_index_blocks.append(
            torch.full(
                (int(graph_pos.shape[0]),),
                int(graph_idx),
                device=batch.batch.device,
                dtype=batch.batch.dtype,
            )
        )

    pos_out = torch.cat(pos_blocks, dim=0)
    l_out = torch.cat(l_blocks, dim=0)
    h_out = torch.cat(h_blocks, dim=0)
    batch_out = torch.cat(batch_index_blocks, dim=0)
    v_out = model.tdm.sample_velocity_noise(pos_out, index=batch_out)
    print(
        f"kldm_dpnpsvd_progress phase=done graphs={int(batch.num_graphs)} "
        f"prior_available_graphs={prior_available_graphs}/{int(batch.num_graphs)} "
        f"prior_used_steps={prior_used_steps_total}/{prior_attempted_steps_total} "
        f"compat_batch_steps={compat_batch_steps_total} "
        f"elapsed_s={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return pos_out, v_out, l_out, h_out
