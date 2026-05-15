from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kldmPlus.data.csp import validate_lattice_configuration
from kldmPlus.data.transform import (
    MatterGenContinuousIntervalLattice,
    cell_lengths_and_angles,
    ensure_mattergen_lattice_cache,
    mattergen_symmetric_cell,
)
from kldmPlus.diffusionModels.continuous import ContinuousMattergenVPDiffusion
from kldmPlus.kldm import ModelKLDM, PreparedTrainingBatch
from kldmPlus.sample_evaluation import sample_evaluation as sample_eval
from kldmPlus.utils.time import BatchTimes


class _SampleStub:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def replace(self, **kwargs):
        payload = dict(self.__dict__)
        payload.update(kwargs)
        return _SampleStub(**payload)


class _ScoreStub(nn.Module):
    def __init__(self, *, out_v: torch.Tensor, out_l: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_out_v", out_v)
        self.register_buffer("_out_l", out_l)

    def forward(self, **_kwargs) -> dict[str, torch.Tensor]:
        return {
            "v": self._out_v.clone(),
            "l": self._out_l.clone(),
        }


def test_mattergen_symmetric_cell_removes_only_rotation_in_row_convention() -> None:
    """The row-convention polar factor should differ from the input only by rotation."""
    cell = torch.tensor(
        [
            [4.1, 0.3, 0.1],
            [0.8, 5.2, 0.4],
            [0.2, 1.1, 6.0],
        ],
        dtype=torch.float64,
    )

    cell_sym = mattergen_symmetric_cell(cell)

    assert torch.allclose(cell_sym, cell_sym.transpose(-1, -2), atol=1e-10, rtol=1e-10)

    # For row-vector cells we expect: cell ≈ cell_sym @ R, with R orthogonal.
    rotation = torch.linalg.solve(cell_sym, cell)
    identity = torch.eye(3, dtype=rotation.dtype)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-8, rtol=1e-8)
    assert torch.allclose(rotation @ rotation.transpose(-1, -2), identity, atol=1e-8, rtol=1e-8)


def test_mattergen_vp_requires_mattergen_representation() -> None:
    with pytest.raises(ValueError, match="requires dataset\\.lattice_representation='mattergen'"):
        validate_lattice_configuration(
            lattice_representation="kldm",
            lattice_parameterization="eps",
            lattice_diffusion_type="mattergenVP",
        )


def test_representation_only_mattergen_ablation_is_allowed() -> None:
    validate_lattice_configuration(
        lattice_representation="mattergen",
        lattice_parameterization="eps",
        lattice_diffusion_type="VP",
    )


def test_mattergen_density_and_variance_adapter_matches_official_limit_variance() -> None:
    average_density = torch.tensor(0.05771451654022283, dtype=torch.float64)
    c = 1.0 / average_density
    nu = torch.tensor(0.25, dtype=torch.float64).pow(1.5)

    diffusion = ContinuousMattergenVPDiffusion(
        c=float(c.item()),
        nu=float(nu.item()),
        parameterization="eps",
    )

    ref = torch.zeros(2, 6, dtype=torch.float64)
    num_atoms = torch.tensor([1.0, 8.0], dtype=torch.float64)
    mu_n, sigma_n = diffusion.mu_sigma_n(num_atoms=num_atoms, ref=ref)

    expected_mu = torch.pow(num_atoms / average_density, 1.0 / 3.0)
    expected_var = torch.pow(num_atoms, 2.0 / 3.0) * 0.25

    assert torch.allclose(mu_n, expected_mu)
    assert torch.allclose(sigma_n.pow(2), expected_var)


def test_mattergen_ancestral_lattice_step_recovers_clean_cell_at_zero_time() -> None:
    average_density = torch.tensor(0.05771451654022283, dtype=torch.float64)
    diffusion = ContinuousMattergenVPDiffusion(
        c=float(average_density.reciprocal().item()),
        nu=float(torch.tensor(0.25, dtype=torch.float64).pow(1.5).item()),
        parameterization="eps",
    )

    x0 = torch.tensor(
        [
            [3.0, 4.0, 5.0, 0.1, 0.2, 0.3],
            [5.5, 6.0, 6.5, -0.2, 0.1, 0.4],
        ],
        dtype=torch.float64,
    )
    noise = torch.tensor(
        [
            [0.5, -0.1, 0.2, 1.1, -0.7, 0.3],
            [-0.4, 0.8, -0.6, 0.2, 0.5, -0.9],
        ],
        dtype=torch.float64,
    )
    t = torch.tensor([0.35, 0.70], dtype=torch.float64)
    num_atoms = torch.tensor([4.0, 8.0], dtype=torch.float64)

    x_t, target = diffusion.forward_sample(
        t=t,
        x0=x0,
        noise=noise,
        num_atoms=num_atoms,
    )
    recovered = diffusion.reverse_step_ancestral(
        t=t,
        x_t=x_t,
        pred=target,
        dt=1.0,
        num_atoms=num_atoms,
        noise=torch.zeros_like(x_t),
    )

    assert torch.allclose(recovered, x0, atol=1e-5, rtol=1e-5)


def test_mattergen_transform_round_trip_preserves_metric_volume_and_pairwise_distances() -> None:
    cell = torch.tensor(
        [
            [4.1, 0.3, 0.1],
            [0.8, 5.2, 0.4],
            [0.2, 1.1, 6.0],
        ],
        dtype=torch.float64,
    )
    frac = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.50, 0.75],
            [0.60, 0.10, 0.40],
            [0.90, 0.80, 0.20],
        ],
        dtype=torch.float64,
    )

    transform = MatterGenContinuousIntervalLattice()
    sample = _SampleStub(cell=cell.unsqueeze(0), pos=frac)
    encoded = transform(sample)
    decoded = transform.invert_to_matrix(encoded.l).squeeze(0)

    orig_lengths, orig_angles = cell_lengths_and_angles(cell)
    decoded_lengths, decoded_angles = cell_lengths_and_angles(decoded)
    assert torch.allclose(decoded, decoded.transpose(-1, -2), atol=1e-10, rtol=1e-10)
    assert torch.allclose(decoded_lengths, orig_lengths, atol=1e-8, rtol=1e-8)
    assert torch.allclose(decoded_angles, orig_angles, atol=1e-8, rtol=1e-8)
    assert torch.allclose(decoded.det().abs(), cell.det().abs(), atol=1e-8, rtol=1e-8)

    orig_cart = torch.einsum("bi,ij->bj", frac, cell)
    decoded_cart = torch.einsum("bi,ij->bj", frac, decoded)
    assert torch.allclose(
        torch.cdist(orig_cart, orig_cart),
        torch.cdist(decoded_cart, decoded_cart),
        atol=1e-8,
        rtol=1e-8,
    )


def test_mattergen_cache_respects_limit_var_scaling_constant_override(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed" / "train"
    processed_dir.mkdir(parents=True)
    cells = np.asarray(
        [
            [[4.0, 0.2, 0.1], [0.3, 5.0, 0.2], [0.1, 0.4, 6.0]],
            [[3.8, 0.1, 0.0], [0.4, 4.7, 0.3], [0.2, 0.5, 5.9]],
        ],
        dtype=np.float64,
    )
    num_atoms = np.asarray([4, 7], dtype=np.int64)
    np.save(processed_dir / "cell.npy", cells)
    np.save(processed_dir / "num_atoms.npy", num_atoms)

    cache_file = tmp_path / "train_mattergen_lattice_stats.json"
    ensure_mattergen_lattice_cache(
        cache_file=cache_file,
        processed_dir=processed_dir,
        limit_var_scaling_constant=0.5,
    )
    payload = json.loads(cache_file.read_text())
    assert payload["limit_var_scaling_constant"] == pytest.approx(0.5)
    assert payload["nu"] == pytest.approx(0.5 ** 1.5)

    ensure_mattergen_lattice_cache(
        cache_file=cache_file,
        processed_dir=processed_dir,
        limit_var_scaling_constant=0.125,
    )
    payload = json.loads(cache_file.read_text())
    assert payload["limit_var_scaling_constant"] == pytest.approx(0.125)
    assert payload["nu"] == pytest.approx(0.125 ** 1.5)

    transform = MatterGenContinuousIntervalLattice(
        cache_file=cache_file,
        limit_var_scaling_constant=0.25,
    )
    c, nu = transform.stats()
    assert c == pytest.approx(payload["c"])
    assert nu == pytest.approx(0.25 ** 1.5)


def test_mattergen_loss_matches_official_weighting_and_graph_reduction() -> None:
    model = ModelKLDM(
        device=torch.device("cpu"),
        lattice_representation="mattergen",
        mattergen_pos_loss_weight=0.1,
        mattergen_cell_loss_weight=1.0,
        mattergen_pos_loss_reduce="sum",
        score_network_kwargs={
            "hidden_dim": 8,
            "time_dim": 8,
            "num_layers": 1,
            "h_dim": 10,
            "num_freqs": 4,
            "ln": False,
            "smooth": False,
            "pred_h": False,
            "pred_l": True,
            "pred_v": True,
            "zero_cog": True,
        },
    )
    model.score_network = _ScoreStub(
        out_v=torch.zeros(3, 3, dtype=torch.float32),
        out_l=torch.zeros(2, 6, dtype=torch.float32),
    )

    prepared = PreparedTrainingBatch(
        times=BatchTimes(
            graph=torch.zeros(2, 1, dtype=torch.float32),
            lattice=torch.zeros(2, dtype=torch.float32),
            nodes=torch.zeros(3, dtype=torch.float32),
        ),
        v_t=torch.zeros(3, 3, dtype=torch.float32),
        f_t=torch.zeros(3, 3, dtype=torch.float32),
        l_t=torch.zeros(2, 6, dtype=torch.float32),
        target_v=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        target_l=torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
        ),
        atomic_numbers=torch.ones(3, dtype=torch.long),
        node_index=torch.tensor([0, 0, 1], dtype=torch.long),
        edge_node_index=torch.zeros(2, 1, dtype=torch.long),
        num_graphs=2,
        lattice_representation="mattergen",
    )

    loss, metrics = model.loss_from_prepared(prepared)

    expected_loss_v_graph = torch.tensor([10.0 / 3.0, 4.0 / 3.0], dtype=torch.float32)
    expected_loss_l_graph = torch.tensor([1.0, 4.0], dtype=torch.float32)
    expected_loss_v_weighted_graph = 0.1 * expected_loss_v_graph
    expected_loss_l_weighted_graph = expected_loss_l_graph
    expected_loss_graph = expected_loss_v_weighted_graph + expected_loss_l_weighted_graph

    assert torch.allclose(metrics["loss_v_graph"], expected_loss_v_graph, atol=1e-6, rtol=1e-6)
    assert torch.allclose(metrics["loss_l_graph"], expected_loss_l_graph, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        metrics["loss_v_weighted_graph"],
        expected_loss_v_weighted_graph,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        metrics["loss_l_weighted_graph"],
        expected_loss_l_weighted_graph,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(metrics["loss_graph"], expected_loss_graph, atol=1e-6, rtol=1e-6)
    assert metrics["loss_v_weighted"].item() == pytest.approx(expected_loss_v_weighted_graph.mean().item())
    assert metrics["loss_l_weighted"].item() == pytest.approx(expected_loss_l_weighted_graph.mean().item())
    assert loss.item() == pytest.approx(expected_loss_graph.mean().item())


def test_build_structure_from_sample_decodes_mattergen_lattice() -> None:
    if sample_eval.Lattice is None or sample_eval.Structure is None:
        pytest.skip("pymatgen-backed structure reconstruction is unavailable in this environment.")

    cell = torch.tensor(
        [
            [4.1, 0.3, 0.1],
            [0.8, 5.2, 0.4],
            [0.2, 1.1, 6.0],
        ],
        dtype=torch.float64,
    )
    frac = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.50, 0.75],
        ],
        dtype=torch.float64,
    )

    transform = MatterGenContinuousIntervalLattice()
    encoded = transform(_SampleStub(cell=cell.unsqueeze(0)))
    structure = sample_eval.build_structure_from_sample(
        f=frac,
        l=encoded.l.squeeze(0),
        a=torch.tensor([14, 8]),
        lattice_transform=transform,
    )

    lengths = torch.tensor(structure.lattice.lengths, dtype=torch.float64)
    angles = torch.tensor(structure.lattice.angles, dtype=torch.float64)
    expected_lengths, expected_angles = cell_lengths_and_angles(cell)
    assert torch.allclose(lengths, expected_lengths, atol=1e-8, rtol=1e-8)
    assert torch.allclose(angles, torch.rad2deg(expected_angles), atol=1e-8, rtol=1e-8)
