from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from kldmPlus.symmetry.wyckoff_templates import WyckoffTemplate


def wyckoff_letter_to_index(label: str, *, max_letter: int = 64) -> int:
    letters = "".join(ch for ch in str(label).lower() if ch.isalpha())
    if not letters:
        return 0
    value = 0
    for char in letters:
        offset = ord(char) - ord("a") + 1
        if 1 <= offset <= 26:
            value = value * 26 + offset
    return max(0, min(int(max_letter) - 1, value - 1))


class WyckoffTemplateRanker(torch.nn.Module):
    """Small permutation-invariant ranker over species-labeled Wyckoff sites."""

    def __init__(
        self,
        *,
        num_elements: int = 119,
        max_sg: int = 231,
        max_mult: int = 256,
        max_letter: int = 64,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.num_elements = int(num_elements)
        self.max_sg = int(max_sg)
        self.max_mult = int(max_mult)
        self.max_letter = int(max_letter)
        self.hidden = int(hidden)

        self.element_emb = torch.nn.Embedding(self.num_elements, self.hidden)
        self.sg_emb = torch.nn.Embedding(self.max_sg + 1, self.hidden)
        self.mult_emb = torch.nn.Embedding(self.max_mult + 1, self.hidden)
        self.letter_emb = torch.nn.Embedding(self.max_letter, self.hidden)
        self.dof_emb = torch.nn.Embedding(4, self.hidden)

        self.site_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden * 4, self.hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden, self.hidden),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(self.hidden * 2, self.hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden, 1),
        )

    def forward(
        self,
        *,
        sg: torch.Tensor,
        site_z: torch.Tensor,
        site_mult: torch.Tensor,
        site_letter: torch.Tensor,
        site_dof: torch.Tensor,
        site_mask: torch.Tensor,
    ) -> torch.Tensor:
        site_z = site_z.clamp(0, self.num_elements - 1)
        site_mult = site_mult.clamp(0, self.max_mult)
        site_letter = site_letter.clamp(0, self.max_letter - 1)
        site_dof = site_dof.clamp(0, 3)
        sg = sg.clamp(0, self.max_sg)

        element_emb = self.element_emb(site_z)
        mult_emb = self.mult_emb(site_mult)
        letter_emb = self.letter_emb(site_letter)
        dof_emb = self.dof_emb(site_dof)
        site_feat = self.site_mlp(torch.cat([element_emb, mult_emb, letter_emb, dof_emb], dim=-1))

        mask = site_mask.unsqueeze(-1).to(dtype=site_feat.dtype)
        pooled = (site_feat * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        sg_feat = self.sg_emb(sg)
        return self.out(torch.cat([pooled, sg_feat], dim=-1)).squeeze(-1)


def templates_to_ranker_batch(
    *,
    templates: list[WyckoffTemplate],
    requested_sg: int,
    device: torch.device | str,
    max_letter: int = 64,
    max_mult: int = 256,
) -> dict[str, torch.Tensor]:
    if not templates:
        raise ValueError("Cannot tensorize an empty template list.")
    max_sites = max(1, max(len(template.site_templates) for template in templates))
    batch_size = len(templates)
    site_z = torch.zeros((batch_size, max_sites), device=device, dtype=torch.long)
    site_mult = torch.zeros((batch_size, max_sites), device=device, dtype=torch.long)
    site_letter = torch.zeros((batch_size, max_sites), device=device, dtype=torch.long)
    site_dof = torch.zeros((batch_size, max_sites), device=device, dtype=torch.long)
    site_mask = torch.zeros((batch_size, max_sites), device=device, dtype=torch.bool)

    for row, template in enumerate(templates):
        sites = sorted(
            template.site_templates,
            key=lambda site: (int(site.atomic_number), str(site.label), int(site.multiplicity), int(site.dof)),
        )
        for col, site in enumerate(sites):
            site_z[row, col] = int(site.atomic_number)
            site_mult[row, col] = min(int(site.multiplicity), int(max_mult))
            site_letter[row, col] = wyckoff_letter_to_index(str(site.label), max_letter=max_letter)
            site_dof[row, col] = min(max(int(site.dof), 0), 3)
            site_mask[row, col] = True

    sg = torch.full((batch_size,), int(requested_sg), device=device, dtype=torch.long)
    return {
        "sg": sg,
        "site_z": site_z,
        "site_mult": site_mult,
        "site_letter": site_letter,
        "site_dof": site_dof,
        "site_mask": site_mask,
    }


@torch.no_grad()
def score_templates(
    *,
    ranker: torch.nn.Module | None,
    templates: list[WyckoffTemplate],
    requested_sg: int,
    device: torch.device | str,
) -> list[float]:
    if ranker is None or not templates:
        return [0.0 for _ in templates]
    max_letter = int(getattr(ranker, "max_letter", 64))
    max_mult = int(getattr(ranker, "max_mult", 256))
    payload = templates_to_ranker_batch(
        templates=templates,
        requested_sg=int(requested_sg),
        device=device,
        max_letter=max_letter,
        max_mult=max_mult,
    )
    ranker.eval()
    scores = ranker(**payload)
    return [float(value) for value in scores.detach().cpu().reshape(-1).tolist()]


def save_template_ranker(
    *,
    path: str | Path,
    model: WyckoffTemplateRanker,
    metadata: dict[str, Any] | None = None,
) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_kwargs": {
                "num_elements": int(model.num_elements),
                "max_sg": int(model.max_sg),
                "max_mult": int(model.max_mult),
                "max_letter": int(model.max_letter),
                "hidden": int(model.hidden),
            },
            "metadata": dict(metadata or {}),
        },
        target,
    )


def load_template_ranker(
    path: str | Path,
    *,
    device: torch.device | str,
) -> WyckoffTemplateRanker:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Template ranker checkpoint does not exist: {source}")
    try:
        payload = torch.load(source, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(source, map_location=device)
    model_kwargs = dict(payload.get("model_kwargs", {}))
    model = WyckoffTemplateRanker(**model_kwargs).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
