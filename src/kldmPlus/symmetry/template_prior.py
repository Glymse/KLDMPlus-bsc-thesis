from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any
import warnings

import torch

from kldmPlus.symmetry.pyxtal_backend import build_pyxtal_wyckoff_result
from kldmPlus.symmetry.wyckoff_templates import requested_composition_key


TemplatePriorKey = tuple[int, tuple[int, ...], tuple[int, ...]]
TemplatePrior = dict[TemplatePriorKey, dict[tuple[tuple[int, str], ...], int]]


def _anonymous_count_key(key: TemplatePriorKey) -> TemplatePriorKey:
    space_group_number, _species_order, species_counts = key
    return int(space_group_number), tuple([-1] * len(species_counts)), tuple(sorted(int(v) for v in species_counts))


def _anonymous_count_signature(
    *,
    key: TemplatePriorKey,
    signature: tuple[tuple[int, str], ...],
) -> tuple[tuple[int, str], ...]:
    _space_group_number, species_order, species_counts = key
    species_to_count = {
        int(species): int(count)
        for species, count in zip(species_order, species_counts)
    }
    return tuple(
        sorted(
            (
                int(species_to_count.get(int(atomic_number), -1)),
                str(label),
            )
            for atomic_number, label in signature
        )
    )


def build_dataset_template_prior(
    *,
    dataset,
    lattice_transform,
    max_samples: int | None = None,
    allowed_keys: set[TemplatePriorKey] | None = None,
    pyxtal_symprec: float = 1e-2,
    pyxtal_tol: float = 1e-2,
) -> TemplatePrior:
    from kldmPlus.sample_evaluation.sample_evaluation import build_structure_from_sample

    counters: dict[TemplatePriorKey, Counter[tuple[tuple[int, str], ...]]] = defaultdict(Counter)
    limit = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    allowed_exact = set(allowed_keys or [])
    allowed_anonymous = {_anonymous_count_key(key) for key in allowed_exact}

    for sample_idx in range(limit):
        sample = dataset[sample_idx]
        try:
            key = requested_composition_key(
                space_group_number=int(torch.as_tensor(sample.space_group).reshape(-1)[0].item()),
                atomic_numbers=sample.atomic_numbers,
            )
            if allowed_exact and key not in allowed_exact and _anonymous_count_key(key) not in allowed_anonymous:
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"No Pauling electronegativity.*",
                    category=UserWarning,
                )
                structure = build_structure_from_sample(
                    f=sample.pos,
                    l=sample.l,
                    a=sample.atomic_numbers,
                    lattice_transform=lattice_transform,
                )
                result = build_pyxtal_wyckoff_result(
                    structure,
                    symprec=pyxtal_symprec,
                    pyxtal_tol=pyxtal_tol,
                )
            signature = tuple(
                sorted(
                    [
                        (int(z), str(label))
                        for z, label in zip(result.anchor_atomic_numbers.tolist(), result.site_labels)
                    ],
                    key=lambda item: (item[0], item[1]),
                )
            )
            counters[key][signature] += 1
            anonymous_key = _anonymous_count_key(key)
            anonymous_signature = _anonymous_count_signature(key=key, signature=signature)
            counters[anonymous_key][anonymous_signature] += 1
        except Exception:
            continue

    return {
        key: dict(counter)
        for key, counter in counters.items()
    }


def template_prior_score(
    *,
    prior: TemplatePrior | None,
    key: TemplatePriorKey,
    signature: tuple[tuple[int, str], ...],
) -> int:
    if not prior:
        return 0
    exact_score = int(prior.get(key, {}).get(signature, 0))
    if exact_score > 0:
        return exact_score
    anonymous_key = _anonymous_count_key(key)
    anonymous_signature = _anonymous_count_signature(key=key, signature=signature)
    return int(prior.get(anonymous_key, {}).get(anonymous_signature, 0))
