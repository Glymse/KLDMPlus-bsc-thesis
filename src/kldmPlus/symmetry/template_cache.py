from __future__ import annotations

from collections import Counter
from pathlib import Path
import pickle
from typing import Any

import torch

from kldmPlus.symmetry.wyckoff_templates import (
    WyckoffTemplate,
    composition_to_species_counts,
)


TemplateCacheKey = tuple[int, tuple[int, ...], tuple[int, ...]]
TemplateSignature = tuple[tuple[int, str], ...]


def template_cache_key(
    *,
    space_group_number: int,
    atomic_numbers: list[int] | torch.Tensor,
) -> TemplateCacheKey:
    species_order, species_counts = composition_to_species_counts(atomic_numbers)
    return int(space_group_number), tuple(species_order), tuple(species_counts)


def empty_template_cache() -> dict[str, Any]:
    return {
        "version": 1,
        "entries": {},
        "metadata": {},
    }


def load_template_cache(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Wyckoff template cache does not exist: {source}")
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    if "entries" not in payload:
        raise ValueError(f"Invalid Wyckoff template cache: {source}")
    return payload


def save_template_cache(path: str | Path, cache: dict[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(target)


def get_cache_entry(
    cache: dict[str, Any] | None,
    *,
    space_group_number: int,
    atomic_numbers: list[int] | torch.Tensor,
) -> dict[str, Any] | None:
    if not cache:
        return None
    key = template_cache_key(
        space_group_number=int(space_group_number),
        atomic_numbers=atomic_numbers,
    )
    return cache.get("entries", {}).get(key)


def put_cache_entry(
    cache: dict[str, Any],
    *,
    key: TemplateCacheKey,
    templates: list[WyckoffTemplate],
) -> dict[str, Any]:
    entries = cache.setdefault("entries", {})
    entry = entries.get(key)
    if entry is None:
        entry = {
            "space_group": int(key[0]),
            "species_order": tuple(int(v) for v in key[1]),
            "species_counts": tuple(int(v) for v in key[2]),
            "templates": list(templates),
            "true_signature_counts": Counter(),
            "samples_seen": 0,
        }
        entries[key] = entry
    return entry


def add_true_signature(
    entry: dict[str, Any],
    *,
    signature: TemplateSignature,
) -> None:
    counts = entry.get("true_signature_counts")
    if not isinstance(counts, Counter):
        counts = Counter(counts or {})
        entry["true_signature_counts"] = counts
    counts[tuple(signature)] += 1
    entry["samples_seen"] = int(entry.get("samples_seen", 0)) + 1
