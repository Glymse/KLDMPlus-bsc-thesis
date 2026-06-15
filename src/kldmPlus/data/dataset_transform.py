from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from mattergen.common.data.dataset import BaseDataset, CrystalDataset
from pymatgen.core.periodic_table import Element


def filter_elements(
    dataset: CrystalDataset,
    exclude_elements: Iterable[int | str],
) -> CrystalDataset:
    """Remove structures containing any excluded element.

    Elements can be given either as atomic numbers, e.g. 86,
    or symbols, e.g. "Rn".
    """
    excluded_z: set[int] = set()

    for element in exclude_elements:
        if isinstance(element, int):
            excluded_z.add(element)
        elif isinstance(element, str):
            excluded_z.add(Element(element).Z)
        else:
            raise TypeError(f"Unsupported element type: {type(element)}")

    atom_is_excluded = np.isin(dataset.atomic_numbers, list(excluded_z))
    excluded_count_per_structure = np.add.reduceat(atom_is_excluded, dataset.index_offset)

    keep_indices = np.nonzero(excluded_count_per_structure == 0)[0]
    return dataset.subset(list(keep_indices))


def filter_energy_above_hull(
    dataset: BaseDataset,
    threshold: float = 0.1,
) -> BaseDataset:
    """Keep only structures with energy above hull <= threshold."""
    if "energy_above_hull" not in dataset.properties:
        return dataset

    energies = dataset.properties["energy_above_hull"]
    keep_indices = np.where(energies <= threshold)[0]
    return dataset.subset(list(keep_indices))
