from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from .dataset import Carbon24, MP20, MPTS52, Perov5, resolve_data_root
from .transform import (
    ContinuousIntervalLattice,
    FullyConnectedGraph,
    ensure_lattice_standardization_cache,
)


# User-facing dataset names mapped to wrapper classes.
DATASET_REGISTRY = {
    "mp20": MP20,
    "mpts52": MPTS52,
    "perov5": Perov5,
    "carbon24": Carbon24,
}


class CSPTask:
    """Build dataloaders for the Crystal Structure Prediction task.

    CSP setting:
        The atom types/composition are given.
        The model generates:
            - fractional coordinates f
            - lattice parameters l

    This class defines:
        - which dataset to use
        - how to transform each sample
        - how to batch samples for training/evaluation
    """

    def __init__(
        self,
        dataset_name: str = "mp20",
        lattice_parameterization: str = "eps",
        lattice_representation: str = "kldm",
    ) -> None:
        """Configure the CSP task.

        Input:
            dataset_name:
                One of:
                    "mp20", "mpts52", "perov5", "carbon24"

            lattice_parameterization:
                "eps":
                    Use raw lattice features.

                "x0":
                    Use standardized lattice features.

        Output:
            CSPTask object used to construct datasets and dataloaders.
        """
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        if lattice_parameterization not in {"eps", "x0"}:
            raise ValueError("lattice_parameterization must be 'eps' or 'x0'")

        self.dataset_name = dataset_name
        self.lattice_parameterization = lattice_parameterization

    @property
    def dataset_cls(self):
        """Return the dataset wrapper class for the selected dataset."""
        return DATASET_REGISTRY[self.dataset_name]

    @property
    def standardize_lattice(self) -> bool:
        """Return whether lattice features should be standardized.

        Output:
            True for x0-parameterized lattice diffusion.
            False for eps-parameterized lattice diffusion.
        """
        return self.lattice_parameterization == "x0"

    def lattice_stats_path(self, root: str | Path | None = None) -> Path:
        """Return the path to train-set lattice standardization statistics.

        Input:
            root:
                Optional data root.

        Output:
            Path to:
                data/<dataset_name>/train_lattice_stats.json
        """
        root = resolve_data_root(root)
        return root / self.dataset_cls.dataset_name / "train_lattice_stats.json"

    def make_lattice_transform(
        self,
        root: str | Path | None = None,
        download: bool = False,
    ) -> ContinuousIntervalLattice:
        """Create the lattice transform.

        Input:
            root:
                Dataset root.

            download:
                If True, download train CSV if needed before computing stats.

        Output:
            ContinuousIntervalLattice transform.

        Behavior:
            - For eps parameterization:
                no standardization is used.

            - For x0 parameterization:
                train split is loaded/processed,
                train lattice mean/std are computed,
                and the transform uses these statistics.
        """
        root = resolve_data_root(root)
        cache_file = None

        if self.standardize_lattice:
            # Ensure processed train data exists so that cell.npy can be read.
            self.dataset_cls(
                root=root,
                split="train",
                transforms=[],
                download=download,
            )

            cache_file = self.lattice_stats_path(root)

            ensure_lattice_standardization_cache(
                cache_file=cache_file,
                processed_dir=root / self.dataset_cls.dataset_name / "processed" / "train",
            )

        return ContinuousIntervalLattice(
            standardize=self.standardize_lattice,
            cache_file=cache_file,
        )

    def make_transforms(
        self,
        root: str | Path | None = None,
        download: bool = False,
    ) -> list:
        """Create all per-sample transforms used by KLDM.

        Input:
            root:
                Dataset root.

            download:
                Passed to lattice-stat construction if needed.

        Output:
            List of transforms:
                1. FullyConnectedGraph
                2. ContinuousIntervalLattice
        """
        return [
            FullyConnectedGraph(),
            self.make_lattice_transform(root=root, download=download),
        ]

    def fit_dataset(
        self,
        root: str | Path | None = None,
        split: str = "train",
        download: bool = False,
    ):
        """Create a transformed dataset split.

        Input:
            root:
                Dataset root.

            split:
                "train", "val", or "test".

            download:
                If True, download the raw CSV split if missing.

        Output:
            Dataset returning transformed ChemGraph samples.

        Each sample contains at least:
            pos:
                fractional coordinates

            atomic_numbers:
                atom types

            edge_node_index:
                fully connected graph edges

            l:
                6D lattice representation
        """
        root = resolve_data_root(root)

        return self.dataset_cls(
            root=root,
            split=split,
            transforms=self.make_transforms(root=root, download=download),
            download=download,
        )

    def dataloader(
        self,
        root: str | Path | None = None,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 2,
        pin_memory: bool = False,
        download: bool = False,
    ) -> DataLoader:
        """Create a PyTorch dataloader for CSP.

        Input:
            root:
                Dataset root.

            split:
                Dataset split.

            batch_size:
                Number of crystal graphs per batch.

            shuffle:
                Whether to shuffle samples.

            num_workers:
                Number of dataloader workers.

            pin_memory:
                Whether to pin CPU memory for faster GPU transfer.

            download:
                If True, download raw CSV if missing.

        Output:
            DataLoader yielding PyG Batch objects.

        Batch fields used by KLDM:
            batch.pos:
                Fractional coordinates.

            batch.l:
                Lattice features.

            batch.atomic_numbers:
                CSP conditioning composition.

            batch.edge_node_index:
                Fully connected graph edges.

            batch.batch:
                Node-to-graph index.
        """
        dataset = self.fit_dataset(
            root=root,
            split=split,
            download=download,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=dataset.collate_fn,
        )
