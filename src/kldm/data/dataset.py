from __future__ import annotations

from pathlib import Path

import requests
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.dataset import CrystalDataset, CrystalDatasetBuilder, DatasetTransform
from mattergen.common.data.transform import Transform
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from tqdm.auto import tqdm




WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data"


def resolve_data_root(root: str | Path | None = None) -> Path:
    """The data root. """
    return DEFAULT_DATA_ROOT if root is None else Path(root).expanduser()


class CrystalDatasetWrapper(Dataset):
    """Wrapper around MatterGen's CrystalDatases from diffCSP.

    This class handles three jobs:

    1. Optionally download the raw CSV split.
    2. Load a processed MatterGen cache if it already exists.
    3. Otherwise build the processed cache from the raw CSV.

    Output:
        A PyTorch Dataset returning MatterGen ChemGraph objects.
    """

    dataset_name: str
    url: str

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transforms: list[Transform] | None = None,
        dataset_transforms: list[DatasetTransform] | None = None,
        download: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of 'train', 'val', or 'test'")

        # Store configuration.
        self.root = Path(root).expanduser()
        self.split = split
        self.transforms = transforms or []
        self.dataset_transforms = dataset_transforms or []

        #Download raw CSV only when explicitly requested.
        if download:
            self.download()

        # Build or load the MatterGen CrystalDataset.
        self.data = self._build()

    @property
    def raw_folder(self) -> Path:
        #Folder containing raw CSV files.
        return self.root / self.dataset_name / "raw"

    @property
    def processed_folder(self) -> Path:
        #Folder containing processed MatterGen caches.
        return self.root / self.dataset_name / "processed"

    @property
    def raw_csv(self) -> Path:
        #Raw CSV path for the selected split.
        return self.raw_folder / f"{self.split}.csv"

    @property
    def processed_split_folder(self) -> Path:
        #Processed cache path for the selected split.
        return self.processed_folder / self.split

    @staticmethod
    def collate_fn(samples: list[ChemGraph]) -> Batch:
        #Convert a list of ChemGraph samples into one PyG Batch.
        return Batch.from_data_list(samples)

    def _build(self) -> CrystalDataset:
        """Load processed cache or build it from raw CSV.

        Output:
            MatterGen CrystalDataset.
        """
        if self.processed_split_folder.exists():
            # Fast path: load cached arrays.
            builder = CrystalDatasetBuilder.from_cache_path(
                cache_path=str(self.processed_split_folder),
                transforms=self.transforms,
            )
        else:
            # Slow path: parse raw CSV and create processed cache.
            if not self.raw_csv.exists():
                raise RuntimeError(
                    f"Raw split not found at {self.raw_csv}. Pass download=True to fetch it first."
                )
            # Code segment inspired from mattergen
            # (mattergen/common/data/dataset.py:528-556,
            #  mattergen/common/data/dataset.py:354-360).
            #
            # Important preprocessing note:
            # CrystalDatasetBuilder.from_csv(...) already parses CIFs, converts
            # them to primitive structures, and applies `get_reduced_structure()`
            # before writing the processed cache. We intentionally rely on that
            # upstream full-structure reduction here instead of trying to reduce
            # only the lattice matrix later in a transform.
            builder = CrystalDatasetBuilder.from_csv(
                csv_path=str(self.raw_csv),
                cache_path=str(self.processed_split_folder),
                transforms=self.transforms,
            )

        return builder.build(
            dataset_class=CrystalDataset,
            dataset_transforms=self.dataset_transforms,
        )

    def download(self) -> None:
        """
        Download the raw CSV split if missing.
        Output:
                data/<dataset_name>/raw/<split>.csv
        """
        if self.raw_csv.exists():
            return

        self.raw_folder.mkdir(parents=True, exist_ok=True)

        response = requests.get(
            self.url + f"{self.split}.csv",
            stream=True,
            timeout=40,
        )
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            self.raw_csv.open("wb") as handle,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {self.dataset_name} {self.split}",
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    handle.write(chunk)
                    pbar.update(len(chunk))

    def __getitem__(self, index: int) -> ChemGraph:
        """Return one crystal graph.

        Input:
            index:
                Integer sample index.

        Output:
            ChemGraph containing fields such as:
                pos, cell, atomic_numbers, num_atoms, etc.
        """
        return self.data[index]

    def __len__(self) -> int:
        """Return number of structures in the selected split."""
        return len(self.data)


class Carbon24(CrystalDatasetWrapper):
    """Carbon-24 dataset wrapper."""
    dataset_name = "carbon_24"
    url = "https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/carbon_24/"


class MP20(CrystalDatasetWrapper):
    """MP-20 dataset wrapper."""
    dataset_name = "mp_20"
    url = "https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/mp_20/"


class MPTS52(CrystalDatasetWrapper):
    """MPTS-52 dataset wrapper."""
    dataset_name = "mpts_52"
    url = "https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/mpts_52/"


class Perov5(CrystalDatasetWrapper):
    """Perov-5 dataset wrapper."""
    dataset_name = "perov_5"
    url = "https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/perov_5/"
