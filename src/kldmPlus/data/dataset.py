from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import socket
import time
import shutil
from typing import Any

import requests
from torch.utils.data import Dataset
from tqdm.auto import tqdm

try:
    from torch_geometric.data import Batch
except ImportError:  # pragma: no cover
    Batch = Any

try:
    import numpy as np
    import torch
    from mattergen.common.data.chemgraph import ChemGraph
    from mattergen.common.data.dataset import CrystalDataset, CrystalDatasetBuilder, DatasetTransform
    from mattergen.common.data.transform import Transform
    from pymatgen.symmetry.groups import SpaceGroup
    MATTERGEN_AVAILABLE = True
    MATTERGEN_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover
    ChemGraph = Any
    CrystalDataset = Any
    CrystalDatasetBuilder = Any
    DatasetTransform = Any
    SpaceGroup = Any
    MATTERGEN_AVAILABLE = False
    MATTERGEN_IMPORT_ERROR = exc

    class Transform:  # type: ignore[override]
        pass




WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data"
SPACE_GROUP_PROPERTY = "space_group"
SPACE_GROUP_COLUMN = "spacegroup.number"
CACHE_SCHEMA_VERSION = 2
CACHE_META_FILENAME = "_kldm_cache_meta.json"
CORE_CACHE_FILENAMES = ("cell.npy", "atomic_numbers.npy", "num_atoms.npy", "pos.npy")
BUILD_LOCK_DIRNAME = ".kldm_build_lock"
BUILD_LOCK_STALE_SECONDS = 6 * 60 * 60
BUILD_LOCK_POLL_SECONDS = 5.0
BUILD_LOCK_WAIT_LOG_SECONDS = 60.0


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
        required_properties: list[str] | None = None,
        download: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of 'train', 'val', or 'test'")
        if not MATTERGEN_AVAILABLE:
            message = "kldmPlus.data.dataset requires mattergen to be importable."
            if MATTERGEN_IMPORT_ERROR is not None:
                message = f"{message} Underlying import error: {type(MATTERGEN_IMPORT_ERROR).__name__}: {MATTERGEN_IMPORT_ERROR}"
            raise ImportError(message)

        # Store configuration.
        self.root = Path(root).expanduser()
        self.split = split
        self.transforms = transforms or []
        self.dataset_transforms = dataset_transforms or []
        requested_properties = list(required_properties or [SPACE_GROUP_PROPERTY])
        if SPACE_GROUP_PROPERTY not in requested_properties:
            requested_properties.append(SPACE_GROUP_PROPERTY)
        self._required_properties = sorted(set(str(name) for name in requested_properties))
        self._space_group_number_map: dict[str, int] | None = None

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

    @property
    def cache_meta_path(self) -> Path:
        return self.processed_split_folder / CACHE_META_FILENAME

    @property
    def cache_lock_dir(self) -> Path:
        return self.processed_folder / f"{self.split}_{BUILD_LOCK_DIRNAME}"

    @property
    def required_properties(self) -> list[str]:
        return list(self._required_properties)

    @staticmethod
    def collate_fn(samples: list[ChemGraph]) -> Batch:
        #Convert a list of ChemGraph samples into one PyG Batch.
        return Batch.from_data_list(samples)

    @staticmethod
    def _space_group_symbol(number: int) -> str:
        return str(SpaceGroup.from_int_number(int(number)).symbol)

    def _load_space_group_number_map(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        with self.raw_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                material_id = str(row["material_id"])
                value = row.get(SPACE_GROUP_COLUMN)
                if value is None or value == "":
                    continue
                mapping[material_id] = int(float(value))
        return mapping

    def _load_space_group_symbol_map(self) -> dict[str, str]:
        number_map = self._load_space_group_number_map()
        return {
            material_id: self._space_group_symbol(number)
            for material_id, number in number_map.items()
        }

    def _ensure_required_properties(self, builder: CrystalDatasetBuilder) -> None:
        unsupported = sorted(set(self.required_properties) - {SPACE_GROUP_PROPERTY})
        if unsupported:
            raise RuntimeError(
                f"Cannot backfill requested cache properties {unsupported}. "
                f"Supported backfilled property: {SPACE_GROUP_PROPERTY!r}."
            )
        if not self.raw_csv.exists():
            raise RuntimeError(
                f"Raw split not found at {self.raw_csv}. Cannot backfill {SPACE_GROUP_PROPERTY!r}."
            )

        # Always overwrite the cached space-group property from the raw CSV.
        # Older caches may contain numeric values, but MatterGen expects the
        # cached property in Hermann-Mauguin symbol form and will normalize it
        # back to an int internally.
        builder.add_property_to_cache(
            SPACE_GROUP_PROPERTY,
            data=self._load_space_group_symbol_map(),
        )

    def _raw_signature(self) -> dict[str, int] | None:
        if not self.raw_csv.exists():
            return None
        stat = self.raw_csv.stat()
        return {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _cache_meta_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "required_properties": sorted(self.required_properties),
            "raw_csv": str(self.raw_csv),
            "raw_signature": self._raw_signature(),
        }

    def _read_cache_meta(self) -> dict[str, Any] | None:
        if not self.cache_meta_path.exists():
            return None
        try:
            with self.cache_meta_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    def _write_cache_meta(self) -> None:
        self.processed_split_folder.mkdir(parents=True, exist_ok=True)
        payload = self._cache_meta_payload()
        with self.cache_meta_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _missing_cache_files(self) -> list[str]:
        missing: list[str] = []
        for filename in CORE_CACHE_FILENAMES:
            path = self.processed_split_folder / filename
            try:
                usable = path.is_file() and path.stat().st_size > 0
            except OSError:
                usable = False
            if not usable:
                missing.append(filename)
        for property_name in self.required_properties:
            path = self.processed_split_folder / f"{property_name}.json"
            try:
                usable = path.is_file() and path.stat().st_size > 0
            except OSError:
                usable = False
            if not usable:
                missing.append(path.name)
        return missing

    def _cache_status(self) -> tuple[bool, str]:
        if not self.processed_split_folder.exists():
            return False, "missing_processed_split"
        missing_files = self._missing_cache_files()
        if missing_files:
            return False, f"missing_cache_files={','.join(missing_files)}"

        meta = self._read_cache_meta()
        if meta is None:
            if self.raw_csv.exists():
                return False, "missing_or_invalid_meta"
            return True, "using_legacy_cache_without_raw_csv"

        if int(meta.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
            return False, "schema_version_mismatch"
        if str(meta.get("dataset_name")) != str(self.dataset_name):
            return False, "dataset_name_mismatch"
        if str(meta.get("split")) != str(self.split):
            return False, "split_mismatch"
        if sorted(meta.get("required_properties", [])) != sorted(self.required_properties):
            return False, "required_properties_mismatch"
        if str(meta.get("raw_csv")) != str(self.raw_csv):
            return False, "raw_csv_path_mismatch"
        if meta.get("raw_signature") != self._raw_signature():
            return False, "raw_signature_mismatch"
        return True, "fresh"

    def _rebuild_cache(self) -> CrystalDatasetBuilder:
        if not self.raw_csv.exists():
            raise RuntimeError(
                f"Raw split not found at {self.raw_csv}. Pass download=True to fetch it first."
            )
        if self.processed_split_folder.exists():
            shutil.rmtree(self.processed_split_folder, ignore_errors=True)
        print(
            f"dataset_cache action=rebuild dataset={self.dataset_name} split={self.split} "
            f"path={self.processed_split_folder}",
            flush=True,
        )
        print(
            f"dataset_cache action=from_csv:start dataset={self.dataset_name} split={self.split}",
            flush=True,
        )
        builder = CrystalDatasetBuilder.from_csv(
            csv_path=str(self.raw_csv),
            cache_path=str(self.processed_split_folder),
            transforms=self.transforms,
        )
        print(
            f"dataset_cache action=from_csv:done dataset={self.dataset_name} split={self.split}",
            flush=True,
        )
        print(
            f"dataset_cache action=ensure_required_properties:start dataset={self.dataset_name} split={self.split}",
            flush=True,
        )
        self._ensure_required_properties(builder)
        print(
            f"dataset_cache action=ensure_required_properties:done dataset={self.dataset_name} split={self.split}",
            flush=True,
        )
        print(
            f"dataset_cache action=write_meta:start dataset={self.dataset_name} split={self.split}",
            flush=True,
        )
        self._write_cache_meta()
        print(
            f"dataset_cache action=write_meta:done dataset={self.dataset_name} split={self.split}",
            flush=True,
        )
        return builder

    def _read_lock_owner(self) -> dict[str, str]:
        owner_path = self.cache_lock_dir / "owner.txt"
        try:
            return dict(
                line.strip().split("=", 1)
                for line in owner_path.read_text(encoding="utf-8").splitlines()
                if "=" in line
            )
        except OSError:
            return {}

    def _lock_is_stale(self) -> tuple[bool, str]:
        if not self.cache_lock_dir.exists():
            return False, "missing_lock"
        try:
            age_seconds = time.time() - self.cache_lock_dir.stat().st_mtime
        except OSError:
            return False, "stat_failed"
        owner = self._read_lock_owner()
        owner_host = owner.get("host")
        owner_pid = owner.get("pid")
        current_host = socket.gethostname()
        if owner_host == current_host and owner_pid is not None:
            try:
                os.kill(int(owner_pid), 0)
            except ProcessLookupError:
                return True, f"dead_owner_pid pid={owner_pid} host={owner_host}"
            except (PermissionError, ValueError):
                pass
        if age_seconds > BUILD_LOCK_STALE_SECONDS:
            return True, f"age_seconds={age_seconds:.0f}"
        return False, f"age_seconds={age_seconds:.0f}"

    def _acquire_build_lock(self) -> bool:
        self.processed_folder.mkdir(parents=True, exist_ok=True)
        last_wait_log = 0.0
        while True:
            try:
                self.cache_lock_dir.mkdir(parents=False, exist_ok=False)
                marker = self.cache_lock_dir / "owner.txt"
                marker.write_text(
                    f"pid={os.getpid()}\nhost={socket.gethostname()}\ntime={time.time():.0f}\n",
                    encoding="utf-8",
                )
                print(
                    f"dataset_cache_lock action=acquired dataset={self.dataset_name} split={self.split} "
                    f"path={self.cache_lock_dir}",
                    flush=True,
                )
                return True
            except FileExistsError:
                cache_fresh, cache_reason = self._cache_status()
                if cache_fresh:
                    print(
                        f"dataset_cache_lock action=skip_wait dataset={self.dataset_name} split={self.split} "
                        f"reason={cache_reason}",
                        flush=True,
                    )
                    return False
                lock_stale, stale_reason = self._lock_is_stale()
                if lock_stale:
                    print(
                        f"dataset_cache_lock action=remove_stale dataset={self.dataset_name} split={self.split} "
                        f"reason={stale_reason} path={self.cache_lock_dir}",
                        flush=True,
                    )
                    shutil.rmtree(self.cache_lock_dir, ignore_errors=True)
                    continue
                now = time.time()
                if now - last_wait_log >= BUILD_LOCK_WAIT_LOG_SECONDS:
                    owner = self._read_lock_owner()
                    owner_text = " ".join(f"{key}={value}" for key, value in sorted(owner.items()))
                    print(
                        f"dataset_cache_lock action=wait dataset={self.dataset_name} split={self.split} "
                        f"reason={stale_reason} owner='{owner_text}' path={self.cache_lock_dir}",
                        flush=True,
                    )
                    last_wait_log = now
                time.sleep(BUILD_LOCK_POLL_SECONDS)

    def _release_build_lock(self) -> None:
        if self.cache_lock_dir.exists():
            shutil.rmtree(self.cache_lock_dir, ignore_errors=True)
            print(
                f"dataset_cache_lock action=released dataset={self.dataset_name} split={self.split} "
                f"path={self.cache_lock_dir}",
                flush=True,
            )

    def _build(self) -> CrystalDataset:
        """Load processed cache or build it from raw CSV.

        Output:
            MatterGen CrystalDataset.
        """
        cache_fresh, cache_reason = self._cache_status()
        try:
            if cache_fresh:
                print(
                    f"dataset_cache action=load dataset={self.dataset_name} split={self.split} "
                    f"reason={cache_reason} path={self.processed_split_folder}",
                    flush=True,
                )
                print(
                    f"dataset_cache action=from_cache_path:start dataset={self.dataset_name} split={self.split}",
                    flush=True,
                )
                builder = CrystalDatasetBuilder.from_cache_path(
                    cache_path=str(self.processed_split_folder),
                    transforms=self.transforms,
                )
                print(
                    f"dataset_cache action=from_cache_path:done dataset={self.dataset_name} split={self.split}",
                    flush=True,
                )
            else:
                acquired_lock = self._acquire_build_lock()
                try:
                    cache_fresh, cache_reason = self._cache_status()
                    if cache_fresh:
                        print(
                            f"dataset_cache action=load dataset={self.dataset_name} split={self.split} "
                            f"reason={cache_reason} path={self.processed_split_folder}",
                            flush=True,
                        )
                        print(
                            f"dataset_cache action=from_cache_path:start dataset={self.dataset_name} split={self.split}",
                            flush=True,
                        )
                        builder = CrystalDatasetBuilder.from_cache_path(
                            cache_path=str(self.processed_split_folder),
                            transforms=self.transforms,
                        )
                        print(
                            f"dataset_cache action=from_cache_path:done dataset={self.dataset_name} split={self.split}",
                            flush=True,
                        )
                    else:
                        print(
                            f"dataset_cache action=stale dataset={self.dataset_name} split={self.split} "
                            f"reason={cache_reason} path={self.processed_split_folder}",
                            flush=True,
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
                        builder = self._rebuild_cache()
                finally:
                    if acquired_lock:
                        self._release_build_lock()

            print(
                f"dataset_cache action=builder_build:start dataset={self.dataset_name} split={self.split}",
                flush=True,
            )
            dataset = builder.build(
                dataset_class=CrystalDataset,
                dataset_transforms=self.dataset_transforms,
            )
            print(
                f"dataset_cache action=builder_build:done dataset={self.dataset_name} split={self.split}",
                flush=True,
            )
            return dataset
        except Exception as exc:
            print(
                f"dataset_cache action=error dataset={self.dataset_name} split={self.split} "
                f"error_type={type(exc).__name__} error={exc}",
                flush=True,
            )
            raise

    def _space_group_for_structure_id(self, structure_id: str) -> int:
        if self._space_group_number_map is None:
            if not self.raw_csv.exists():
                raise RuntimeError(
                    f"Raw split not found at {self.raw_csv}. Cannot attach {SPACE_GROUP_PROPERTY!r}."
                )
            self._space_group_number_map = self._load_space_group_number_map()

        try:
            return int(self._space_group_number_map[structure_id])
        except KeyError as exc:
            raise KeyError(f"Missing {SPACE_GROUP_PROPERTY!r} for structure_id={structure_id!r}.") from exc

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
        sample = self.data[index]
        structure_id = str(self.data.structure_id[index])
        space_group = torch.tensor(
            self._space_group_for_structure_id(structure_id),
            dtype=torch.long,
        )
        return sample.replace(space_group=space_group)

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
