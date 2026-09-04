import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .config import SabdabConfig
from .entries import SabdabEntry

try:
    import torch  # type: ignore
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - fallback for environments without torch
    torch = None

    class Dataset:  # type: ignore
        def __len__(self) -> int:  # pragma: no cover - minimal stub
            raise NotImplementedError

        def __getitem__(self, idx):  # pragma: no cover - minimal stub
            raise NotImplementedError


class ClusterIndex:
    def __init__(self, cluster_file: str) -> None:
        self.cluster_file = cluster_file
        self.id_to_cluster: Dict[str, str] = {}
        self.cluster_to_ids: Dict[str, List[str]] = {}
        self._load()

    def _load(self) -> None:
        path = Path(self.cluster_file)
        if not path.exists():
            logging.warning("Cluster file %s not found; continuing without clustering.", path)
            return

        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                cluster_name, data_id = parts
                self.id_to_cluster[data_id] = cluster_name
                self.cluster_to_ids.setdefault(cluster_name, []).append(data_id)

    def cluster_for(self, data_id: str) -> str:
        return self.id_to_cluster.get(data_id, f"unclustered::{data_id}")


class SabdabSplitter:
    def __init__(self, config: SabdabConfig, cluster_index: ClusterIndex) -> None:
        self.config = config
        self.cluster_index = cluster_index

    def split(self, entries: Sequence[SabdabEntry]) -> Dict[str, List[str]]:
        test_ids = [entry.id for entry in entries if entry.pdbcode in self.config.split.test_pdb_ids]
        test_clusters = {self.cluster_index.cluster_for(data_id) for data_id in test_ids}

        train_val_ids: List[str] = []
        for entry in entries:
            cluster_name = self.cluster_index.cluster_for(entry.id)
            if cluster_name not in test_clusters:
                train_val_ids.append(entry.id)

        random.Random(self.config.split.seed).shuffle(train_val_ids)
        val_size = min(self.config.split.val_size, len(train_val_ids))

        return {
            "test": test_ids,
            "val": train_val_ids[:val_size],
            "train": train_val_ids[val_size:],
        }


class ProcessedDataset(Dataset):
    def __init__(self, data: Sequence[dict], ids: Sequence[str]) -> None:
        self.data_map = {item["id"]: item for item in data}
        self.ids = list(ids)

    @classmethod
    def from_npz(cls, npz_path: str, ids: Sequence[str]) -> "ProcessedDataset":
        path = Path(npz_path)
        data: List[dict] = []

        if path.exists():
            loaded = np.load(path, allow_pickle=True)["data"]
            data = [item for item in loaded]
        else:
            logging.warning("Processed NPZ file %s not found; dataset will be empty.", path)

        return cls(data=data, ids=ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        data_id = self.ids[idx]
        item = self.data_map.get(data_id)
        if item is None:
            raise KeyError(f"{data_id} not found in processed data")
        return item
