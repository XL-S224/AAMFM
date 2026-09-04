from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_NPZ_PATHS: tuple[str, ...] = ()


def load_npz_data(npz_paths: Sequence[str], key: str = "data") -> np.ndarray:
    if not npz_paths:
        raise ValueError("npz_paths cannot be empty.")
    loaded: list[np.ndarray] = []
    for path in npz_paths:
        payload = np.load(path, allow_pickle=True)
        if key not in payload:
            raise KeyError(f"Key '{key}' not found in {path}")
        loaded.append(payload[key])
    if len(loaded) == 1:
        return loaded[0]
    return np.concatenate(loaded, axis=0)


@dataclass
class NPZDatasetConfig:
    npz_paths: Optional[Sequence[str]] = None
    key: str = "data"
    squeeze: bool = True


class NPZAntibodyDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        config = NPZDatasetConfig(**kwargs)
        self.config = config
        npz_paths = list(config.npz_paths) if config.npz_paths else list(DEFAULT_NPZ_PATHS)
        self.data = load_npz_data(npz_paths, key=config.key)

    @classmethod
    def from_npz(
        cls,
        npz_path: str,
        *,
        key: str = "data",
        squeeze: bool = True,
    ) -> "NPZAntibodyDataset":
        return cls(npz_paths=[npz_path], key=key, squeeze=squeeze)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        processed_item = {}
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                value = torch.tensor(value)
                if self.config.squeeze:
                    value = value.squeeze()
            processed_item[key] = value
        return processed_item
