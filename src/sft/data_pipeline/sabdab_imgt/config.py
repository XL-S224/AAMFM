import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class SabdabPaths:
    summary_path: str
    imgt_dir: str
    processed_dir: str
    processed_npz: Optional[str] = None
    cluster_file: Optional[str] = None

    def __post_init__(self) -> None:
        processed_dir_path = Path(self.processed_dir)
        if self.processed_npz is None:
            self.processed_npz = str(processed_dir_path / "processed_single_epitope.npz")
        if self.cluster_file is None:
            self.cluster_file = str(processed_dir_path / "cluster_result_cluster.tsv")


@dataclass
class SabdabFilterConfig:
    allowed_antigen_types: list[str] = field(default_factory=list)
    resolution_threshold: float = 5.0


@dataclass
class SabdabSplitConfig:
    test_pdb_ids: list[str] = field(default_factory=list)
    seed: int = 42
    val_size: int = 20


@dataclass
class SabdabProcessingConfig:
    max_length: int = 1024


@dataclass
class SabdabConfig:
    paths: SabdabPaths
    filtering: SabdabFilterConfig
    split: SabdabSplitConfig = field(default_factory=SabdabSplitConfig)
    processing: SabdabProcessingConfig = field(default_factory=SabdabProcessingConfig)


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        logging.warning("Config file %s not found. Falling back to defaults.", config_path)
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> SabdabConfig:
    config_path = (
        Path(path)
        if path
        else Path(__file__).resolve().parents[2] / "config" / "sabdab_imgt.yaml"
    )
    raw_cfg = _load_yaml_config(config_path)
    if overrides:
        raw_cfg = _deep_update(raw_cfg, overrides)

    paths_cfg = raw_cfg.get("paths", {})
    filter_cfg = raw_cfg.get("filter", {})
    split_cfg = raw_cfg.get("split", {})
    processing_cfg = raw_cfg.get("processing", {})

    paths = SabdabPaths(**paths_cfg)
    filtering = SabdabFilterConfig(**filter_cfg)
    split = SabdabSplitConfig(**split_cfg)
    processing = SabdabProcessingConfig(**processing_cfg)
    return SabdabConfig(paths=paths, filtering=filtering, split=split, processing=processing)
