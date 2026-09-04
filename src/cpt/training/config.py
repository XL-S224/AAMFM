from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import yaml


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "CPT.yaml"


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _parse_npz_paths(raw: Optional[str | Sequence[str]]) -> Optional[Sequence[str]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [str(p).strip() for p in raw if str(p).strip()]
    if isinstance(raw, str):
        paths: list[str] = []
        if "," in raw:
            paths.extend([p.strip() for p in raw.split(",") if p.strip()])
        else:
            paths.append(raw)
        return paths
    return None


def _normalize_data_section(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(raw)
    if "npz" in data and "npz_paths" not in data:
        data["npz_paths"] = data.pop("npz")
    data["npz_paths"] = _parse_npz_paths(data.get("npz_paths"))
    data["npz_paths_eval"] = _parse_npz_paths(data.get("npz_paths_eval"))
    return data


def _normalize_training_section(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(raw)
    if "warmup_ratio" in data and "warmup_radio" not in data:
        data["warmup_radio"] = data["warmup_ratio"]
    betas = data.get("betas")
    if betas is not None and not isinstance(betas, tuple):
        data["betas"] = tuple(betas)
    return data


def _serialize(value: Any) -> Any:
    if dataclass_isinstance(value):
        return {
            field.name: _serialize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(val) for key, val in value.items()}
    return value


def dataclass_isinstance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__")


T = TypeVar("T")


def _extract_section(
    raw: Dict[str, Any],
    section: str,
    section_type: Type[T],
    normalizer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> T:
    """
    Parse and load the config dataclass from raw data.
    """
    section_raw = raw.get(section)
    if section_raw is None:
        section_raw = {}
    if section_raw and not isinstance(section_raw, dict):
        raise ValueError(f"Section '{section}' must be a mapping.")

    field_names = {field.name for field in fields(section_type)}
    legacy_raw = {key: value for key, value in raw.items() if key in field_names}
    merged = {**legacy_raw, **section_raw}
    if normalizer is not None:
        merged = normalizer(merged)
    payload = {key: value for key, value in merged.items() if key in field_names}
    return section_type(**payload)


@dataclass
class DataConfig:
    npz_paths: Optional[Sequence[str]] = None
    npz_paths_eval: Optional[Sequence[str]] = None
    npz_key: str = "data"
    batch_size: int = 64
    num_workers: int = 4
    train_ratio: float = 0.95


@dataclass
class ModelConfig:
    model_name: Optional[str] = None
    model_path: Optional[str] = None


@dataclass
class CommonConfig:
    seed: Optional[int] = None


@dataclass
class TrainingConfig:
    epochs: int = 1
    accumulation_steps: int = 1
    lr_max: float = 5e-5
    lr_min: float = 1e-6
    decay_steps: int = 0
    decay_ratio: float = 0.5
    weight_decay: float = 0.01
    scheduler_epochs: int = 1
    warmup_steps: int = 0
    warmup_radio: Optional[float] = None
    betas: Optional[Tuple[float, float]] = None
    mixed_precision: str = "bf16"


@dataclass
class EvalConfig:
    eval_every_steps: int = 0
    every_save_steps: int = 0
    output_dir: str = "outputs/cpt"


@dataclass
class wandbConfig:
    wandb_enabled: bool = False
    wandb_project: str = "cpt"
    wandb_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_dir: Optional[str] = None


@dataclass
class FullFinetuneConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: wandbConfig = field(default_factory=wandbConfig)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(self)


def load_full_finetune_config(path: Optional[str] = None) -> FullFinetuneConfig:
    config_path = Path(path) if path else _default_config_path()
    if path and not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = _load_yaml_config(config_path)
    if raw and not isinstance(raw, dict):
        raise ValueError(f"Config file must be a mapping: {config_path}")
    raw = raw or {}
    return FullFinetuneConfig(
        data=_extract_section(raw, "data", DataConfig, _normalize_data_section),
        model=_extract_section(raw, "model", ModelConfig),
        common=_extract_section(raw, "common", CommonConfig),
        training=_extract_section(
            raw, "training", TrainingConfig, _normalize_training_section
        ),
        eval=_extract_section(raw, "eval", EvalConfig),
        wandb=_extract_section(raw, "wandb", wandbConfig),
    )
