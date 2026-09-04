from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type, TypeVar

import yaml


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "DPO.yaml"


def _serialize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            config_field.name: _serialize(getattr(value, config_field.name))
            for config_field in fields(value)
        }
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


@dataclass
class CommonConfig:
    seed: int = 42


@dataclass
class DataConfig:
    train_csv: Optional[str] = None
    val_csv: Optional[str] = None
    val_split: float = 0.05
    min_score_diff: float = 0.2
    top_k: int = 2
    loss_type: str = "paired"
    max_length: int = 1024
    summary_path: Optional[str] = None
    imgt_dir: Optional[str] = None
    processed_dir: Optional[str] = None
    antigen_feature_path: Optional[str] = None
    antigen_feature_id_key: str = "pdbid"
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class ModelConfig:
    model_path: Optional[str] = None
    ref_model_device: str = "same"
    strict_load: bool = True

@dataclass
class TokenConfig:
    mask_token_id: int = 32
    pad_token_id: int = 1
    chain_token_id: int = 31


@dataclass
class TrainingConfig:
    lr: float = 5e-5
    batch_size: int = 1
    epochs: int = 10
    accumulation_steps: int = 32
    max_grad_norm: float = 1.0
    beta: float = 0.1
    mixed_precision: str = "bf16"
    normalize_logps: bool = True
    warmup_ratio: float = 0.1

@dataclass
class EvalConfig:
    eval_every_steps: int = 0
    every_save_steps: int = 40000
    output_dir: Optional[str] = None
    metric_for_best: str = "val/loss"
    greater_is_better: bool = False


@dataclass
class WandbConfig:
    wandb_enabled: bool = True
    wandb_project: str = "cdr_ranked_dpo"
    log_every_steps: int = 50


@dataclass
class CheckpointConfig:
    save_epochs: bool = True
    resume_from: Optional[str] = None
    auto_resume: bool = True

@dataclass
class DPOConfig:
    common: CommonConfig = field(default_factory=CommonConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tokens: TokenConfig = field(default_factory=TokenConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    debug: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_yaml(cls, path: str) -> "DPOConfig":
        return load_dpo_config(path)

    @classmethod
    def from_dict(
        cls, raw: Dict[str, Any], base_dir: Optional[Path] = None
    ) -> "DPOConfig":
        del base_dir
        return _parse_dpo_config(raw)


T = TypeVar("T")

_SECTION_TYPES: Dict[str, Type[Any]] = {
    "common": CommonConfig,
    "data": DataConfig,
    "model": ModelConfig,
    "tokens": TokenConfig,
    "training": TrainingConfig,
    "eval": EvalConfig,
    "wandb": WandbConfig,
    "checkpoint": CheckpointConfig,
}
_ALLOWED_TOP_LEVEL_KEYS = {*_SECTION_TYPES, "logging", "debug"}
_LEGACY_SFT_WRAPPER_DIMENSIONS = {
    "d_model": 1536,
    "n_heads": 24,
    "v_heads": 256,
    "n_layers": 48,
}


def _mapping(value: Any, section: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Section '{section}' must be a mapping.")
    return dict(value)


def _move_legacy_key(
    source: Dict[str, Any], target: Dict[str, Any], old_key: str, new_key: str
) -> None:
    if old_key in source:
        value = source.pop(old_key)
        target.setdefault(new_key, value)


def _drop_legacy_wrapper_dimensions(model: Dict[str, Any]) -> None:
    for name, expected in _LEGACY_SFT_WRAPPER_DIMENSIONS.items():
        if name not in model:
            continue
        actual = model.pop(name)
        if actual != expected:
            raise ValueError(
                f"DPO model.{name} must be {expected} to match the fixed SFT wrapper "
                f"architecture; got {actual!r}."
            )


def _normalize_legacy_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    for key in raw:
        if key not in _ALLOWED_TOP_LEVEL_KEYS:
            raise ValueError(f"Unknown DPO config field: root.{key}")

    normalized = {
        section: _mapping(raw.get(section), section) for section in _SECTION_TYPES
    }
    logging = _mapping(raw.get("logging"), "logging")

    _move_legacy_key(
        normalized["model"], normalized["model"], "pretrained_model_path", "model_path"
    )
    _drop_legacy_wrapper_dimensions(normalized["model"])
    _move_legacy_key(normalized["training"], normalized["training"], "learning_rate", "lr")
    _move_legacy_key(normalized["training"], normalized["training"], "num_epochs", "epochs")
    _move_legacy_key(
        normalized["training"],
        normalized["training"],
        "gradient_accumulation_steps",
        "accumulation_steps",
    )
    _move_legacy_key(normalized["training"], normalized["common"], "seed", "seed")

    _move_legacy_key(normalized["data"], normalized["tokens"], "mask_idx", "mask_token_id")
    _move_legacy_key(normalized["data"], normalized["tokens"], "pad_idx", "pad_token_id")
    _move_legacy_key(normalized["data"], normalized["tokens"], "chain_idx", "chain_token_id")

    _move_legacy_key(logging, normalized["wandb"], "log_with_wandb", "wandb_enabled")
    _move_legacy_key(logging, normalized["wandb"], "project_name", "wandb_project")
    _move_legacy_key(logging, normalized["wandb"], "log_every_steps", "log_every_steps")
    for key in logging:
        raise ValueError(f"Unknown DPO config field: logging.{key}")

    _move_legacy_key(
        normalized["checkpoint"], normalized["eval"], "save_steps", "every_save_steps"
    )
    _move_legacy_key(
        normalized["checkpoint"], normalized["eval"], "output_dir", "output_dir"
    )
    return normalized


def _validate_fields(section: str, payload: Mapping[str, Any], section_type: Type[T]) -> None:
    field_names = {config_field.name for config_field in fields(section_type)}
    for key in payload:
        if key not in field_names:
            raise ValueError(f"Unknown DPO config field: {section}.{key}")


def _parse_dpo_config(raw: Dict[str, Any]) -> DPOConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("DPO config must be a mapping.")

    normalized = _normalize_legacy_config(dict(raw))
    sections: Dict[str, Any] = {}
    for section, section_type in _SECTION_TYPES.items():
        payload = normalized[section]
        _validate_fields(section, payload, section_type)
        sections[section] = section_type(**payload)

    debug = raw.get("debug", False)
    if not isinstance(debug, bool):
        raise ValueError("Section 'debug' must be a boolean.")
    return DPOConfig(**sections, debug=debug)


def load_dpo_config(path: Optional[str] = None) -> DPOConfig:
    config_path = Path(path) if path else _default_config_path()
    if path and not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.exists():
        return DPOConfig()
    with config_path.open("r") as config_file:
        raw = yaml.safe_load(config_file) or {}
    return _parse_dpo_config(raw)


__all__ = [
    "CheckpointConfig",
    "CommonConfig",
    "DataConfig",
    "DPOConfig",
    "EvalConfig",
    "ModelConfig",
    "TokenConfig",
    "TrainingConfig",
    "WandbConfig",
    "load_dpo_config",
]
