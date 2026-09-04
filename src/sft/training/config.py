from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, TypeVar

import yaml


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "SFT.yaml"


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


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


def _normalize_training_section(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(raw)
    betas = data.get("betas")
    if betas is not None and not isinstance(betas, tuple):
        data["betas"] = tuple(betas)
    return data


def _normalize_masking_section(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(raw)
    mixed_weights = data.get("mixed_weights")
    if mixed_weights is not None and not isinstance(mixed_weights, tuple):
        data["mixed_weights"] = tuple(mixed_weights)
    random_beta = data.get("random_beta")
    if random_beta is not None and not isinstance(random_beta, tuple):
        data["random_beta"] = tuple(random_beta)
    return data


@dataclass
class DataConfig:
    max_length: int = 1024
    summary_path: Optional[str] = None
    imgt_dir: Optional[str] = None
    processed_dir: Optional[str] = None
    init: bool = False
    train_split: str = "train"
    val_split: str = "test"
    batch_size: int = 1
    num_workers: int = 6
    antigen_feature_path: Optional[str] = None
    antigen_feature_id_key: str = "pdbid"


@dataclass
class ModelConfig:
    model_path: Optional[str] = None
    freeze_output_heads: bool = True
    adapter_module: Optional[str] = None
    adapter_wrapper: Optional[str] = None
    adapter_freeze_fn: Optional[str] = None
    adapter_state_path: Optional[str] = None
    adapter_param_keyword: str = "adapter"


@dataclass
class CommonConfig:
    seed: Optional[int] = None


@dataclass
class TokenConfig:
    mask_token_id: int = 32
    pad_token_id: int = 1
    chain_token_id: int = 31
    structure_mask_id: int = 4096
    structure_special_tokens: Sequence[int] = field(
        default_factory=lambda: [4096, 4097, 4098, 4099, 4100]
    )


@dataclass
class MaskingConfig:
    strategy: str = "mixed"
    mixed_weights: Tuple[float, float, float] = (0.25, 0.25, 0.5)
    random_mask_prob: Optional[float] = None
    random_beta: Tuple[float, float] = (3.0, 9.0)
    single_cdr_index: Optional[int] = None
    random_single_cdr_prob: float = 0.25
    mask_structure: bool = True


@dataclass
class TrainingConfig:
    mode: str = "imgt"
    epochs: int = 1
    accumulation_steps: int = 1
    lr: float = 1e-5
    weight_decay: float = 0.01
    betas: Optional[Tuple[float, float]] = None
    mixed_precision: str = "bf16"
    max_grad_norm: float = 1.0
    scheduler: str = "linear"
    warmup_ratio: float = 0.1
    unfreeze_epoch: Optional[int] = None
    backbone_lr: Optional[float] = None
    adapter_lr: Optional[float] = None
    use_structure_loss: bool = True
    structure_loss_weight: float = 1.0


@dataclass
class EvalConfig:
    eval_every_steps: int = 0
    every_save_steps: int = 0
    output_dir: str = "outputs/sft"
    eval_strategy: Optional[str] = None


@dataclass
class wandbConfig:
    wandb_enabled: bool = False
    wandb_project: str = "sft"
    wandb_name: Optional[str] = None
    wandb_dir: Optional[str] = None
    wandb_entity: Optional[str] = None


@dataclass
class SFTConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    tokens: TokenConfig = field(default_factory=TokenConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: wandbConfig = field(default_factory=wandbConfig)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(self)


def load_sft_config(path: Optional[str] = None) -> SFTConfig:
    config_path = Path(path) if path else _default_config_path()
    if path and not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = _load_yaml_config(config_path)
    if raw and not isinstance(raw, dict):
        raise ValueError(f"Config file must be a mapping: {config_path}")
    raw = raw or {}
    return SFTConfig(
        data=_extract_section(raw, "data", DataConfig),
        model=_extract_section(raw, "model", ModelConfig),
        common=_extract_section(raw, "common", CommonConfig),
        tokens=_extract_section(raw, "tokens", TokenConfig),
        masking=_extract_section(
            raw, "masking", MaskingConfig, _normalize_masking_section
        ),
        training=_extract_section(
            raw, "training", TrainingConfig, _normalize_training_section
        ),
        eval=_extract_section(raw, "eval", EvalConfig),
        wandb=_extract_section(raw, "wandb", wandbConfig),
    )
