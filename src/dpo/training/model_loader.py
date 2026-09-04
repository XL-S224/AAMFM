from __future__ import annotations

from typing import Tuple

import torch
from esm.models.esm3 import ESM3
from esm.pretrained import (
    ESM3_function_decoder_v0,
    ESM3_structure_decoder_v0,
    ESM3_structure_encoder_v0,
)
from esm.tokenization import get_esm3_model_tokenizers
from esm.utils.constants.models import ESM3_OPEN_SMALL

from sft.training.adapter_gear import ESM3Wrapper

from .config import ModelConfig


_SFT_WRAPPER_ARCHITECTURE = {
    "d_model": 1536,
    "n_heads": 24,
    "v_heads": 256,
    "n_layers": 48,
}


def _build_base_model(_config: ModelConfig) -> ESM3:
    return ESM3(
        **_SFT_WRAPPER_ARCHITECTURE,
        structure_encoder_fn=ESM3_structure_encoder_v0,
        structure_decoder_fn=ESM3_structure_decoder_v0,
        function_decoder_fn=ESM3_function_decoder_v0,
        tokenizers=get_esm3_model_tokenizers(ESM3_OPEN_SMALL),
    )


def _build_wrapped_model(config: ModelConfig) -> torch.nn.Module:
    return ESM3Wrapper(_build_base_model(config))


def build_policy_and_ref_models(
    config: ModelConfig,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    if not config.model_path:
        raise ValueError("model.model_path is required")
    policy_model = _build_wrapped_model(config)
    ref_model = _build_wrapped_model(config)

    state_dict = torch.load(config.model_path, map_location="cpu")
    policy_model.load_state_dict(state_dict, strict=config.strict_load)
    ref_model.load_state_dict(state_dict, strict=config.strict_load)

    for parameter in ref_model.parameters():
        parameter.requires_grad = False
    ref_model.eval()
    return policy_model, ref_model
