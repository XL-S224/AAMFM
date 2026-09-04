from __future__ import annotations

import math
import os
import random
import sys
from datetime import datetime
from importlib import import_module
from typing import Optional, Sequence

import numpy as np
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from esm.models.esm3 import ESM3
from esm.pretrained import (
    ESM3_function_decoder_v0,
    ESM3_structure_decoder_v0,
    ESM3_structure_encoder_v0,
)
from esm.tokenization import get_esm3_model_tokenizers
from esm.utils.constants.models import ESM3_OPEN_SMALL

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from .config import SFTConfig, load_sft_config
    from .masking import (
        mask_antibody_seq_cdr,
        mask_antibody_stru_cdr,
        mask_seq_single_cdr,
        mask_stru_single_cdr,
        random_mask_antibody_seq_cdr_tokens,
        random_mask_antibody_seq_single_cdr_tokens,
        random_mask_antibody_seq_tokens,
        random_mask_antibody_stru_cdr_token,
        random_mask_antibody_stru_single_cdr_token,
        random_mask_antibody_stru_token,
    )
except ImportError:  # pragma: no cover - fallback for script usage
    from sft.training.config import SFTConfig, load_sft_config
    from sft.training.masking import (
        mask_antibody_seq_cdr,
        mask_antibody_stru_cdr,
        mask_seq_single_cdr,
        mask_stru_single_cdr,
        random_mask_antibody_seq_cdr_tokens,
        random_mask_antibody_seq_single_cdr_tokens,
        random_mask_antibody_seq_tokens,
        random_mask_antibody_stru_cdr_token,
        random_mask_antibody_stru_single_cdr_token,
        random_mask_antibody_stru_token,
    )

try:
    from sft.data_pipeline.sabdab_data_imgt import ProteinDataset_antigen
except ImportError:  # pragma: no cover - fallback for local execution
    from sft.data_pipeline.sabdab_data_imgt import ProteinDataset_antigen


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_wandb(config: SFTConfig) -> Optional[str]:
    if not config.wandb.wandb_enabled:
        return
    if wandb is None:
        raise ImportError("wandb is not installed but wandb_enabled is True.")
    init_kwargs = {"project": config.wandb.wandb_project}
    exp_name = None
    if config.wandb.wandb_name:
        exp_name = (
            f"{config.wandb.wandb_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        init_kwargs["name"] = exp_name
    if config.wandb.wandb_dir:
        init_kwargs["dir"] = config.wandb.wandb_dir
    if config.wandb.wandb_entity:
        init_kwargs["entity"] = config.wandb.wandb_entity
    init_kwargs["config"] = config.to_dict()
    wandb.init(**init_kwargs)
    return exp_name


def _freeze_output_heads(model: nn.Module) -> None:
    patterns_to_freeze = [
        "output_heads.ss8_head",
        "output_heads.sasa_head",
        "output_heads.function_head",
        "output_heads.residue_head",
    ]
    for name, param in model.named_parameters():
        for pattern in patterns_to_freeze:
            if pattern in name:
                param.requires_grad = False


def _resolve_adapter_module(adapter_module: Optional[str]):
    module_path = adapter_module or "sft.training.adapter_gear"
    return import_module(module_path)


def build_model(config: SFTConfig) -> nn.Module:
    if not config.model.model_path:
        raise ValueError("model.model_path must be set explicitly.")
    base_model = ESM3(
        d_model=1536,
        n_heads=24,
        v_heads=256,
        n_layers=48,
        structure_encoder_fn=ESM3_structure_encoder_v0,
        structure_decoder_fn=ESM3_structure_decoder_v0,
        function_decoder_fn=ESM3_function_decoder_v0,
        tokenizers=get_esm3_model_tokenizers(ESM3_OPEN_SMALL),
    )

    state_dict = torch.load(config.model.model_path, map_location="cpu")
    base_model.load_state_dict(state_dict, strict=True)

    adapter_module = _resolve_adapter_module(config.model.adapter_module)
    wrapper_name = config.model.adapter_wrapper or "ESM3Wrapper"
    wrapper_cls = getattr(adapter_module, wrapper_name, None)
    if wrapper_cls is None:
        model = base_model
    else:
        model = wrapper_cls(base_model)
        # Only load pretrained weights for components that exist in state_dict
        # The Modified_transformer with adapter will be loaded via strict=False
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if config.model.freeze_output_heads:
        _freeze_output_heads(model)

    if config.model.adapter_state_path:
        adapter_state = torch.load(config.model.adapter_state_path, map_location="cpu")
        model.load_state_dict(adapter_state, strict=False)

    return model


def _freeze_adapter_only(model: nn.Module, adapter_module) -> None:
    has_adapter = any("adapter" in name for name, _ in model.named_parameters())
    if not has_adapter:
        return
    freeze_fn = None
    if adapter_module is not None:
        freeze_fn = getattr(adapter_module, "freeze_model_except_adapter", None)
    if freeze_fn is None:
        for name, param in model.named_parameters():
            param.requires_grad = "adapter" in name
    else:
        freeze_fn(model)


def _split_param_groups(model: nn.Module, adapter_keyword: str):
    backbone_params = []
    adapter_params = []
    for name, param in model.named_parameters():
        if adapter_keyword in name:
            adapter_params.append(param)
        else:
            backbone_params.append(param)
    return backbone_params, adapter_params


def _build_staged_optimizer(
    model: nn.Module,
    *,
    adapter_keyword: str,
    backbone_lr: float,
    adapter_lr: float,
    weight_decay: float,
    betas: tuple[float, float],
):
    backbone_params, adapter_params = _split_param_groups(model, adapter_keyword)
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": adapter_params, "lr": adapter_lr},
        ],
        weight_decay=weight_decay,
        betas=betas,
    )


def _activate_full_training(
    model: nn.Module,
    optimizer,
    scheduler,
    *,
    freeze_output_heads: bool,
):
    for param in model.parameters():
        param.requires_grad = True
    if freeze_output_heads:
        _freeze_output_heads(model)
    return optimizer, scheduler


def _build_scheduler(
    optimizer, num_training_steps: int, warmup_ratio: float, scheduler_name: str
):
    warmup_steps = int(num_training_steps * warmup_ratio) if warmup_ratio else 0
    if scheduler_name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    if scheduler_name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def _load_antigen_features(path: Optional[str], id_key: str) -> Optional[dict]:
    if not path:
        return None
    data = torch.load(path, map_location="cpu")
    items = data.values() if isinstance(data, dict) else data
    features = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        if id_key not in item or "node_feature" not in item:
            continue
        features[item[id_key]] = item["node_feature"]
    return features or None


def _build_collate_fn(antigen_features: Optional[dict]):
    def collate_fn(batch):
        ids = [item["id"] for item in batch]
        out = {
            "id": ids,
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "structure_tokens": torch.stack(
                [item["structure_tokens"] for item in batch]
            ),
            "chain_id": torch.stack([item["chain_id"] for item in batch]),
            "cdr_pos": torch.stack([item["cdr_pos"] for item in batch]),
            "H_chain": torch.tensor(
                [item["H_chain"] for item in batch], dtype=torch.long
            ),
            "L_chain": torch.tensor(
                [item["L_chain"] for item in batch], dtype=torch.long
            ),
        }

        if "interface" in batch[0]:
            out["interface"] = torch.stack([item["interface"] for item in batch])

        if antigen_features:
            feats = []
            for data_id in ids:
                feat = antigen_features.get(data_id)
                if feat is None:
                    feats.append(None)
                else:
                    feats.append(torch.as_tensor(feat, dtype=torch.float))
            valid = [f for f in feats if f is not None]
            if valid:
                max_len = max(f.shape[0] for f in valid)
                feat_dim = valid[0].shape[-1]
                padded = []
                for f in feats:
                    if f is None:
                        padded.append(
                            torch.zeros((max_len, feat_dim), dtype=valid[0].dtype)
                        )
                    elif f.shape[0] < max_len:
                        pad = torch.zeros(
                            (max_len - f.shape[0], feat_dim), dtype=f.dtype, device=f.device
                        )
                        padded.append(torch.cat([f, pad], dim=0))
                    else:
                        padded.append(f[:max_len])
                out["antigen_features"] = torch.stack(padded)
            else:
                out["antigen_features"] = None
        else:
            out["antigen_features"] = None
        return out

    return collate_fn


def _build_reconstruction_labels(
    targets: torch.Tensor,
    masked_inputs: torch.Tensor,
    *,
    ignore_index: int,
):
    masked_positions = masked_inputs.ne(targets)
    labels = torch.full_like(targets, ignore_index)
    labels[masked_positions] = targets[masked_positions]
    return labels, masked_positions


def _masked_reconstruction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masked_inputs: torch.Tensor,
    criterion,
    *,
    ignore_index: int,
):
    labels, masked_positions = _build_reconstruction_labels(
        targets, masked_inputs, ignore_index=ignore_index
    )
    if masked_positions.any().item():
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    else:
        loss = logits.sum() * 0.0
    return loss, masked_positions.sum()


def _resolve_mode_settings(config: SFTConfig, mode: str):
    defaults = {}
    if mode == "imgt":
        defaults = {
            "strategy": "mixed",
            "mask_structure": True,
            "use_structure_loss": True,
            "unfreeze_epoch": 40,
            "single_cdr_index": config.masking.single_cdr_index,
        }
    elif mode == "imgt_if":
        defaults = {
            "strategy": "mixed",
            "mask_structure": False,
            "use_structure_loss": False,
            "unfreeze_epoch": 20,
            "single_cdr_index": config.masking.single_cdr_index,
        }
    elif mode == "imgt_h3":
        defaults = {
            "strategy": "single_cdr",
            "mask_structure": True,
            "use_structure_loss": True,
            "unfreeze_epoch": 20,
            "single_cdr_index": 3,
        }
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return {
        "strategy": defaults["strategy"],
        "mask_structure": defaults["mask_structure"],
        "use_structure_loss": defaults["use_structure_loss"],
        "unfreeze_epoch": config.training.unfreeze_epoch
        if config.training.unfreeze_epoch is not None
        else defaults["unfreeze_epoch"],
        "single_cdr_index": config.masking.single_cdr_index
        if config.masking.single_cdr_index is not None
        else defaults["single_cdr_index"],
    }


def _apply_training_masks(batch, config: SFTConfig, mode_settings: dict):
    tokens = batch["input_ids"]
    struct_tokens = batch["structure_tokens"]
    chain_id = batch["chain_id"]
    cdr_pos = batch["cdr_pos"]
    h_chain = batch["H_chain"]
    l_chain = batch["L_chain"]

    mask_idx = config.tokens.mask_token_id
    pad_idx = config.tokens.pad_token_id
    chain_idx = config.tokens.chain_token_id
    stru_mask_idx = config.tokens.structure_mask_id
    special_tokens = config.tokens.structure_special_tokens

    strategy = mode_settings["strategy"]
    if strategy == "mixed":
        weights = config.masking.mixed_weights
        if weights:
            total = sum(weights)
            weights = [w / total for w in weights]
        else:
            weights = [0.25, 0.25, 0.5]
        r = random.random()
        if r < weights[0]:
            masked_seq = random_mask_antibody_seq_tokens(
                tokens,
                mask_idx,
                pad_idx,
                chain_id,
                h_chain,
                l_chain,
                chain_idx,
                random_mask_prob=config.masking.random_mask_prob,
                random_beta=config.masking.random_beta,
            )
            masked_stru = (
                random_mask_antibody_stru_token(
                    struct_tokens,
                    stru_mask_idx,
                    special_tokens,
                    chain_id,
                    h_chain,
                    l_chain,
                )
                if mode_settings["mask_structure"]
                else struct_tokens
            )
        elif r < weights[0] + weights[1]:
            masked_seq = random_mask_antibody_seq_cdr_tokens(
                tokens,
                cdr_pos,
                mask_idx,
                pad_idx,
                chain_id,
                h_chain,
                l_chain,
                chain_idx,
                random_mask_prob=config.masking.random_mask_prob,
                random_beta=config.masking.random_beta,
            )
            masked_stru = (
                random_mask_antibody_stru_cdr_token(
                    struct_tokens,
                    cdr_pos,
                    stru_mask_idx,
                    special_tokens,
                    chain_id,
                    h_chain,
                    l_chain,
                )
                if mode_settings["mask_structure"]
                else struct_tokens
            )
        else:
            masked_seq = mask_antibody_seq_cdr(
                tokens,
                cdr_pos,
                mask_idx,
                pad_idx,
                chain_id,
                h_chain,
                l_chain,
                chain_idx,
            )
            masked_stru = (
                mask_antibody_stru_cdr(
                    struct_tokens,
                    cdr_pos,
                    stru_mask_idx,
                    special_tokens,
                    chain_id,
                    h_chain,
                    l_chain,
                )
                if mode_settings["mask_structure"]
                else struct_tokens
            )
    elif strategy == "single_cdr":
        cdr_index = mode_settings["single_cdr_index"]
        if cdr_index is None:
            raise ValueError("single_cdr_index must be set for single_cdr mode.")
        if random.random() < config.masking.random_single_cdr_prob:
            masked_seq = random_mask_antibody_seq_single_cdr_tokens(
                tokens,
                cdr_pos,
                cdr_index,
                mask_idx,
                pad_idx,
                chain_id,
                h_chain,
                l_chain,
                chain_idx,
                random_mask_prob=config.masking.random_mask_prob,
                random_beta=config.masking.random_beta,
            )
            masked_stru = (
                random_mask_antibody_stru_single_cdr_token(
                    struct_tokens,
                    cdr_pos,
                    cdr_index,
                    stru_mask_idx,
                    special_tokens,
                    chain_id,
                    h_chain,
                    l_chain,
                )
                if mode_settings["mask_structure"]
                else struct_tokens
            )
        else:
            masked_seq = mask_seq_single_cdr(
                tokens,
                cdr_pos,
                cdr_index,
                mask_idx,
                pad_idx,
                chain_id,
                h_chain,
                l_chain,
                chain_idx,
            )
            masked_stru = (
                mask_stru_single_cdr(
                    struct_tokens,
                    cdr_pos,
                    cdr_index,
                    stru_mask_idx,
                    special_tokens,
                    chain_id,
                    h_chain,
                    l_chain,
                )
                if mode_settings["mask_structure"]
                else struct_tokens
            )
    else:
        raise ValueError(f"Unsupported masking strategy: {strategy}")

    return masked_seq, masked_stru


def _apply_eval_masks(batch, config: SFTConfig, mode_settings: dict):
    tokens = batch["input_ids"]
    struct_tokens = batch["structure_tokens"]
    chain_id = batch["chain_id"]
    cdr_pos = batch["cdr_pos"]
    h_chain = batch["H_chain"]
    l_chain = batch["L_chain"]

    mask_idx = config.tokens.mask_token_id
    pad_idx = config.tokens.pad_token_id
    chain_idx = config.tokens.chain_token_id
    stru_mask_idx = config.tokens.structure_mask_id
    special_tokens = config.tokens.structure_special_tokens

    strategy = mode_settings["strategy"]
    if strategy == "single_cdr":
        cdr_index = mode_settings["single_cdr_index"]
        masked_seq = mask_seq_single_cdr(
            tokens,
            cdr_pos,
            cdr_index,
            mask_idx,
            pad_idx,
            chain_id,
            h_chain,
            l_chain,
            chain_idx,
        )
        masked_stru = (
            mask_stru_single_cdr(
                struct_tokens,
                cdr_pos,
                cdr_index,
                stru_mask_idx,
                special_tokens,
                chain_id,
                h_chain,
                l_chain,
            )
            if mode_settings["mask_structure"]
            else struct_tokens
        )
    else:
        masked_seq = mask_antibody_seq_cdr(
            tokens, cdr_pos, mask_idx, pad_idx, chain_id, h_chain, l_chain, chain_idx
        )
        masked_stru = (
            # mask CDR & Chain
            mask_antibody_stru_cdr(
                struct_tokens,
                cdr_pos,
                stru_mask_idx,
                special_tokens,
                chain_id,
                h_chain,
                l_chain,
            )
            if mode_settings["mask_structure"]
            else struct_tokens
        )
    return masked_seq, masked_stru


def evaluate_perplexity(
    model,
    dataloader,
    accelerator,
    config: SFTConfig,
    mode_settings: dict,
    criterion_seq,
    criterion_stru,
):
    previous_training_mode = model.training
    model.eval()
    total_loss_seq = torch.tensor(0.0, device=accelerator.device)
    total_loss_stru = torch.tensor(0.0, device=accelerator.device)
    num_seq_tokens = torch.tensor(0.0, device=accelerator.device)
    num_stru_tokens = torch.tensor(0.0, device=accelerator.device)

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            batch = {
                k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            masked_seq, masked_stru = _apply_eval_masks(batch, config, mode_settings)
            interface = batch.get("interface")
            if interface is None:
                interface = torch.zeros_like(batch["input_ids"])
            outputs = model(
                sequence_tokens=masked_seq,
                structure_tokens=masked_stru,
                interface=interface,
                chain_id=batch["chain_id"],
                antigen_feat=batch.get("antigen_features"),
            )
            loss_seq, seq_token_count = _masked_reconstruction_loss(
                outputs.sequence_logits,
                batch["input_ids"],
                masked_seq,
                criterion_seq,
                ignore_index=config.tokens.pad_token_id,
            )
            total_loss_seq += loss_seq * seq_token_count
            num_seq_tokens += seq_token_count

            if mode_settings["use_structure_loss"]:
                loss_stru, stru_token_count = _masked_reconstruction_loss(
                    outputs.structure_logits,
                    batch["structure_tokens"],
                    masked_stru,
                    criterion_stru,
                    ignore_index=-100,
                )
                total_loss_stru += loss_stru * stru_token_count
                num_stru_tokens += stru_token_count

    stats = torch.stack(
        [total_loss_seq, total_loss_stru, num_seq_tokens, num_stru_tokens]
    ).to(accelerator.device)
    stats = accelerator.reduce(stats, reduction="sum")
    global_loss_seq = stats[0]
    global_loss_stru = stats[1]
    global_seq_tokens = stats[2]
    global_stru_tokens = stats[3]

    model.train(previous_training_mode)

    perplexity_seq = (
        torch.exp(global_loss_seq / global_seq_tokens).item()
        if global_seq_tokens > 0
        else float("inf")
    )
    perplexity_stru = (
        torch.exp(global_loss_stru / global_stru_tokens).item()
        if mode_settings["use_structure_loss"] and global_stru_tokens > 0
        else float("inf")
        if mode_settings["use_structure_loss"]
        else float("nan")
    )
    return perplexity_seq, perplexity_stru


def train(
    config: Optional[SFTConfig] = None, mode_override: Optional[str] = None
) -> None:
    config = config or load_sft_config()
    mode = mode_override or config.training.mode
    mode_settings = _resolve_mode_settings(config, mode)

    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.accumulation_steps,
    )

    _set_seed(config.common.seed)
    exp_name = None
    if accelerator.is_main_process:
        exp_name = _init_wandb(config)
    accelerator.wait_for_everyone()

    # Load the externally supplied antigen-feature cache.
    antigen_features = _load_antigen_features(
        config.data.antigen_feature_path, config.data.antigen_feature_id_key
    )
    collate_fn = _build_collate_fn(antigen_features)

    train_dataset = ProteinDataset_antigen(
        max_length=config.data.max_length,
        summary_path=config.data.summary_path,
        imgt_dir=config.data.imgt_dir,
        processed_dir=config.data.processed_dir,
        init=config.data.init,
        split=config.data.train_split,
    )
    val_dataset = ProteinDataset_antigen(
        max_length=config.data.max_length,
        summary_path=config.data.summary_path,
        imgt_dir=config.data.imgt_dir,
        processed_dir=config.data.processed_dir,
        init=False,
        split=config.data.val_split,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = build_model(config)
    adapter_module = _resolve_adapter_module(config.model.adapter_module)
    _freeze_adapter_only(model, adapter_module)

    backbone_lr = config.training.backbone_lr or config.training.lr
    adapter_lr = config.training.adapter_lr or config.training.lr
    optimizer = _build_staged_optimizer(
        model,
        adapter_keyword=config.model.adapter_param_keyword,
        backbone_lr=backbone_lr,
        adapter_lr=adapter_lr,
        weight_decay=config.training.weight_decay,
        betas=config.training.betas if config.training.betas is not None else (0.9, 0.95),
    )

    steps_per_epoch = math.ceil(len(train_loader) / config.training.accumulation_steps)
    num_training_steps = steps_per_epoch * config.training.epochs
    scheduler = _build_scheduler(
        optimizer,
        num_training_steps,
        config.training.warmup_ratio,
        config.training.scheduler,
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    criterion_seq = nn.CrossEntropyLoss(ignore_index=config.tokens.pad_token_id)
    criterion_stru = nn.CrossEntropyLoss(ignore_index=-100)

    pre_seq_ppl, pre_stru_ppl = evaluate_perplexity(
        model,
        val_loader,
        accelerator,
        config,
        mode_settings,
        criterion_seq,
        criterion_stru,
    )

    if accelerator.is_main_process:
        print(
            f"Pre-trained Model - Val Perplexity sequence: {pre_seq_ppl:.4f}, "
            f"Val Perplexity structure: {pre_stru_ppl:.4f}"
        )
        if wandb is not None and config.wandb.wandb_enabled:
            wandb.log(
                {
                    "pre_trained_val_perplexity": pre_seq_ppl,
                    "pre_trained_val_perplexity_stru": pre_stru_ppl,
                }
            )

    accelerator.wait_for_everyone()

    global_step = 0
    for epoch in range(config.training.epochs):
        if (
            mode_settings["unfreeze_epoch"] is not None
            and epoch == mode_settings["unfreeze_epoch"]
        ):
            optimizer, scheduler = _activate_full_training(
                model,
                optimizer,
                scheduler,
                freeze_output_heads=config.model.freeze_output_heads,
            )
            if (
                accelerator.is_main_process
                and wandb is not None
                and config.wandb.wandb_enabled
            ):
                wandb.log({"backbone_lr": backbone_lr, "adapter_lr": adapter_lr})

        model.train()
        for batch in tqdm(train_loader, disable=not accelerator.is_local_main_process):
            batch = {
                k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with accelerator.accumulate(model):
                masked_seq, masked_stru = _apply_training_masks(
                    batch, config, mode_settings
                )
                interface = batch.get("interface")
                if interface is None:
                    interface = torch.zeros_like(batch["input_ids"])

                outputs = model(
                    sequence_tokens=masked_seq,
                    structure_tokens=masked_stru,
                    interface=interface,
                    chain_id=batch["chain_id"],
                    antigen_feat=batch.get("antigen_features"),
                )

                loss_seq, _ = _masked_reconstruction_loss(
                    outputs.sequence_logits,
                    batch["input_ids"],
                    masked_seq,
                    criterion_seq,
                    ignore_index=config.tokens.pad_token_id,
                )

                loss = loss_seq
                loss_stru = torch.tensor(0.0, device=accelerator.device)
                if mode_settings["use_structure_loss"]:
                    loss_stru, _ = _masked_reconstruction_loss(
                        outputs.structure_logits,
                        batch["structure_tokens"],
                        masked_stru,
                        criterion_stru,
                        ignore_index=-100,
                    )
                    loss = loss_seq + config.training.structure_loss_weight * loss_stru

                accelerator.backward(loss)
                # print("[Debug] Detecting Unused Parameters (Grad is None):")

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), max_norm=config.training.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    avg_loss = accelerator.gather(loss).mean().item()
                    avg_loss_seq = accelerator.gather(loss_seq).mean().item()
                    avg_loss_stru = accelerator.gather(loss_stru).mean().item()
                    if (
                        accelerator.is_main_process
                        and wandb is not None
                        and config.wandb.wandb_enabled
                    ):
                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/loss_seq": avg_loss_seq,
                                "train/loss_stru": avg_loss_stru,
                                "train/epoch": epoch,
                                "train/global_step": global_step,
                                "train/lr": scheduler.get_last_lr()[0],
                            }
                        )

                    if (
                        config.eval.eval_every_steps > 0
                        and global_step % config.eval.eval_every_steps == 0
                    ):
                        seq_ppl, stru_ppl = evaluate_perplexity(
                            model,
                            val_loader,
                            accelerator,
                            config,
                            mode_settings,
                            criterion_seq,
                            criterion_stru,
                        )
                        if (
                            accelerator.is_main_process
                            and wandb is not None
                            and config.wandb.wandb_enabled
                        ):
                            wandb.log(
                                {
                                    "intermediate_val_perplexity": seq_ppl,
                                    "intermediate_val_perplexity_stru": stru_ppl,
                                }
                            )

                    if (
                        config.eval.every_save_steps > 0
                        and global_step % config.eval.every_save_steps == 0
                    ):
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            save_path = os.path.join(
                                config.eval.output_dir,
                                f"{exp_name or 'sft'}/checkpoint_SFT_step_{global_step}_epoch_{epoch}_{timestamp_str}.pt",
                            )
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            torch.save(
                                accelerator.unwrap_model(model).state_dict(), save_path
                            )
                            print(f"Saved model to {save_path}")
                        accelerator.wait_for_everyone()

        seq_ppl, stru_ppl = evaluate_perplexity(
            model,
            val_loader,
            accelerator,
            config,
            mode_settings,
            criterion_seq,
            criterion_stru,
        )
        if accelerator.is_main_process:
            print(
                f"Epoch {epoch + 1} - Val seq perplexity: {seq_ppl:.4f}, "
                f"Val structure perplexity: {stru_ppl:.4f}"
            )
            if wandb is not None and config.wandb.wandb_enabled:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "val_seq_perplexity": seq_ppl,
                        "val_stru_perplexity": stru_ppl,
                    }
                )

    if accelerator.is_main_process:
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(
            config.eval.output_dir,
            f"{exp_name or 'sft'}/checkpoint_SFT_{timestamp_str}.pt",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
        print(f"Saved model to {save_path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SFT trainer (imgt-first)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config. Defaults to sft/config/SFT.yaml",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["imgt", "imgt_if", "imgt_h3"],
        help="Override training mode (default: config.training.mode).",
    )
    args = parser.parse_args(argv)

    config = load_sft_config(args.config)
    train(config, mode_override=args.mode)


if __name__ == "__main__":
    # """
    # fork shared environment
    # can not fork the original cuda environment handle
    # need to use spawn to re-initialize the environment
    # """

    main()
