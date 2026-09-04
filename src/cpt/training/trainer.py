from __future__ import annotations

import math
import os
import random
import sys
from datetime import datetime
from typing import Optional, Sequence

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset, random_split
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


from cpt.data_pipeline.npz_dataset import NPZAntibodyDataset
from cpt.training.masking import (
    mask_antibody_seq_cdr,
    mask_antibody_stru_cdr,
    mask_stru_tokens,
    mask_tokens,
)

try:
    from .config import FullFinetuneConfig, load_full_finetune_config
except ImportError:  # pragma: no cover - fallback for script usage
    from cpt.training.config import FullFinetuneConfig, load_full_finetune_config


def build_model(model_path: Optional[str] = None) -> ESM3:
    if not model_path:
        raise ValueError("model.model_path must be set explicitly.")
    tokenizers = get_esm3_model_tokenizers(ESM3_OPEN_SMALL)
    _ensure_gap_token(tokenizers)
    model = ESM3(
        d_model=1536,
        n_heads=24,
        v_heads=256,
        n_layers=48,
        structure_encoder_fn=ESM3_structure_encoder_v0,
        structure_decoder_fn=ESM3_structure_decoder_v0,
        function_decoder_fn=ESM3_function_decoder_v0,
        tokenizers=tokenizers,
    )
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    _resize_sequence_vocab(model, len(tokenizers.sequence))
    for param in model.parameters():
        param.requires_grad = True

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


    return model


def _ensure_gap_token(tokenizers, gap_token: str = "[GAP]") -> None:
    sequence_tokenizer = tokenizers.sequence
    vocab = sequence_tokenizer.get_vocab()
    if gap_token in vocab:
        return
    sequence_tokenizer.add_special_tokens(
        {"additional_special_tokens": [gap_token]}
    )


def _resize_sequence_vocab(model: ESM3, new_vocab_size: int) -> None:
    seq_embed = model.encoder.sequence_embed
    if new_vocab_size <= seq_embed.num_embeddings:
        return
    old_weight = seq_embed.weight.data
    new_embed = nn.Embedding(new_vocab_size, seq_embed.embedding_dim).to(
        old_weight.device, old_weight.dtype
    )
    new_embed.weight.data[: old_weight.shape[0]] = old_weight
    std = float(old_weight.std().item())
    if std == 0.0:
        new_embed.weight.data[old_weight.shape[0] :].zero_()
    else:
        nn.init.normal_(new_embed.weight.data[old_weight.shape[0] :], mean=0.0, std=std)
    model.encoder.sequence_embed = new_embed

    seq_head = model.output_heads.sequence_head
    last_linear = seq_head[-1]
    new_linear = nn.Linear(last_linear.in_features, new_vocab_size).to(
        last_linear.weight.device, last_linear.weight.dtype
    )
    new_linear.weight.data[: last_linear.out_features] = last_linear.weight.data
    new_linear.bias.data[: last_linear.out_features] = last_linear.bias.data
    if std == 0.0:
        new_linear.weight.data[last_linear.out_features :].zero_()
        new_linear.bias.data[last_linear.out_features :].zero_()
    else:
        nn.init.normal_(
            new_linear.weight.data[last_linear.out_features :], mean=0.0, std=std
        )
        new_linear.bias.data[last_linear.out_features :].zero_()
    seq_head[-1] = new_linear


def get_warmup_stable_cosine_scheduler(
    optimizer,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        if current_step < num_warmup_steps + num_stable_steps:
            return 1.0

        if current_step < num_warmup_steps + num_stable_steps + num_decay_steps:
            progress = float(
                current_step - num_warmup_steps - num_stable_steps
            ) / float(max(1, num_decay_steps))

            cosine_val = 0.5 * (
                1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)
            )

            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val

        return min_lr_ratio

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def _split_dataset(dataset, train_ratio: float, seed: Optional[int]):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    indices = list(range(len(dataset)))
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_split, val_split = random_split(
        indices, [train_size, val_size], generator=generator
    )
    train_indices = (
        train_split.indices if isinstance(train_split, Subset) else train_split
    )
    val_indices = val_split.indices if isinstance(val_split, Subset) else val_split
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _init_wandb(config: FullFinetuneConfig) -> None:
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


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UncertaintyWeighting(nn.Module):
    """
    Learn task weights via homoscedastic uncertainty.
    """

    def __init__(self, n_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        total = losses[0].new_zeros(())
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + 0.5 * precision * loss + 0.5 * self.log_vars[i]
        return total

    def get_weights(self) -> torch.Tensor:
        return 0.5 * torch.exp(-self.log_vars).detach()


def evaluate_perplexity(
    model,
    dataloader,
    accelerator,
    criterion_seq,
    criterion_stru,
    mask_idx,
    pad_idx,
    chain_idx,
    special_tokens,
):
    previous_training_mode = model.training
    model.eval()
    total_loss_seq = torch.tensor(0.0, device=accelerator.device)
    total_loss_stru = torch.tensor(0.0, device=accelerator.device)
    num_seq_tokens = torch.tensor(0.0, device=accelerator.device)
    num_stru_tokens = torch.tensor(0.0, device=accelerator.device)

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            masked_tokens, seq_mask = mask_tokens(
                batch["input_ids"], mask_idx, pad_idx, chain_idx, return_mask=True
            )
            masked_stru_tokens, stru_mask = mask_stru_tokens(
                batch["structure_tokens"], 4096, special_tokens, return_mask=True
            )

            # Forward
            outputs = model(
                sequence_tokens=masked_tokens, structure_tokens=masked_stru_tokens
            )

            seq_token_count = seq_mask.sum()
            if seq_token_count.item() > 0:
                seq_labels = build_masked_labels(batch["input_ids"], seq_mask, pad_idx)
                loss_seq = criterion_seq(
                    outputs.sequence_logits.view(-1, outputs.sequence_logits.size(-1)),
                    seq_labels.view(-1),
                )
                total_loss_seq += loss_seq * seq_token_count
                num_seq_tokens += seq_token_count

            stru_token_count = stru_mask.sum()
            if stru_token_count.item() > 0:
                stru_labels = build_masked_labels(batch["structure_tokens"], stru_mask, -100)
                loss_stru = criterion_stru(
                    outputs.structure_logits.view(-1, outputs.structure_logits.size(-1)),
                    stru_labels.view(-1),
                )
                total_loss_stru += loss_stru * stru_token_count
                num_stru_tokens += stru_token_count

    stats = torch.stack([total_loss_seq, total_loss_stru, num_seq_tokens, num_stru_tokens]).to(
        accelerator.device
    )
    stats = accelerator.reduce(stats, reduction="sum")

    global_loss_seq = stats[0]
    global_loss_stru = stats[1]
    global_seq_tokens = stats[2]
    global_stru_tokens = stats[3]

    if global_seq_tokens.item() == 0:
        perplexity_seq = float("inf")
    else:
        perplexity_seq = torch.exp(global_loss_seq / global_seq_tokens).item()

    if global_stru_tokens.item() == 0:
        perplexity_stru = float("inf")
    else:
        perplexity_stru = torch.exp(global_loss_stru / global_stru_tokens).item()

    model.train(previous_training_mode)
    return perplexity_seq, perplexity_stru


def build_masked_labels(targets, masked_positions, ignore_index):
    labels = torch.full_like(targets, ignore_index)
    labels[masked_positions] = targets[masked_positions]
    return labels


def train(config: Optional[FullFinetuneConfig] = None) -> None:
    config = config or load_full_finetune_config()


    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.accumulation_steps,
    )



    _set_seed(config.common.seed)
    if accelerator.is_main_process:
        exp_name = _init_wandb(config)

    accelerator.wait_for_everyone()

    dataset = NPZAntibodyDataset(
        npz_paths=config.data.npz_paths, key=config.data.npz_key
    )
    if config.data.npz_paths_eval:
        train_dataset = dataset
        val_dataset = NPZAntibodyDataset(
            npz_paths=config.data.npz_paths_eval, key=config.data.npz_key
        )
    else:
        train_dataset, val_dataset = _split_dataset(
            dataset, config.data.train_ratio, config.common.seed
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    model = build_model(config.model.model_path)
    criterion_seq = nn.CrossEntropyLoss(ignore_index=1)
    criterion_stru = nn.CrossEntropyLoss(ignore_index=-100)
    uncertainty_weighting = UncertaintyWeighting(n_tasks=2)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(uncertainty_weighting.parameters()),
        lr=config.training.lr_max,
        weight_decay=config.training.weight_decay,
        betas=config.training.betas
        if config.training.betas is not None
        else (0.9, 0.95),
    )

    model, uncertainty_weighting, optimizer, train_loader, val_loader = accelerator.prepare(
        model, uncertainty_weighting, optimizer, train_loader, val_loader
    )

    num_training_steps = len(train_loader) * config.training.scheduler_epochs

    if config.training.warmup_radio is not None:
        config.training.warmup_steps = int(
            num_training_steps * config.training.warmup_radio
        )

    if config.training.decay_ratio is not None:
        config.training.decay_steps = int(
            num_training_steps * config.training.decay_ratio
        )

    stable_steps = (
        num_training_steps - config.training.decay_steps - config.training.warmup_steps
    )

    stable_steps = max(0, stable_steps)

    min_lr_ratio = config.training.lr_min / config.training.lr_max

    scheduler = get_warmup_stable_cosine_scheduler(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_stable_steps=stable_steps,
        num_decay_steps=config.training.decay_steps,
        min_lr_ratio=min_lr_ratio,
    )

    mask_idx = 32
    pad_idx = 1
    chain_idx = 31
    special_tokens = [4096, 4097, 4098, 4099, 4100]

    pre_seq_ppl, pre_stru_ppl = evaluate_perplexity(
        model,
        val_loader,
        accelerator,
        criterion_seq,
        criterion_stru,
        mask_idx,
        pad_idx,
        chain_idx,
        special_tokens,
    )

    if accelerator.is_main_process:
        print(
            "Pre-trained Model - Val Perplexity sequence: "
            f"{pre_seq_ppl:.4f}, Val Perplexity structure: {pre_stru_ppl:.4f}"
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
        model.train()
        for batch in tqdm(train_loader):
            with accelerator.accumulate(model):
                use_cdr_mask = random.random() >= 0.7
                if not use_cdr_mask:
                    masked_tokens_seq, seq_mask = mask_tokens(
                        batch["input_ids"], mask_idx, pad_idx, chain_idx, return_mask=True
                    )
                    masked_stru_tokens, stru_mask = mask_stru_tokens(
                        batch["structure_tokens"], 4096, special_tokens, return_mask=True
                    )
                else:
                    masked_tokens_seq, seq_mask = mask_antibody_seq_cdr(
                        batch["input_ids"],
                        batch["cdr_pos"],
                        mask_idx,
                        pad_idx,
                        batch["chain_id"],
                        batch["H_chain"],
                        batch["L_chain"],
                        chain_idx,
                        return_mask=True,
                    )
                    masked_stru_tokens, stru_mask = mask_antibody_stru_cdr(
                        batch["structure_tokens"],
                        batch["cdr_pos"],
                        4096,
                        special_tokens,
                        batch["chain_id"],
                        batch["H_chain"],
                        batch["L_chain"],
                        return_mask=True,
                    )

                outputs = model(
                    sequence_tokens=masked_tokens_seq,
                    structure_tokens=masked_stru_tokens,
                )
                logits_seq = outputs.sequence_logits
                stru_logits = outputs.structure_logits
                seq_labels = build_masked_labels(batch["input_ids"], seq_mask, pad_idx)
                stru_labels = build_masked_labels(batch["structure_tokens"], stru_mask, -100)

                if seq_mask.any().item():
                    loss_seq = criterion_seq(
                        logits_seq.view(-1, logits_seq.size(-1)),
                        seq_labels.view(-1),
                    )
                else:
                    loss_seq = logits_seq.new_zeros(())

                if stru_mask.any().item():
                    loss_stru = criterion_stru(
                        stru_logits.view(-1, stru_logits.size(-1)),
                        stru_labels.view(-1),
                    )
                else:
                    loss_stru = stru_logits.new_zeros(())

                loss = uncertainty_weighting([loss_seq, loss_stru])

                accelerator.backward(loss)

                #     print("[Debug] Detecting Unused Parameters (Grad is None):")


                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    avg_loss = accelerator.gather(loss).mean().item()
                    avg_loss_seq = accelerator.gather(loss_seq).mean().item()
                    avg_loss_stru = accelerator.gather(loss_stru).mean().item()
                    uw_weights = accelerator.unwrap_model(
                        uncertainty_weighting
                    ).get_weights()
                    seq_weight = float(uw_weights[0].item())
                    stru_weight = float(uw_weights[1].item())
                    if accelerator.is_main_process and config.wandb.wandb_enabled:
                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/loss_seq": avg_loss_seq,
                                "train/loss_stru": avg_loss_stru,
                                "train/weight_seq": seq_weight,
                                "train/weight_stru": stru_weight,
                                "train/epoch": epoch,
                                "train/global_step": global_step,
                                "train/lr": scheduler.get_last_lr()[0],
                            }
                        )

                    if (
                        config.eval.eval_every_steps > 0
                        and global_step % config.eval.eval_every_steps == 0
                    ):
                        pre_seq_ppl, pre_stru_ppl = evaluate_perplexity(
                            model,
                            val_loader,
                            accelerator,
                            criterion_seq,
                            criterion_stru,
                            mask_idx,
                            pad_idx,
                            chain_idx,
                            special_tokens,
                        )

                        if accelerator.is_main_process:
                            print(f"Running evaluation at step {global_step}...")
                            if config.wandb.wandb_enabled:
                                wandb.log(
                                    {
                                        "eval/intermediate_val_perplexity": pre_seq_ppl,
                                        "eval/intermediate_val_perplexity_stru": pre_stru_ppl,
                                    }
                                )

                    if (
                        config.eval.every_save_steps > 0
                        and global_step % config.eval.every_save_steps == 0
                    ):
                        accelerator.wait_for_everyone()

                        if accelerator.is_main_process:
                            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            unwrapped_model = accelerator.unwrap_model(model)
                            save_path = os.path.join(
                                config.eval.output_dir,
                                f"{exp_name}/checkpoint_CPT_step_{global_step}_epoch_{epoch}_{timestamp_str}.pt",
                            )

                            os.makedirs(os.path.dirname(save_path), exist_ok=True)

                            torch.save(unwrapped_model.state_dict(), save_path)
                            print(f"Saved model to {save_path}")

                        accelerator.wait_for_everyone()

        model.eval()
        fine_seq_ppl, fine_stru_ppl = evaluate_perplexity(
            model,
            val_loader,
            accelerator,
            criterion_seq,
            criterion_stru,
            mask_idx,
            pad_idx,
            chain_idx,
            special_tokens,
        )
        if (
            accelerator.is_main_process
            and wandb is not None
            and config.wandb.wandb_enabled
        ):
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "val_seq_perplexity": fine_seq_ppl,
                    "val_stru_perplexity": fine_stru_ppl,
                }
            )
    if accelerator.is_main_process:
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        unwrapped_model = accelerator.unwrap_model(model)
        save_path = os.path.join(
            config.eval.output_dir,
            f"{exp_name}/checkpoint_CPT_{timestamp_str}.pt",
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(unwrapped_model.state_dict(), save_path)
        print(f"Saved model to {save_path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Full finetune runner")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config. Defaults to cpt/config/CPT.yaml",
    )

    args = parser.parse_args(argv)

    config = load_full_finetune_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
