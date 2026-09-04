from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

# Direct execution starts with ``src/dpo/training`` on sys.path. Add ``src``
# before importing canonical dpo modules so the documented command is portable.
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from dpo.data_pipeline.preference_dataset import (
    PreferenceCollator,
    PreferenceContextStore,
    ProteinPreferenceDataset,
    ProteinPreferenceDatasetRank,
)
from dpo.training.config import DPOConfig, load_dpo_config
from dpo.training.trainer_utils import CheckpointManager, TrainerState

try:
    import wandb
except ImportError:  # pragma: no cover - depends on the optional environment
    wandb = None


@dataclass
class LossOutput:
    loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]


class MetricsTracker:
    def __init__(self, accelerator: Accelerator) -> None:
        self.accelerator = accelerator
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, torch.Tensor]) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach()
                total = value.sum().item()
                count = value.numel()
            else:
                total = float(value)
                count = 1
            self.sums[key] = self.sums.get(key, 0.0) + total
            self.counts[key] = self.counts.get(key, 0) + count

    def compute(self) -> Dict[str, float]:
        results = {}
        for key in self.sums:
            sum_tensor = torch.tensor(self.sums[key], device=self.accelerator.device)
            count_tensor = torch.tensor(
                self.counts[key], device=self.accelerator.device
            )
            sum_tensor = self.accelerator.reduce(sum_tensor, reduction="sum")
            count_tensor = self.accelerator.reduce(count_tensor, reduction="sum")
            results[key] = (sum_tensor / count_tensor).item()
        return results


def _flatten_ranked_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    candidate_ids = batch["candidate_ids"]
    if candidate_ids.ndim != 3:
        raise ValueError("candidate_ids must have shape [batch, candidates, length]")
    batch_size, candidate_count, sequence_length = candidate_ids.shape
    flattened: Dict[str, torch.Tensor] = {
        "candidate_ids": candidate_ids.reshape(
            batch_size * candidate_count, sequence_length
        ),
        "candidate_scores": batch["candidate_scores"].reshape(
            batch_size * candidate_count
        ),
    }
    for key in (
        "masked_input_ids",
        "chain_id",
        "interface",
        "cdr_mask",
        "attention_mask",
        "antigen_features",
        "antigen_feature_mask",
    ):
        if key not in batch:
            continue
        value = batch[key]
        expanded = value.unsqueeze(1).expand(
            batch_size, candidate_count, *value.shape[1:]
        )
        flattened[key] = expanded.reshape(
            batch_size * candidate_count, *value.shape[1:]
        )
    return flattened


class CDRDPOTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        tokenizer,
        train_dataset,
        val_dataset,
        config: DPOConfig,
        accelerator: Optional[Accelerator] = None,
    ) -> None:
        if not config.eval.output_dir:
            raise ValueError("eval.output_dir is required")
        self.config = config
        self.tokenizer = tokenizer
        self.loss_type = config.data.loss_type
        self.normalize_logps = config.training.normalize_logps
        self.global_step = 0
        self._wandb_enabled = bool(
            config.wandb.wandb_enabled and wandb is not None
        )
        self.accelerator = accelerator or Accelerator(
            mixed_precision=config.training.mixed_precision,
            gradient_accumulation_steps=config.training.accumulation_steps,
            log_with="wandb" if self._wandb_enabled else None,
            project_dir=config.eval.output_dir,
        )
        self.device = self.accelerator.device

        self.model = model
        self.ref_model = ref_model
        for parameter in self.ref_model.parameters():
            parameter.requires_grad = False
        self.ref_model.eval()
        self.ref_device = self.device
        if config.model.ref_model_device == "cpu":
            self.ref_device = torch.device("cpu")
        elif config.model.ref_model_device != "same":
            raise ValueError(
                f"Unsupported ref_model_device: {config.model.ref_model_device}"
            )
        self.ref_model.to(self.ref_device)
        self.model.to(self.device)

        collator = PreferenceCollator()
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            collate_fn=collator,
        )
        self.val_dataloader = None
        if val_dataset is not None:
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                pin_memory=config.data.pin_memory,
                collate_fn=collator,
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.training.lr
        )
        steps_per_epoch = math.ceil(
            len(self.train_dataloader)
            / config.training.accumulation_steps
        )
        total_steps = steps_per_epoch * config.training.epochs
        warmup_steps = int(total_steps * config.training.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        )
        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        self.state = TrainerState()
        self.checkpoints = CheckpointManager(
            config.eval.output_dir,
            self.accelerator,
            config.eval.metric_for_best,
            config.eval.greater_is_better,
        )
        if self._wandb_enabled and self.accelerator.is_main_process:
            wandb.init(
                project=config.wandb.wandb_project,
                config={
                    "loss_type": self.loss_type,
                    "beta": config.training.beta,
                    "lr": config.training.lr,
                    "batch_size": config.training.batch_size,
                    "epochs": config.training.epochs,
                    "accumulation_steps": config.training.accumulation_steps,
                    "normalize_logps": self.normalize_logps,
                },
            )

    def _get_cdr_logps_for_sequence(
        self,
        model: torch.nn.Module,
        masked_input_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        chain_id: torch.Tensor,
        antigen_features: torch.Tensor,
        interface: torch.Tensor,
        cdr_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        antigen_feature_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        masked_input_ids = masked_input_ids.to(device)
        candidate_ids = candidate_ids.to(device)
        chain_id = chain_id.to(device)
        antigen_features = antigen_features.to(device)
        interface = interface.to(device)
        cdr_mask = cdr_mask.to(device)
        if attention_mask is None:
            attention_mask = masked_input_ids.ne(self.tokenizer.pad_token_id)
        else:
            attention_mask = attention_mask.to(device)
        if antigen_feature_mask is not None:
            antigen_feature_mask = antigen_feature_mask.to(device).bool()

        if antigen_feature_mask is None or antigen_feature_mask.all():
            outputs = model(
                sequence_tokens=masked_input_ids,
                chain_id=chain_id,
                interface=interface,
                antigen_feat=antigen_features,
            )
            sequence_logits = outputs.sequence_logits
        else:
            sequence_logits_parts = []
            for sample_index in range(masked_input_ids.shape[0]):
                valid_nodes = antigen_feature_mask[sample_index]
                if not valid_nodes.any():
                    raise ValueError(
                        "antigen_feature_mask requires at least one valid node per sample"
                    )
                outputs = model(
                    sequence_tokens=masked_input_ids[sample_index : sample_index + 1],
                    chain_id=chain_id[sample_index : sample_index + 1],
                    interface=interface[sample_index : sample_index + 1],
                    antigen_feat=antigen_features[sample_index][valid_nodes].unsqueeze(0),
                )
                sequence_logits_parts.append(outputs.sequence_logits)
            sequence_logits = torch.cat(sequence_logits_parts, dim=0)

        log_probs = F.log_softmax(sequence_logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs, dim=-1, index=candidate_ids.unsqueeze(-1)
        ).squeeze(-1)
        cdr_attention_mask = attention_mask.bool() & cdr_mask.bool()
        sum_log_probs = selected_log_probs.masked_fill(
            ~cdr_attention_mask, 0.0
        ).sum(dim=1)

        if self.normalize_logps:
            cdr_token_counts = cdr_attention_mask.sum(dim=1).clamp(min=1)
            sum_log_probs = sum_log_probs / cdr_token_counts
        return sum_log_probs

    def _paired_loss_and_metrics(
        self,
        masked_input_ids: torch.Tensor,
        preferred_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chain_id: torch.Tensor,
        antigen_features: torch.Tensor,
        interface: torch.Tensor,
        cdr_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        antigen_feature_mask: Optional[torch.Tensor],
    ) -> LossOutput:
        policy_preferred = self._get_cdr_logps_for_sequence(
            self.model,
            masked_input_ids,
            preferred_ids,
            chain_id,
            antigen_features,
            interface,
            cdr_mask,
            attention_mask,
            antigen_feature_mask,
        )
        policy_rejected = self._get_cdr_logps_for_sequence(
            self.model,
            masked_input_ids,
            rejected_ids,
            chain_id,
            antigen_features,
            interface,
            cdr_mask,
            attention_mask,
            antigen_feature_mask,
        )
        with torch.no_grad():
            reference_preferred = self._get_cdr_logps_for_sequence(
                self.ref_model,
                masked_input_ids,
                preferred_ids,
                chain_id,
                antigen_features,
                interface,
                cdr_mask,
                attention_mask,
                antigen_feature_mask,
            ).to(policy_preferred.device)
            reference_rejected = self._get_cdr_logps_for_sequence(
                self.ref_model,
                masked_input_ids,
                rejected_ids,
                chain_id,
                antigen_features,
                interface,
                cdr_mask,
                attention_mask,
                antigen_feature_mask,
            ).to(policy_rejected.device)

        preferred_logratios = policy_preferred - reference_preferred
        rejected_logratios = policy_rejected - reference_rejected
        inside_term = self.config.training.beta * (
            preferred_logratios - rejected_logratios
        )
        loss = -F.logsigmoid(inside_term).mean()
        metrics = {
            "pi_preferred_logps": policy_preferred.detach(),
            "pi_rejected_logps": policy_rejected.detach(),
            "ref_preferred_logps": reference_preferred.detach(),
            "ref_rejected_logps": reference_rejected.detach(),
            "pi_reward_gap": (policy_preferred - policy_rejected).detach(),
            "ref_reward_gap": (reference_preferred - reference_rejected).detach(),
            "pairwise_accuracy": (policy_preferred > policy_rejected).float().detach(),
            "inside_term": inside_term.detach(),
        }
        return LossOutput(loss=loss, metrics=metrics)

    def _ranked_loss_and_metrics(
        self, batch: Dict[str, torch.Tensor]
    ) -> LossOutput:
        batch_size, candidate_count, _ = batch["candidate_ids"].shape
        candidate_scores = batch["candidate_scores"].to(
            device=self.device, dtype=torch.float32
        )
        flattened = {
            key: value.to(self.device)
            for key, value in _flatten_ranked_batch(batch).items()
        }
        policy_logps_flat = self._get_cdr_logps_for_sequence(
            self.model,
            flattened["masked_input_ids"],
            flattened["candidate_ids"],
            flattened["chain_id"],
            flattened["antigen_features"],
            flattened["interface"],
            flattened["cdr_mask"],
            flattened.get("attention_mask"),
            flattened.get("antigen_feature_mask"),
        )
        with torch.no_grad():
            reference_logps_flat = self._get_cdr_logps_for_sequence(
                self.ref_model,
                flattened["masked_input_ids"],
                flattened["candidate_ids"],
                flattened["chain_id"],
                flattened["antigen_features"],
                flattened["interface"],
                flattened["cdr_mask"],
                flattened.get("attention_mask"),
                flattened.get("antigen_feature_mask"),
            ).to(policy_logps_flat.device)

        policy_logps = policy_logps_flat.reshape(batch_size, candidate_count)
        reference_logps = reference_logps_flat.reshape(batch_size, candidate_count)
        total_loss = policy_logps.new_zeros(())
        total_valid_pairs = policy_logps.new_zeros(())
        inside_terms = []
        for preferred_index in range(candidate_count):
            for rejected_index in range(preferred_index + 1, candidate_count):
                valid_mask = (
                    candidate_scores[:, preferred_index]
                    > candidate_scores[:, rejected_index]
                )
                if not valid_mask.any():
                    continue
                preferred_logratios = (
                    policy_logps[:, preferred_index]
                    - reference_logps[:, preferred_index]
                )
                rejected_logratios = (
                    policy_logps[:, rejected_index]
                    - reference_logps[:, rejected_index]
                )
                inside_term = self.config.training.beta * (
                    preferred_logratios - rejected_logratios
                )
                total_loss = total_loss + (-F.logsigmoid(inside_term[valid_mask])).sum()
                total_valid_pairs = total_valid_pairs + valid_mask.sum()
                inside_terms.append(inside_term[valid_mask])

        if inside_terms:
            loss = total_loss / total_valid_pairs
            inside_term_tensor = torch.cat(inside_terms)
        else:
            loss = policy_logps.sum() * 0.0
            inside_term_tensor = policy_logps.new_zeros(1)
        metrics = {
            "pi_preferred_logps": policy_logps[:, 0].detach(),
            "pi_rejected_logps": policy_logps[:, -1].detach(),
            "ref_preferred_logps": reference_logps[:, 0].detach(),
            "ref_rejected_logps": reference_logps[:, -1].detach(),
            "pi_reward_gap": (policy_logps[:, 0] - policy_logps[:, -1]).detach(),
            "ref_reward_gap": (
                reference_logps[:, 0] - reference_logps[:, -1]
            ).detach(),
            "pairwise_accuracy": (inside_term_tensor > 0).float().detach(),
            "inside_term": inside_term_tensor.detach(),
        }
        return LossOutput(loss=loss, metrics=metrics)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> LossOutput:
        masked_input_ids = batch["masked_input_ids"].to(self.device)
        chain_id = batch["chain_id"].to(self.device)
        antigen_features = batch["antigen_features"].to(self.device)
        interface = batch["interface"].to(self.device)
        cdr_mask = batch["cdr_mask"].to(self.device)
        attention_mask = batch.get("attention_mask")
        antigen_feature_mask = batch.get("antigen_feature_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if antigen_feature_mask is not None:
            antigen_feature_mask = antigen_feature_mask.to(self.device)

        if self.loss_type == "paired":
            return self._paired_loss_and_metrics(
                masked_input_ids,
                batch["preferred_ids"].to(self.device),
                batch["rejected_ids"].to(self.device),
                chain_id,
                antigen_features,
                interface,
                cdr_mask,
                attention_mask,
                antigen_feature_mask,
            )
        if self.loss_type == "ranked":
            return self._ranked_loss_and_metrics(batch)
        raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def _set_dataloader_epoch(self, epoch: int) -> None:
        dataloader = self.train_dataloader
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

    def save_checkpoint(self, tag: str) -> None:
        self.checkpoints.save(tag, self.state)

    def load_checkpoint(self, path: str) -> None:
        self.state = self.checkpoints.load(path, self.state)
        self.global_step = self.state.global_step

    def maybe_resume(self) -> None:
        resume_from = self.config.checkpoint.resume_from
        if resume_from:
            self.load_checkpoint(resume_from)
            return
        if self.config.checkpoint.auto_resume and self.checkpoints.has_best():
            self.load_checkpoint(str(self.checkpoints.best_dir()))

    def train(self) -> None:
        self.maybe_resume()
        self.model.train()
        start_epoch = self.state.epoch
        for epoch in range(start_epoch, self.config.training.epochs):
            if epoch != start_epoch:
                self.state.step_in_epoch = 0
            self.state.epoch = epoch
            self._set_dataloader_epoch(epoch)
            self.accelerator.wait_for_everyone()

            iterator = iter(self.train_dataloader)
            if epoch == start_epoch and self.state.step_in_epoch > 0:
                skipped_microbatches = (
                    self.state.step_in_epoch
                    * self.config.training.accumulation_steps
                )
                for _ in range(skipped_microbatches):
                    try:
                        next(iterator)
                    except StopIteration:
                        break
            if self.accelerator.is_main_process:
                progress = tqdm(iterator, desc=f"Epoch {epoch + 1}")
            else:
                progress = iterator

            for batch in progress:
                with self.accelerator.accumulate(self.model):
                    loss_out = self.compute_loss(batch)
                    self.accelerator.backward(loss_out.loss)
                    if (
                        self.accelerator.sync_gradients
                        and self.config.training.max_grad_norm > 0
                    ):
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.max_grad_norm,
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.is_main_process and isinstance(progress, tqdm):
                    progress.set_postfix({"loss": loss_out.loss.item()})
                if not self.accelerator.sync_gradients:
                    continue

                self.global_step += 1
                self.state.global_step = self.global_step
                self.state.step_in_epoch += 1
                if (
                    self.accelerator.is_main_process
                    and self._wandb_enabled
                    and self.config.wandb.log_every_steps > 0
                    and self.global_step % self.config.wandb.log_every_steps == 0
                ):
                    wandb.log(
                        {"train/loss": loss_out.loss.item()}, step=self.global_step
                    )
                if (
                    self.config.eval.every_save_steps > 0
                    and self.global_step % self.config.eval.every_save_steps == 0
                ):
                    self.save_checkpoint(f"step_{self.global_step}")

                if (
                    self.val_dataloader is not None
                    and self.config.eval.eval_every_steps > 0
                    and self.global_step % self.config.eval.eval_every_steps == 0
                ):
                    eval_metrics = self.evaluate()
                    if (
                        self.config.eval.metric_for_best in eval_metrics
                        and self.checkpoints.is_better(
                            eval_metrics[self.config.eval.metric_for_best],
                            self.state.best_metric,
                        )
                    ):
                        self.state.best_metric = eval_metrics[
                            self.config.eval.metric_for_best
                        ]
                        self.save_checkpoint("best")
                    if self.accelerator.is_main_process and self._wandb_enabled:
                        wandb.log(eval_metrics, step=self.global_step)

            eval_metrics = {}
            if self.val_dataloader is not None and self.config.eval.eval_every_steps == 0:
                eval_metrics = self.evaluate()
                if (
                    self.config.eval.metric_for_best in eval_metrics
                    and self.checkpoints.is_better(
                        eval_metrics[self.config.eval.metric_for_best],
                        self.state.best_metric,
                    )
                ):
                    self.state.best_metric = eval_metrics[
                        self.config.eval.metric_for_best
                    ]
                    self.save_checkpoint("best")
            if self.config.checkpoint.save_epochs:
                self.save_checkpoint(f"epoch_{epoch + 1}")
            if self.accelerator.is_main_process and self._wandb_enabled:
                payload = {"train/epoch": epoch + 1}
                payload.update(eval_metrics)
                wandb.log(payload, step=self.global_step)

        if self.accelerator.is_main_process and self._wandb_enabled:
            wandb.finish()

    def evaluate(self) -> Dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        tracker = MetricsTracker(self.accelerator)
        loss_tracker = MetricsTracker(self.accelerator)
        if self.accelerator.is_main_process:
            iterator = tqdm(self.val_dataloader, desc="Validating")
        else:
            iterator = self.val_dataloader
        try:
            with torch.no_grad():
                for batch in iterator:
                    loss_out = self.compute_loss(batch)
                    loss_tracker.update({"val/loss": loss_out.loss.detach()})
                    tracker.update(loss_out.metrics)
            metrics = {
                f"val/{key}": value for key, value in tracker.compute().items()
            }
            metrics.update(loss_tracker.compute())
            return metrics
        finally:
            self.model.train(was_training)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    accelerate_set_seed(seed)


def build_datasets(config: DPOConfig):
    for field_name in ("train_csv", "processed_dir", "antigen_feature_path"):
        if not getattr(config.data, field_name):
            raise ValueError(f"data.{field_name} is required")
    # ESM tokenization is imported only when a real training run reaches data
    # construction; importing the trainer and requesting --help stay lightweight.
    from esm.tokenization import EsmSequenceTokenizer

    tokenizer = EsmSequenceTokenizer()
    context_store = PreferenceContextStore.from_config(config)
    dataset_type = {
        "paired": ProteinPreferenceDataset,
        "ranked": ProteinPreferenceDatasetRank,
    }.get(config.data.loss_type)
    if dataset_type is None:
        raise ValueError(f"Unsupported loss_type: {config.data.loss_type}")

    def create_dataset(csv_path: str):
        return dataset_type(
            csv_path=csv_path,
            tokenizer=tokenizer,
            context_store=context_store,
            data_config=config.data,
            token_config=config.tokens,
        )

    full_dataset = create_dataset(config.data.train_csv)
    if config.data.val_csv:
        return tokenizer, full_dataset, create_dataset(config.data.val_csv)

    val_size = int(len(full_dataset) * config.data.val_split)
    if val_size == 0:
        return tokenizer, full_dataset, None
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    split_at = len(full_dataset) - val_size
    return (
        tokenizer,
        Subset(full_dataset, indices[:split_at]),
        Subset(full_dataset, indices[split_at:]),
    )


def build_policy_and_ref_models(model_config):
    from dpo.training.model_loader import build_policy_and_ref_models as build_models

    return build_models(model_config)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_config = Path(__file__).resolve().parents[1] / "config" / "DPO.yaml"
    parser = argparse.ArgumentParser(description="Train CDR-DPO with config")
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to the DPO YAML config file",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Optional checkpoint path to resume from",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = load_dpo_config(args.config)
    if args.resume_from is not None:
        config.checkpoint.resume_from = args.resume_from

    if not config.eval.output_dir:
        raise ValueError("eval.output_dir is required")
    os.makedirs(config.eval.output_dir, exist_ok=True)
    set_seed(config.common.seed)
    tokenizer, train_dataset, val_dataset = build_datasets(config)
    policy_model, reference_model = build_policy_and_ref_models(config.model)
    trainer = CDRDPOTrainer(
        model=policy_model,
        ref_model=reference_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )
    trainer.train()


__all__ = [
    "CDRDPOTrainer",
    "LossOutput",
    "MetricsTracker",
    "_flatten_ranked_batch",
    "build_datasets",
    "main",
    "parse_args",
    "set_seed",
]


if __name__ == "__main__":
    main()
