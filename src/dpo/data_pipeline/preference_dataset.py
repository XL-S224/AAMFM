from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dpo.training.config import DataConfig, TokenConfig


def _normalize_target_id(value: Any) -> str:
    """Match preference-design IDs to the ten-character SabDab target ID."""

    return str(value).strip()[:10]


def _cpu_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value).cpu()


def _torch_load_cpu(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(path, map_location="cpu")


def load_antigen_features(path: str, id_key: str) -> Dict[str, torch.Tensor]:
    """Load antigen node features keyed by normalized SabDab target ID."""

    loaded = _torch_load_cpu(path)
    records: Iterable[Any]
    if isinstance(loaded, Mapping):
        records = loaded.values()
    else:
        records = loaded

    features: Dict[str, torch.Tensor] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("Each antigen feature record must be a mapping")
        if id_key not in record:
            raise KeyError(f"Antigen feature record is missing ID key: {id_key}")
        if "node_feature" not in record:
            raise KeyError("Antigen feature record is missing node_feature")
        target_id = _normalize_target_id(record[id_key])
        features[target_id] = _cpu_tensor(record["node_feature"])
    return features


def _load_source_records(processed_dir: str) -> Dict[str, Mapping[str, Any]]:
    path = Path(processed_dir)
    if path.is_dir():
        path = path / "processed_single_epitope.npz"
    with np.load(path, allow_pickle=True) as archive:
        loaded = archive["data"]

    records: Dict[str, Mapping[str, Any]] = {}
    values = loaded.values() if isinstance(loaded, Mapping) else loaded
    for raw_record in values:
        if isinstance(raw_record, np.ndarray) and raw_record.shape == ():
            raw_record = raw_record.item()
        if not isinstance(raw_record, Mapping):
            raise TypeError("Each processed source record must be a mapping")
        if "id" not in raw_record:
            raise KeyError("Processed source record is missing id")
        records[_normalize_target_id(raw_record["id"])] = dict(raw_record)
    return records


class PreferenceContextStore:
    """Processed SFT prompt records joined with antigen features."""

    def __init__(
        self,
        source_records: Mapping[str, Mapping[str, Any]],
        antigen_features: Mapping[str, Any],
    ) -> None:
        self.source_records = {
            _normalize_target_id(target_id): record
            for target_id, record in source_records.items()
        }
        self.antigen_features = {
            _normalize_target_id(target_id): _cpu_tensor(feature)
            for target_id, feature in antigen_features.items()
        }

    @classmethod
    def from_config(cls, config: Any) -> "PreferenceContextStore":
        data_config = config.data if hasattr(config, "data") else config
        if not isinstance(data_config, DataConfig):
            raise TypeError("config must be DataConfig or expose a DataConfig at .data")
        if not data_config.processed_dir:
            raise ValueError("data.processed_dir is required")
        if not data_config.antigen_feature_path:
            raise ValueError("data.antigen_feature_path is required")
        return cls(
            source_records=_load_source_records(data_config.processed_dir),
            antigen_features=load_antigen_features(
                data_config.antigen_feature_path,
                data_config.antigen_feature_id_key,
            ),
        )

    def has_target(self, target_id: str) -> bool:
        return (
            target_id in self.source_records
            and target_id in self.antigen_features
        )

    def build_prompt_context(
        self,
        target_id: str,
        *,
        tokenizer: Any,
        data_config: DataConfig,
        token_config: TokenConfig,
    ) -> Dict[str, Any]:
        source = self.source_records[target_id]
        max_length = data_config.max_length
        input_ids = _resize_1d(
            source["input_ids"], max_length, token_config.pad_token_id
        )
        chain_id = _resize_1d(source["chain_id"], max_length, -2)
        cdr_positions = _resize_1d(source["cdr_pos"], max_length, 0)
        interface = _resize_1d(source["interface"], max_length, 0)
        antibody_positions = chain_id.eq(0) | chain_id.eq(1)
        cdr_mask = cdr_positions.gt(0) & antibody_positions
        masked_input_ids = input_ids.clone()
        masked_input_ids[cdr_mask] = token_config.mask_token_id
        attention_mask = input_ids.ne(token_config.pad_token_id)
        return {
            "target_id": target_id,
            "masked_input_ids": masked_input_ids,
            "chain_id": chain_id,
            "cdr_mask": cdr_mask,
            "interface": interface,
            "attention_mask": attention_mask,
            "antigen_features": self.antigen_features[target_id],
            "_antigen_sequence": _source_antigen_sequence(
                source, tokenizer, input_ids, chain_id, attention_mask
            ),
        }


def _resize_1d(value: Any, length: int, pad_value: int) -> torch.Tensor:
    tensor = _cpu_tensor(value).reshape(-1).to(dtype=torch.long)[:length]
    if tensor.numel() < length:
        padding = torch.full(
            (length - tensor.numel(),), pad_value, dtype=tensor.dtype
        )
        tensor = torch.cat((tensor, padding))
    return tensor


def _source_antigen_sequence(
    source: Mapping[str, Any],
    tokenizer: Any,
    input_ids: torch.Tensor,
    chain_id: torch.Tensor,
    attention_mask: torch.Tensor,
) -> str:
    for key in (
        "antigen_sequence",
        "antigen_sequences",
        "ag_sequence",
        "ag_sequences",
    ):
        if key not in source:
            continue
        value = source[key]
        if isinstance(value, str):
            return value.strip("|")
        return "|".join(str(sequence).strip("|") for sequence in value)

    antigen_positions = chain_id.ge(2) & attention_mask
    if antigen_positions.any() and hasattr(tokenizer, "decode"):
        antigen_chain_ids = dict.fromkeys(chain_id[antigen_positions].tolist())
        sequences = []
        for antigen_chain_id in antigen_chain_ids:
            antigen_ids = input_ids[chain_id.eq(antigen_chain_id) & attention_mask]
            sequence = tokenizer.decode(
                antigen_ids.tolist(), skip_special_tokens=True
            )
            sequences.append(str(sequence).replace(" ", "").strip("|"))
        return "|".join(sequences)
    raise ValueError("Processed source record does not provide an antigen sequence")


def _optional_float(value: Any) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    parsed = float(value)
    if np.isnan(parsed):
        return None
    return parsed


def _read_candidate_records(csv_path: str) -> list[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], Dict[str, list[float]]] = {}
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            raw_target_id = row.get("target_id", row.get("id"))
            raw_sequence = row.get("candidate_sequence", row.get("antibody"))
            if raw_target_id is None or raw_sequence is None:
                raise ValueError(
                    "Preference CSV requires id/target_id and "
                    "antibody/candidate_sequence columns"
                )
            key = (_normalize_target_id(raw_target_id), str(raw_sequence))
            aggregate = grouped.setdefault(
                key, {"ranking_scores": [], "pll_sums": []}
            )
            aggregate["ranking_scores"].append(float(row["ranking_score"]))
            pll_sum = _optional_float(row.get("pll_sum"))
            if pll_sum is not None:
                aggregate["pll_sums"].append(pll_sum)

    records = []
    for (target_id, sequence), aggregate in grouped.items():
        pll_values = aggregate["pll_sums"]
        records.append(
            {
                "target_id": target_id,
                "candidate_sequence": sequence,
                "ranking_score": sum(aggregate["ranking_scores"])
                / len(aggregate["ranking_scores"]),
                "pll_sum": sum(pll_values) / len(pll_values)
                if pll_values
                else None,
            }
        )
    return records


class _ProteinPreferenceDatasetBase(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer: Any,
        mask_function: Any = None,
        mask_idx: int = 32,
        pad_idx: int = 1,
        chain_idx: int = 31,
        special_tokens: Optional[list[int]] = None,
        min_score_diff: float = 0.04,
        device: str = "cpu",
        antigen_dict: Optional[Mapping[str, Any]] = None,
        *,
        context_store: Optional[PreferenceContextStore] = None,
        data_config: Optional[DataConfig] = None,
        token_config: Optional[TokenConfig] = None,
    ) -> None:
        del mask_function, chain_idx, special_tokens, device
        self.tokenizer = tokenizer
        self.data_config = data_config or DataConfig(
            train_csv=csv_path,
            min_score_diff=min_score_diff,
        )
        self.token_config = token_config or TokenConfig(
            mask_token_id=mask_idx,
            pad_token_id=pad_idx,
        )
        self.context_store = context_store
        if self.context_store is None:
            self.context_store = PreferenceContextStore.from_config(self.data_config)
        if antigen_dict is not None:
            self.context_store.antigen_features.update(
                {
                    _normalize_target_id(key): _cpu_tensor(value)
                    for key, value in antigen_dict.items()
                }
            )

        self.records = _read_candidate_records(csv_path)
        self._records_by_target: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
        for record in self.records:
            self._records_by_target[record["target_id"]].append(record)

        self.skipped_ids = {
            target_id
            for target_id in self._records_by_target
            if not self.context_store.has_target(target_id)
        }

    def _common_sample(self, target_id: str) -> Dict[str, Any]:
        return self.context_store.build_prompt_context(
            target_id,
            tokenizer=self.tokenizer,
            data_config=self.data_config,
            token_config=self.token_config,
        )

    def _tokenize_candidate(
        self, candidate_sequence: str, antigen_sequence: str
    ) -> torch.Tensor:
        complex_sequence = candidate_sequence.strip("|")
        if antigen_sequence:
            complex_sequence += "|" + antigen_sequence.strip("|")
        encoded = self.tokenizer(
            complex_sequence,
            truncation=True,
            padding="max_length",
            max_length=self.data_config.max_length,
            return_tensors="pt",
        )
        return _resize_1d(
            encoded["input_ids"],
            self.data_config.max_length,
            self.token_config.pad_token_id,
        )


class ProteinPreferenceDataset(_ProteinPreferenceDatasetBase):
    """Paired protein preferences joined to shared prompt context."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.data: list[Dict[str, Any]] = []
        for target_id, candidates in self._records_by_target.items():
            if target_id in self.skipped_ids:
                continue
            candidates.sort(
                key=lambda candidate: (
                    candidate["ranking_score"],
                    candidate["pll_sum"]
                    if candidate["pll_sum"] is not None
                    else float("-inf"),
                ),
                reverse=True,
            )
            for preferred_index, preferred in enumerate(candidates):
                for rejected in candidates[preferred_index + 1 :]:
                    if self._is_preference_pair(preferred, rejected):
                        self.data.append(
                            {
                                "target_id": target_id,
                                "preferred": preferred,
                                "rejected": rejected,
                            }
                        )

    def _is_preference_pair(
        self, preferred: Mapping[str, Any], rejected: Mapping[str, Any]
    ) -> bool:
        preferred_pll = preferred["pll_sum"]
        rejected_pll = rejected["pll_sum"]
        return bool(
            preferred_pll is not None
            and rejected_pll is not None
            and preferred["ranking_score"] > rejected["ranking_score"]
            and preferred_pll > rejected_pll
            and preferred["ranking_score"] - rejected["ranking_score"]
            >= self.data_config.min_score_diff
            and preferred_pll - rejected_pll >= 0.1
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        pair = self.data[index]
        sample = self._common_sample(pair["target_id"])
        antigen_sequence = sample.pop("_antigen_sequence")
        sample["preferred_ids"] = self._tokenize_candidate(
            pair["preferred"]["candidate_sequence"], antigen_sequence
        )
        sample["rejected_ids"] = self._tokenize_candidate(
            pair["rejected"]["candidate_sequence"], antigen_sequence
        )
        return sample


class ProteinPreferenceDatasetRank(_ProteinPreferenceDatasetBase):
    """Top-k protein candidates ordered by their combined preference scores."""

    def __init__(
        self,
        csv_path: str,
        tokenizer: Any,
        mask_function: Any = None,
        mask_idx: int = 32,
        pad_idx: int = 1,
        chain_idx: int = 31,
        special_tokens: Optional[list[int]] = None,
        min_score_diff: float = 0.0,
        device: str = "cpu",
        antigen_dict: Optional[Mapping[str, Any]] = None,
        top_k: int = 5,
        *,
        context_store: Optional[PreferenceContextStore] = None,
        data_config: Optional[DataConfig] = None,
        token_config: Optional[TokenConfig] = None,
    ) -> None:
        super().__init__(
            csv_path=csv_path,
            tokenizer=tokenizer,
            mask_function=mask_function,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
            chain_idx=chain_idx,
            special_tokens=special_tokens,
            min_score_diff=min_score_diff,
            device=device,
            antigen_dict=antigen_dict,
            context_store=context_store,
            data_config=data_config,
            token_config=token_config,
        )
        configured_top_k = data_config.top_k if data_config is not None else top_k
        self.top_k = configured_top_k
        self.data: list[Dict[str, Any]] = []
        for target_id, candidates in self._records_by_target.items():
            if target_id in self.skipped_ids:
                continue
            candidates.sort(
                key=lambda candidate: candidate["ranking_score"]
                + (
                    candidate["pll_sum"]
                    if candidate["pll_sum"] is not None
                    else float("-inf")
                ),
                reverse=True,
            )
            ranked_candidates = candidates[: self.top_k]
            if len(ranked_candidates) == self.top_k:
                self.data.append(
                    {"target_id": target_id, "candidates": ranked_candidates}
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ranked_group = self.data[index]
        sample = self._common_sample(ranked_group["target_id"])
        antigen_sequence = sample.pop("_antigen_sequence")
        sample["candidate_ids"] = torch.stack(
            [
                self._tokenize_candidate(
                    candidate["candidate_sequence"], antigen_sequence
                )
                for candidate in ranked_group["candidates"]
            ]
        )
        sample["candidate_scores"] = torch.tensor(
            [
                candidate["ranking_score"]
                for candidate in ranked_group["candidates"]
            ],
            dtype=torch.float64,
        )
        return sample


class PreferenceCollator:
    """Stack preference samples and pad variable-size antigen node features."""

    _COMMON_TENSOR_KEYS = (
        "masked_input_ids",
        "chain_id",
        "cdr_mask",
        "interface",
        "attention_mask",
    )

    def __call__(self, samples: list[Mapping[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise ValueError("PreferenceCollator requires at least one sample")

        ranked = "candidate_ids" in samples[0]
        expected_key = "candidate_ids" if ranked else "preferred_ids"
        if any(expected_key not in sample for sample in samples):
            raise ValueError("Cannot mix paired and ranked preference samples")

        batch: Dict[str, Any] = {
            "target_id": [str(sample["target_id"]) for sample in samples]
        }
        for key in self._COMMON_TENSOR_KEYS:
            batch[key] = torch.stack([sample[key] for sample in samples])

        if ranked:
            batch["candidate_ids"] = torch.stack(
                [sample["candidate_ids"] for sample in samples]
            )
            batch["candidate_scores"] = torch.stack(
                [sample["candidate_scores"] for sample in samples]
            )
        else:
            batch["preferred_ids"] = torch.stack(
                [sample["preferred_ids"] for sample in samples]
            )
            batch["rejected_ids"] = torch.stack(
                [sample["rejected_ids"] for sample in samples]
            )

        features = [sample["antigen_features"] for sample in samples]
        if any(feature.ndim != 2 for feature in features):
            raise ValueError("antigen_features must have shape [nodes, features]")
        feature_width = features[0].shape[1]
        if feature_width != 3072:
            raise ValueError("antigen_features must have feature width 3072")
        if any(feature.shape[1] != feature_width for feature in features):
            raise ValueError("All antigen feature widths must match")
        max_nodes = max(feature.shape[0] for feature in features)
        padded_features = features[0].new_zeros(
            (len(features), max_nodes, feature_width)
        )
        feature_mask = torch.zeros(
            (len(features), max_nodes), dtype=torch.bool
        )
        for batch_index, feature in enumerate(features):
            node_count = feature.shape[0]
            padded_features[batch_index, :node_count] = feature
            feature_mask[batch_index, :node_count] = True
        batch["antigen_features"] = padded_features
        batch["antigen_feature_mask"] = feature_mask
        return batch


__all__ = [
    "PreferenceContextStore",
    "PreferenceCollator",
    "ProteinPreferenceDataset",
    "ProteinPreferenceDatasetRank",
    "load_antigen_features",
]
