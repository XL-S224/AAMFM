from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any


MAX_LENGTH = 1024
MASK_IDX = 32
PAD_IDX = 1
STRUCTURE_MASK_IDX = 4096
SPECIAL_STRUCTURE_TOKENS = [4096, 4097, 4098, 4099, 4100]
CSV_FIELDS = ["id", "antibody_seq", "antigen_seq", "h_chain_id", "l_chain_id"]
ESM3_WEIGHT_FILES = (
    "esm3_sm_open_v1.pth",
    "esm3_structure_encoder_v0.pth",
    "esm3_structure_decoder_v0.pth",
    "esm3_function_decoder_v0.pth",
)


@dataclass(frozen=True)
class GenerationSuccess:
    target_id: str
    sample_index: int
    design_id: str
    pdb_path: Path
    feature_source: str


@dataclass(frozen=True)
class GenerationFailure:
    target_id: str
    sample_index: int | None
    stage: str
    error: str

    @property
    def design_id(self) -> str | None:
        if self.sample_index is None:
            return None
        return f"{self.target_id}_{self.sample_index}"


@dataclass
class GenerationSummary:
    requested_targets: int
    samples_per_target: int
    successes: list[GenerationSuccess] = field(default_factory=list)
    failures: list[GenerationFailure] = field(default_factory=list)
    fallback_target_ids: set[str] = field(default_factory=set)

    @property
    def expected_designs(self) -> int:
        return self.requested_targets * self.samples_per_target

    @property
    def failed_designs(self) -> int:
        return self.expected_designs - len(self.successes)

    @property
    def fallback_design_ids(self) -> list[str]:
        return sorted(
            success.design_id
            for success in self.successes
            if success.feature_source == "zero_fallback"
        )

    def record_fallback_target(self, target_id: str) -> None:
        self.fallback_target_ids.add(str(target_id))

    def record_success(
        self,
        target_id: str,
        sample_index: int,
        pdb_path: Path,
        feature_source: str,
    ) -> None:
        target_id = str(target_id)
        self.successes.append(
            GenerationSuccess(
                target_id=target_id,
                sample_index=int(sample_index),
                design_id=f"{target_id}_{sample_index}",
                pdb_path=Path(pdb_path),
                feature_source=feature_source,
            )
        )

    def record_failure(
        self,
        target_id: str,
        sample_index: int | None,
        stage: str,
        error: BaseException | str,
    ) -> None:
        self.failures.append(
            GenerationFailure(
                target_id=str(target_id),
                sample_index=sample_index,
                stage=stage,
                error=str(error),
            )
        )

    def validate_complete(self) -> None:
        produced = len(self.successes)
        if produced != self.expected_designs:
            raise RuntimeError(
                f"expected {self.expected_designs} designs, produced {produced}"
            )

        design_ids = [success.design_id for success in self.successes]
        if len(set(design_ids)) != len(design_ids):
            raise RuntimeError("generation produced duplicate design IDs")

        target_counts = Counter(success.target_id for success in self.successes)
        if len(target_counts) != self.requested_targets:
            raise RuntimeError(
                f"expected {self.requested_targets} targets, produced {len(target_counts)}"
            )
        wrong_counts = {
            target_id: count
            for target_id, count in target_counts.items()
            if count != self.samples_per_target
        }
        if wrong_counts:
            raise RuntimeError(
                "incorrect designs per target: "
                + ", ".join(
                    f"{target_id}={count}"
                    for target_id, count in sorted(wrong_counts.items())
                )
            )

        missing_pdbs = [
            str(success.pdb_path)
            for success in self.successes
            if not success.pdb_path.is_file() or success.pdb_path.stat().st_size == 0
        ]
        if missing_pdbs:
            raise RuntimeError(
                "missing or empty generated PDBs: " + ", ".join(missing_pdbs)
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reference-guided AAMFM generation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rabd-json", default="datasets/eval/rabd/rabd.json")
    parser.add_argument("--pdb-dir", default="datasets/eval/rabd/pdb")
    parser.add_argument(
        "--antigen-features",
        default="datasets/eval/rabd/gearnet_node_features.pt",
    )
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--allow-missing-antigen-features", action="store_true")
    parser.add_argument("--cdr-mode", choices=["h3", "full"], default="h3")
    parser.add_argument("--cdr-index", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-targets", type=int, default=60)
    parser.add_argument("--samples-per-target", type=int, default=10)
    parser.add_argument("--sequence-steps", type=int, default=100)
    parser.add_argument("--structure-steps", type=int, default=100)
    parser.add_argument("--schedule", default="linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", required=True)
    return parser


def resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _validate_positive_counts(args: argparse.Namespace) -> None:
    for name in (
        "mini_batch_size",
        "num_targets",
        "samples_per_target",
        "sequence_steps",
        "structure_steps",
    ):
        value = getattr(args, name)
        if value <= 0:
            option = name.replace("_", "-")
            raise ValueError(f"--{option} must be positive")
    if not 1 <= args.cdr_index <= 6:
        raise ValueError("--cdr-index must be between 1 and 6")


def resolve_and_validate_paths(
    args: argparse.Namespace, project_root: Path
) -> dict[str, Path]:
    _validate_positive_counts(args)
    paths = {
        "checkpoint": resolve_path(args.checkpoint, project_root),
        "rabd_json": resolve_path(args.rabd_json, project_root),
        "pdb_dir": resolve_path(args.pdb_dir, project_root),
        "antigen_features": resolve_path(args.antigen_features, project_root),
        "esm3_weights": project_root / "model/base_model/esm3/data/weights",
        "output_dir": resolve_path(args.output_dir, project_root),
    }
    for key in ("checkpoint", "rabd_json", "antigen_features"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"{key.replace('_', ' ')} not found: {paths[key]}")
    if not paths["pdb_dir"].is_dir():
        raise FileNotFoundError(f"PDB directory not found: {paths['pdb_dir']}")
    for filename in ESM3_WEIGHT_FILES:
        weight_path = paths["esm3_weights"] / filename
        if not weight_path.is_file():
            raise FileNotFoundError(f"ESM3 base weight not found: {weight_path}")
    if paths["output_dir"].exists() and not paths["output_dir"].is_dir():
        raise NotADirectoryError(f"output directory is not a directory: {paths['output_dir']}")
    if paths["output_dir"].is_dir():
        control_paths = [
            paths["output_dir"] / filename
            for filename in (
                "generation.csv",
                "generation.csv.tmp",
                "generation_manifest.json",
            )
        ]
        prior_artifacts = [path for path in control_paths if path.exists()]
        prior_artifacts.extend(sorted(paths["output_dir"].glob("*.pdb")))
        if prior_artifacts:
            names = ", ".join(path.name for path in prior_artifacts)
            raise FileExistsError(
                f"output directory contains prior generation artifacts: {names}"
            )
    return paths


def clean_decoded_sequence(text: str) -> str:
    text = re.sub(r"<pad>|<cls>|<eos>", "", text)
    text = re.sub(r"<mask>", "_", text)
    return re.sub(r"\s+", "", text)


def unwrap_state_dict(state_dict: Any) -> Any:
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        if isinstance(state_dict, dict):
            state_dict = {
                key.replace("module.", ""): value
                for key, value in state_dict.items()
            }
    return state_dict


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_antigen_feature(
    antigen_features: dict[str, Any], sample_id: str, allow_missing: bool
) -> tuple[Any, str]:
    feature = antigen_features.get(sample_id)
    feature_source = "gearnet"
    if feature is None:
        if not allow_missing:
            raise KeyError(f"missing GearNet node_feature for {sample_id}")
        import torch

        feature = torch.zeros((1, 3072), dtype=torch.float32)
        feature_source = "zero_fallback"
    return feature, feature_source


def write_generation_csv(
    rows: list[dict[str, Any]], output_dir: Path, summary: GenerationSummary
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "generation.csv"
    tmp_path = output_dir / "generation.csv.tmp"
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        summary.validate_complete()
        row_ids = [str(row.get("id", "")) for row in rows]
        success_ids = [success.design_id for success in summary.successes]
        if len(row_ids) != summary.expected_designs:
            raise RuntimeError(
                f"expected {summary.expected_designs} CSV rows, wrote {len(row_ids)}"
            )
        if len(set(row_ids)) != len(row_ids):
            raise RuntimeError("generation CSV contains duplicate design IDs")
        if set(row_ids) != set(success_ids):
            raise RuntimeError("generation CSV and generated PDB IDs disagree")

        os.replace(tmp_path, csv_path)
        return csv_path
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def to_structure_encoder_inputs(
    chain: Any, should_normalize_coordinates: bool = True
) -> tuple[Any, Any, Any]:
    import torch

    coords = torch.tensor(chain.atom37_positions, dtype=torch.float32)
    plddt = torch.tensor(chain.confidence, dtype=torch.float32)
    residue_index = torch.tensor(chain.residue_index, dtype=torch.long)

    if should_normalize_coordinates:
        finite_mask = torch.isfinite(coords).all(dim=-1, keepdim=True)
        finite_coords = coords.masked_select(finite_mask).view(-1, 3)
        if finite_coords.numel() > 0:
            center = finite_coords.mean(dim=0)
            coords = torch.where(finite_mask, coords - center, coords)

    return coords.unsqueeze(0), plddt.unsqueeze(0), residue_index.unsqueeze(0)


def build_cdr_positions(chain_ids: Any, residue_index: Any) -> Any:
    import torch

    cdr_pos = torch.zeros(MAX_LENGTH, dtype=torch.long)
    h_mask = chain_ids == 0
    l_mask = chain_ids == 1

    cdr_pos[(residue_index >= 27) & (residue_index <= 38) & h_mask] = 1
    cdr_pos[(residue_index >= 56) & (residue_index <= 65) & h_mask] = 2
    cdr_pos[(residue_index >= 105) & (residue_index <= 117) & h_mask] = 3
    cdr_pos[(residue_index >= 27) & (residue_index <= 38) & l_mask] = 4
    cdr_pos[(residue_index >= 56) & (residue_index <= 65) & l_mask] = 5
    cdr_pos[(residue_index >= 105) & (residue_index <= 117) & l_mask] = 6
    return cdr_pos


def build_interface_mask(
    heavy_chain: Any,
    light_chain: Any,
    antigen_chains: list[Any],
    *,
    max_length: int = MAX_LENGTH,
    max_interface_residues: int = 64,
) -> Any:
    """Select the antigen residues closest to either antibody chain."""
    import numpy as np
    import torch

    interface = torch.zeros((1, max_length), dtype=torch.long)
    chain_coordinates: dict[int, np.ndarray] = {}
    chain_sequence_indices: dict[int, list[int]] = {}
    current_sequence_index = 1  # position zero is the sequence BOS token

    for chain_index, chain in enumerate(
        [heavy_chain, light_chain, *antigen_chains]
    ):
        coordinates: list[Any] = []
        sequence_indices: list[int] = []
        for residue_coordinates in chain.atom37_positions:
            ca_coordinate = residue_coordinates[1]
            if not np.isnan(ca_coordinate).any():
                coordinates.append(ca_coordinate)
                sequence_indices.append(current_sequence_index)
            current_sequence_index += 1
        chain_coordinates[chain_index] = np.asarray(coordinates)
        chain_sequence_indices[chain_index] = sequence_indices

    ranked_antigen_residues: list[tuple[float, int, int]] = []
    for antigen_index in range(2, 2 + len(antigen_chains)):
        antigen_coordinates = chain_coordinates[antigen_index]
        if not len(antigen_coordinates):
            continue
        minimum_distances = np.full(len(antigen_coordinates), np.inf)
        for antibody_index in (0, 1):
            antibody_coordinates = chain_coordinates[antibody_index]
            if not len(antibody_coordinates):
                continue
            distances = np.linalg.norm(
                antigen_coordinates[:, None, :]
                - antibody_coordinates[None, :, :],
                axis=-1,
            )
            minimum_distances = np.minimum(
                minimum_distances, distances.min(axis=1)
            )
        ranked_antigen_residues.extend(
            (float(distance), antigen_index, local_index)
            for local_index, distance in enumerate(minimum_distances)
        )

    ranked_antigen_residues.sort(key=lambda record: record[0])
    for _, chain_index, local_index in ranked_antigen_residues[
        :max_interface_residues
    ]:
        sequence_index = chain_sequence_indices[chain_index][local_index]
        if sequence_index < max_length:
            interface[0, sequence_index] = 1
    return interface


def build_structure_tokens_and_metadata(
    entry: dict[str, Any],
    pdb_path: Path,
    tokenizer: Any,
    encoder: Any,
    device: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from esm.utils.structure.protein_chain import ProteinChain

    h_chain = ProteinChain.from_pdb(str(pdb_path), entry["H_chain"])
    l_chain = ProteinChain.from_pdb(str(pdb_path), entry["L_chain"])
    ag_chains = [
        ProteinChain.from_pdb(str(pdb_path), chain_id)
        for chain_id in entry["ag_chains"]
    ]

    h_sequence = h_chain.sequence
    l_sequence = l_chain.sequence
    ag_sequences = [chain.sequence for chain in ag_chains]
    complex_tokens = tokenizer(
        h_sequence + "|" + l_sequence + "|" + "|".join(ag_sequences),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    h_coordinates, _, h_index = to_structure_encoder_inputs(h_chain)
    l_coordinates, _, l_index = to_structure_encoder_inputs(l_chain)
    ag_inputs = [to_structure_encoder_inputs(chain) for chain in ag_chains]

    h_coordinates = h_coordinates.to(device)
    l_coordinates = l_coordinates.to(device)
    h_index = h_index.to(device)
    l_index = l_index.to(device)
    ag_coordinates = [coords.to(device) for coords, _, _ in ag_inputs]
    ag_index = [index.to(device) for _, _, index in ag_inputs]

    separator = torch.full((1, 1), 4100, dtype=torch.long, device=device)
    _, h_structure_tokens = encoder.encode(h_coordinates, residue_index=h_index)
    _, l_structure_tokens = encoder.encode(l_coordinates, residue_index=l_index)
    ag_structure_tokens = [
        encoder.encode(coords, residue_index=index)[1]
        for coords, index in zip(ag_coordinates, ag_index)
    ]

    ag_tokens_with_separator: list[Any] = []
    for tokens in ag_structure_tokens:
        ag_tokens_with_separator.extend((tokens, separator))
    structure_tokens = torch.cat(
        [h_structure_tokens, separator, l_structure_tokens, separator]
        + ag_tokens_with_separator,
        dim=1,
    )
    combined_structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
    combined_structure_tokens[:, 0] = 4098
    combined_structure_tokens[:, -1] = 4097
    combined_structure_tokens = combined_structure_tokens[:, :MAX_LENGTH]

    chain_separator = torch.full((1, 1), -1, dtype=torch.long, device=device)
    h_chain_ids = torch.full_like(h_structure_tokens, 0)
    l_chain_ids = torch.full_like(l_structure_tokens, 1)
    ag_chain_ids = [
        torch.full(
            (tokens.size(0), tokens.size(1)),
            index + 2,
            dtype=torch.long,
            device=device,
        )
        for index, tokens in enumerate(ag_structure_tokens)
    ]
    ag_chain_ids_with_separator: list[Any] = []
    for chain_ids in ag_chain_ids:
        ag_chain_ids_with_separator.extend((chain_ids, chain_separator))
    chain_ids = torch.cat(
        [h_chain_ids, chain_separator, l_chain_ids, chain_separator]
        + ag_chain_ids_with_separator,
        dim=1,
    )
    chain_ids = F.pad(chain_ids, (1, 1), value=-2)[:, :MAX_LENGTH]

    index_separator = torch.full((1, 1), -1, dtype=torch.long, device=device)
    ag_index_with_separator: list[Any] = []
    for index in ag_index:
        ag_index_with_separator.extend((index, index_separator))
    index = torch.cat(
        [h_index, index_separator, l_index, index_separator]
        + ag_index_with_separator,
        dim=1,
    )
    residue_index = F.pad(index, (1, 1), value=-1)[:, :MAX_LENGTH]

    current_length = combined_structure_tokens.size(1)
    if current_length < MAX_LENGTH:
        padding_length = MAX_LENGTH - current_length
        combined_structure_tokens = F.pad(
            combined_structure_tokens, (0, padding_length), value=4099
        )
        chain_ids = F.pad(chain_ids, (0, padding_length), value=-2)
        residue_index = F.pad(residue_index, (0, padding_length), value=-1)

    cdr_pos = build_cdr_positions(
        chain_ids.squeeze(0).cpu(), residue_index.squeeze(0).cpu()
    )
    interface = build_interface_mask(h_chain, l_chain, ag_chains)
    return {
        "complex_tokens": complex_tokens,
        "combined_structure_tokens": combined_structure_tokens,
        "chain_ids": chain_ids,
        "residue_index": residue_index,
        "cdr_pos": cdr_pos,
        "interface": interface,
        "h_sequence": h_sequence,
        "l_sequence": l_sequence,
        "ag_sequences": ag_sequences,
    }


def write_generated_pdb(structure_output: Any, pdb_path: Path) -> None:
    try:
        pdb_text = structure_output.to_pdb()
    except TypeError:
        structure_output.to_pdb(str(pdb_path))
        return

    if isinstance(pdb_text, str):
        pdb_path.write_text(pdb_text, encoding="utf-8")
        return
    structure_output.to_pdb(str(pdb_path))


def bind_esm_data_root(esm3_root: str | Path) -> Any:
    """Bind ESM's relative data lookup to this validated repository checkout."""
    from esm.utils.constants import esm3 as esm3_constants

    resolved_root = Path(esm3_root).resolve()
    original_data_root = esm3_constants.data_root

    def repository_data_root(model: str) -> Path:
        if model.startswith("esm3"):
            return resolved_root
        return original_data_root(model)

    esm3_constants.data_root = repository_data_root
    import esm.pretrained as esm_pretrained

    esm_pretrained.data_root = repository_data_root
    return esm_pretrained


def generate_for_sample(
    *,
    model: Any,
    tokenizer: Any,
    sample_id: str,
    sample_ctx: dict[str, Any],
    interface: Any,
    antigen_feat: Any,
    feature_source: str,
    cdr_index: int,
    cdr_mode: str,
    num_return_sequences: int,
    mini_batch_size: int,
    sequence_steps: int,
    structure_steps: int,
    schedule: str,
    temperature: float,
    device: str,
    output_dir: Path,
    h_chain_id: str,
    l_chain_id: str,
    summary: GenerationSummary,
) -> list[dict[str, Any]]:
    import torch
    from esm.sdk.api import ESMProtein, GenerationConfig
    from sft.training.masking import mask_antibody_stru_cdr, mask_seq_single_cdr

    chain_ids_cpu = sample_ctx["chain_ids"].cpu()
    cdr_pos_cpu = sample_ctx["cdr_pos"].cpu()
    complex_tokens = sample_ctx["complex_tokens"]["input_ids"]
    combined_structure_tokens = sample_ctx["combined_structure_tokens"]

    if cdr_mode == "full":
        masked_seq_tokens = complex_tokens
        for index in range(1, 7):
            masked_seq_tokens = mask_seq_single_cdr(
                masked_seq_tokens,
                cdr_pos_cpu,
                index,
                MASK_IDX,
                PAD_IDX,
                chain_ids_cpu,
                torch.full((1,), 0),
                torch.full((1,), 1),
            )
        masked_seq_tokens = masked_seq_tokens.to(device)
    else:
        masked_seq_tokens = mask_seq_single_cdr(
            complex_tokens,
            cdr_pos_cpu,
            cdr_index,
            MASK_IDX,
            PAD_IDX,
            chain_ids_cpu,
            torch.full((1,), 0),
            torch.full((1,), 1),
        ).to(device)

    mask_antibody_stru_cdr(
        combined_structure_tokens.cpu(),
        cdr_pos_cpu,
        STRUCTURE_MASK_IDX,
        SPECIAL_STRUCTURE_TOKENS,
        chain_ids_cpu,
        torch.full((1,), 0),
        torch.full((1,), 1),
    ).to(device)

    masked_seq = clean_decoded_sequence(tokenizer.decode(masked_seq_tokens[0]))
    prompt = ESMProtein(sequence=masked_seq)
    sequence_config = GenerationConfig(
        track="sequence",
        num_steps=sequence_steps,
        schedule=schedule,
        temperature=temperature,
    )
    interface_on_device = interface.to(device)
    antigen_feat_on_device = antigen_feat.to(device)

    indexed_outputs: list[tuple[int, Any]] = []
    for start in range(0, num_return_sequences, mini_batch_size):
        end = min(start + mini_batch_size, num_return_sequences)
        batch_size = end - start
        try:
            batch_outputs = model.batch_generate(
                [prompt] * batch_size,
                [sequence_config] * batch_size,
                [interface_on_device] * batch_size,
                [antigen_feat_on_device] * batch_size,
            )
        except Exception as exc:
            for sample_index in range(start, end):
                summary.record_failure(
                    sample_id, sample_index, "sequence_generation", exc
                )
            continue
        for offset, generated_protein in enumerate(batch_outputs[:batch_size]):
            indexed_outputs.append((start + offset, generated_protein))
        for sample_index in range(start + len(batch_outputs), end):
            summary.record_failure(
                sample_id,
                sample_index,
                "sequence_generation",
                "model returned no sequence",
            )

    h_pos = (sample_ctx["chain_ids"] == 0).squeeze(0).cpu()
    l_pos = (sample_ctx["chain_ids"] == 1).squeeze(0).cpu()
    antigen_seq = "|".join(sample_ctx["ag_sequences"])
    rows: list[dict[str, Any]] = []

    for sample_index, generated_protein in indexed_outputs:
        pdb_path = output_dir / f"{sample_id}_{sample_index}.pdb"
        try:
            generated_sequence = getattr(generated_protein, "sequence", None)
            if not generated_sequence:
                raise RuntimeError(str(generated_protein))
            generated_tokens = tokenizer(
                generated_sequence, return_tensors="pt"
            )["input_ids"][0]
            h_tokens = generated_tokens[h_pos[: generated_tokens.size(0)]]
            l_tokens = generated_tokens[l_pos[: generated_tokens.size(0)]]
            h_sequence = clean_decoded_sequence(tokenizer.decode(h_tokens))
            l_sequence = clean_decoded_sequence(tokenizer.decode(l_tokens))
            antibody_seq = h_sequence + "|" + l_sequence

            structure_output = model.generate(
                ESMProtein(sequence=antibody_seq),
                GenerationConfig(
                    track="structure",
                    num_steps=structure_steps,
                    schedule=schedule,
                    temperature=temperature,
                ),
                interface_on_device,
                antigen_feat_on_device,
            )
            write_generated_pdb(structure_output, pdb_path)
            summary.record_success(
                sample_id, sample_index, pdb_path, feature_source
            )
            rows.append(
                {
                    "id": f"{sample_id}_{sample_index}",
                    "antibody_seq": antibody_seq,
                    "antigen_seq": antigen_seq,
                    "h_chain_id": h_chain_id,
                    "l_chain_id": l_chain_id,
                }
            )
        except Exception as exc:
            pdb_path.unlink(missing_ok=True)
            summary.record_failure(sample_id, sample_index, "structure_generation", exc)
    return rows


def load_model(
    checkpoint_path: Path, device: str, esm3_root: Path
) -> tuple[Any, Any, Any]:
    esm_pretrained = bind_esm_data_root(esm3_root)
    import torch
    from esm.models.esm3 import ESM3
    from esm.tokenization import EsmSequenceTokenizer, get_esm3_model_tokenizers
    from esm.utils.constants.models import ESM3_OPEN_SMALL
    from sft.training.adapter_gear import ESM3Wrapper

    structure_encoder = esm_pretrained.ESM3_structure_encoder_v0
    structure_decoder = esm_pretrained.ESM3_structure_decoder_v0
    function_decoder = esm_pretrained.ESM3_function_decoder_v0
    tokenizer = EsmSequenceTokenizer()
    tokenizers = get_esm3_model_tokenizers(ESM3_OPEN_SMALL)
    encoder = structure_encoder().to(device)
    base_model = ESM3(
        d_model=1536,
        n_heads=24,
        v_heads=256,
        n_layers=48,
        structure_encoder_fn=structure_encoder,
        structure_decoder_fn=structure_decoder,
        function_decoder_fn=function_decoder,
        tokenizers=tokenizers,
    )
    model = ESM3Wrapper(base_model)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(unwrap_state_dict(state_dict), strict=False)
    model.to(device)
    model.eval()
    return model, tokenizer, encoder


def _validate_entry(entry: dict[str, Any]) -> None:
    if not entry.get("id"):
        raise ValueError("RAbD entry has an empty ID")
    for field_name in ("H_chain", "L_chain"):
        value = entry.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{entry['id']} has an invalid {field_name}")
    antigen_chains = entry.get("ag_chains")
    if (
        not isinstance(antigen_chains, list)
        or not antigen_chains
        or any(not isinstance(chain, str) or not chain.strip() for chain in antigen_chains)
    ):
        raise ValueError(f"{entry['id']} has invalid antigen chains")


def _reference_pdb_path(pdb_dir: Path, pdb_code: str) -> Path:
    candidates = [
        pdb_dir / f"{pdb_code}_cut.pdb",
        pdb_dir / f"{pdb_code}.pdb",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"reference PDB not found for {pdb_code}: expected {candidates[0]}"
    )


def _build_manifest(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path],
    asset_report: Any,
    summary: GenerationSummary,
    checkpoint_digest: str,
    csv_path: Path | None,
    completion_error: BaseException | None,
) -> dict[str, Any]:
    return {
        "status": "complete" if completion_error is None else "incomplete",
        "requested_targets": summary.requested_targets,
        "samples_per_target": summary.samples_per_target,
        "expected_designs": summary.expected_designs,
        "successful_designs": len(summary.successes),
        "failed_designs": summary.failed_designs,
        "failure_records": len(summary.failures),
        "fallback_target_ids": sorted(summary.fallback_target_ids),
        "fallback_design_ids": summary.fallback_design_ids,
        "failures": [
            {
                **asdict(failure),
                "design_id": failure.design_id,
            }
            for failure in summary.failures
        ],
        "resolved_paths": {key: str(value) for key, value in paths.items()},
        "generation_csv": str(csv_path) if csv_path is not None else None,
        "checkpoint_sha256": checkpoint_digest,
        "seed": args.seed,
        "temperature": args.temperature,
        "sequence_steps": args.sequence_steps,
        "structure_steps": args.structure_steps,
        "schedule": args.schedule,
        "cdr_mode": args.cdr_mode,
        "cdr_index": args.cdr_index,
        "mini_batch_size": args.mini_batch_size,
        "allow_missing_antigen_features": args.allow_missing_antigen_features,
        "asset_validation": asdict(asset_report),
        "completion_error": str(completion_error) if completion_error else None,
    }


def run_generation(args: argparse.Namespace) -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[2]
    paths = resolve_and_validate_paths(args, project_root)
    source_root = str(project_root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    import numpy as np
    import torch
    from eval.runtime_data import (
        load_antigen_features,
        load_rabd_entries,
        normalize_sample_id,
        validate_runtime_assets,
    )

    asset_report = validate_runtime_assets(
        json_path=paths["rabd_json"],
        pdb_dir=paths["pdb_dir"],
        feature_path=paths["antigen_features"],
        require_features=False,
    )
    entries = load_rabd_entries(paths["rabd_json"])
    entry_ids = [normalize_sample_id(str(entry["id"])) for entry in entries]
    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError("RAbD dataset contains duplicate sample IDs")
    if args.num_targets > len(entries):
        raise ValueError(
            f"requested {args.num_targets} targets, dataset contains {len(entries)}"
        )

    target_contexts: list[tuple[str, dict[str, Any], Path]] = []
    required_ids: list[str] = []
    for index in range(args.num_targets):
        entry = entries[index]
        _validate_entry(entry)
        sample_id = str(entry["id"])
        pdb_path = _reference_pdb_path(paths["pdb_dir"], str(entry["pdbcode"]))
        target_contexts.append((sample_id, entry, pdb_path))
        required_ids.append(sample_id)

    antigen_features = load_antigen_features(
        paths["antigen_features"],
        required_ids,
        allow_missing=args.allow_missing_antigen_features,
    )
    checkpoint_digest = checkpoint_sha256(paths["checkpoint"])

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = GenerationSummary(args.num_targets, args.samples_per_target)
    rows: list[dict[str, Any]] = []
    model, tokenizer, encoder = load_model(
        paths["checkpoint"],
        args.device,
        paths["esm3_weights"].parent.parent,
    )

    for sample_id, entry, pdb_path in target_contexts:
        feature, feature_source = select_antigen_feature(
            antigen_features,
            sample_id,
            allow_missing=args.allow_missing_antigen_features,
        )
        if feature_source == "zero_fallback":
            summary.record_fallback_target(sample_id)
        print(f"[{sample_id}] feature={feature_source}", flush=True)
        try:
            sample_ctx = build_structure_tokens_and_metadata(
                entry, pdb_path, tokenizer, encoder, args.device
            )
            rows.extend(
                generate_for_sample(
                    model=model,
                    tokenizer=tokenizer,
                    sample_id=sample_id,
                    sample_ctx=sample_ctx,
                    interface=sample_ctx["interface"],
                    antigen_feat=feature.float(),
                    feature_source=feature_source,
                    cdr_index=args.cdr_index,
                    cdr_mode=args.cdr_mode,
                    num_return_sequences=args.samples_per_target,
                    mini_batch_size=args.mini_batch_size,
                    sequence_steps=args.sequence_steps,
                    structure_steps=args.structure_steps,
                    schedule=args.schedule,
                    temperature=args.temperature,
                    device=args.device,
                    output_dir=output_dir,
                    h_chain_id=str(entry["H_chain"]),
                    l_chain_id=str(entry["L_chain"]),
                    summary=summary,
                )
            )
        except Exception as exc:
            summary.record_failure(sample_id, None, "target_generation", exc)
            print(f"[{sample_id}] failed: {exc}", file=sys.stderr, flush=True)

    csv_path: Path | None = None
    completion_error: BaseException | None = None
    try:
        csv_path = write_generation_csv(rows, output_dir, summary)
    except BaseException as exc:
        completion_error = exc

    manifest_path = output_dir / "generation_manifest.json"
    write_manifest(
        manifest_path,
        _build_manifest(
            args=args,
            paths=paths,
            asset_report=asset_report,
            summary=summary,
            checkpoint_digest=checkpoint_digest,
            csv_path=csv_path,
            completion_error=completion_error,
        ),
    )
    print(
        f"requested={summary.expected_designs} successful={len(summary.successes)} "
        f"failed={summary.failed_designs} fallbacks={len(summary.fallback_design_ids)}",
        flush=True,
    )
    if completion_error is not None:
        raise completion_error
    assert csv_path is not None
    return csv_path, manifest_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_generation(args)
    except Exception as exc:
        print(f"generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
