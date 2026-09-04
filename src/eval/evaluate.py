"""Evaluate reference-guided antibody designs against native RAbD structures.

The evaluator consumes the exact artifacts emitted by ``eval.generate``:
top-level chain-A PDB files, ``generation.csv``, and (when present)
``generation_manifest.json``.  ESM structure helpers are imported only while
evaluation is running so ``--help`` and metric-only imports remain CPU-safe.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd
import torch


HYDROPHOBIC_RESIDUES = frozenset("AVLIPFMWG")
CA_CLASH_THRESHOLD = 3.0
CDR_DEFINITIONS = {
    "H": {1: (27, 38), 2: (56, 65), 3: (105, 117)},
    "L": {1: (27, 38), 2: (56, 65), 3: (105, 117)},
}

STAT_COLS_FULL = [
    "TM_score_H",
    "TM_score_L",
    "RMSD_H",
    "RMSD_L",
    "RMSD_CDR_H1",
    "RMSD_CDR_H2",
    "RMSD_CDR_H3",
    "C_RMSD_CDR_H1",
    "C_RMSD_CDR_H2",
    "C_RMSD_CDR_H3",
    "Aligned_RMSD_CDR_H1",
    "Aligned_RMSD_CDR_H2",
    "Aligned_RMSD_CDR_H3",
    "C_Aligned_RMSD_CDR_H1",
    "C_Aligned_RMSD_CDR_H2",
    "C_Aligned_RMSD_CDR_H3",
    "AAR_H1",
    "AAR_H2",
    "AAR_H3",
    "C_AAR_H1",
    "C_AAR_H2",
    "C_AAR_H3",
    "PHR_H3",
    "SeqSim_H3",
    "CN_Score_H3_Mean",
    "CN_Score_H3_Std",
    "Clashes_H3_inner",
    "Clashes_H3_outer",
    "TM_score_H3",
    "RMSD_CDR_L1",
    "RMSD_CDR_L2",
    "RMSD_CDR_L3",
    "C_RMSD_CDR_L1",
    "C_RMSD_CDR_L2",
    "C_RMSD_CDR_L3",
    "Aligned_RMSD_CDR_L1",
    "Aligned_RMSD_CDR_L2",
    "Aligned_RMSD_CDR_L3",
    "C_Aligned_RMSD_CDR_L1",
    "C_Aligned_RMSD_CDR_L2",
    "C_Aligned_RMSD_CDR_L3",
    "AAR_L1",
    "AAR_L2",
    "AAR_L3",
    "C_AAR_L1",
    "C_AAR_L2",
    "C_AAR_L3",
]

STAT_COLS_H3ONLY = [
    "TM_score_H",
    "RMSD_H",
    "RMSD_CDR_H3",
    "C_RMSD_CDR_H3",
    "Aligned_RMSD_CDR_H3",
    "C_Aligned_RMSD_CDR_H3",
    "AAR_H3",
    "C_AAR_H3",
    "PHR_H3",
    "SeqSim_H3",
    "CN_Score_H3_Mean",
    "CN_Score_H3_Std",
    "Clashes_H3_inner",
    "Clashes_H3_outer",
    "TM_score_H3",
]


@dataclass(frozen=True)
class DesignRecord:
    design_id: str
    target_id: str
    sample_index: int | None
    designed_path: Path
    original_path: Path
    heavy_chain_id: str
    light_chain_id: str
    heavy_length: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate generated antibody PDBs against native RAbD structures"
    )
    parser.add_argument("--mode", choices=["h3only", "full"], default="h3only")
    parser.add_argument("--designed-dir", required=True)
    parser.add_argument("--original-dir", default="datasets/eval/rabd/pdb")
    parser.add_argument("--usalign-executable", default="tools/USalign")
    parser.add_argument("--results-dir", required=True)
    return parser


def to_structure_encoder_inputs(
    chain: Any, should_normalize_coordinates: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a ProteinChain without importing either training dataset."""
    coordinates = torch.tensor(chain.atom37_positions, dtype=torch.float32)
    confidence = torch.tensor(chain.confidence, dtype=torch.float32)
    residue_index = torch.tensor(chain.residue_index, dtype=torch.long)
    if should_normalize_coordinates:
        finite = torch.isfinite(coordinates).all(dim=-1, keepdim=True)
        finite_coordinates = coordinates.masked_select(finite).view(-1, 3)
        if finite_coordinates.numel():
            center = finite_coordinates.mean(dim=0)
            coordinates = torch.where(finite, coordinates - center, coordinates)
    return (
        coordinates.unsqueeze(0),
        confidence.unsqueeze(0),
        residue_index.unsqueeze(0),
    )


def kabsch_align(
    reference: torch.Tensor, mobile: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rigidly align row-vector ``mobile`` coordinates onto ``reference``."""
    if reference.shape != mobile.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {mobile.shape}")
    if reference.ndim != 2 or reference.shape[1] != 3 or not reference.shape[0]:
        raise ValueError("Kabsch inputs must have shape (N, 3) with N > 0")
    if not torch.isfinite(reference).all() or not torch.isfinite(mobile).all():
        raise ValueError("Kabsch inputs must contain only finite coordinates")

    reference_center = reference.mean(dim=0)
    mobile_center = mobile.mean(dim=0)
    centered_reference = reference - reference_center
    centered_mobile = mobile - mobile_center
    u, _, vh = torch.linalg.svd(centered_mobile.T @ centered_reference)
    correction = torch.eye(3, dtype=reference.dtype, device=reference.device)
    correction[-1, -1] = torch.det(u @ vh)
    rotation = u @ correction @ vh
    translation = reference_center - mobile_center @ rotation
    return mobile @ rotation + translation, rotation, translation


def kabsch_rmsd(reference: torch.Tensor, mobile: torch.Tensor) -> float:
    aligned, _, _ = kabsch_align(reference, mobile)
    return torch.sqrt(torch.mean(torch.sum((reference - aligned) ** 2, dim=-1))).item()


def get_cdr_mask(chain_type: str, residue_indices: torch.Tensor) -> torch.Tensor:
    if residue_indices.ndim == 2:
        residue_indices = residue_indices[0]
    if residue_indices.ndim != 1:
        raise ValueError("residue indices must have shape (N,) or (1, N)")
    mask = torch.zeros_like(residue_indices, dtype=torch.long)
    for cdr_number, (start, end) in CDR_DEFINITIONS[chain_type.upper()].items():
        mask[(residue_indices >= start) & (residue_indices <= end)] = cdr_number
    return mask


def extract_cdr_coordinates(
    coordinates: torch.Tensor, cdr_mask: torch.Tensor, cdr_number: int
) -> torch.Tensor:
    if coordinates.ndim == 4:
        coordinates = coordinates[0]
    if coordinates.ndim != 3 or coordinates.shape[-2:] != (37, 3):
        raise ValueError("coordinates must have shape (N, 37, 3) or (1, N, 37, 3)")
    if coordinates.shape[0] != cdr_mask.shape[0]:
        raise ValueError(
            f"coordinate/mask length mismatch: {coordinates.shape[0]} vs {cdr_mask.shape[0]}"
        )
    selected = coordinates[cdr_mask == cdr_number]
    if not selected.shape[0]:
        raise ValueError(f"CDR{cdr_number} is absent from the reference chain")
    return selected


def extract_cdr_sequence(sequence: str, cdr_mask: torch.Tensor, cdr_number: int) -> str:
    if len(sequence) != cdr_mask.shape[0]:
        raise ValueError(
            f"sequence/mask length mismatch: {len(sequence)} vs {cdr_mask.shape[0]}"
        )
    positions = (cdr_mask == cdr_number).cpu().numpy()
    return "".join(np.asarray(list(sequence))[positions])


def rmsd_cdr(
    reference: torch.Tensor, mobile: torch.Tensor, atom_index: int = 1
) -> float:
    if reference.shape != mobile.shape:
        return float("nan")
    reference_atoms = reference[:, atom_index, :] if reference.ndim == 3 else reference
    mobile_atoms = mobile[:, atom_index, :] if mobile.ndim == 3 else mobile
    if (
        not torch.isfinite(reference_atoms).all()
        or not torch.isfinite(mobile_atoms).all()
    ):
        return float("nan")
    return torch.sqrt(
        torch.mean(torch.sum((reference_atoms - mobile_atoms) ** 2, dim=-1))
    ).item()


def aligned_rmsd_cdr(
    reference: torch.Tensor, mobile: torch.Tensor, atom_index: int = 1
) -> float:
    if reference.shape != mobile.shape:
        return float("nan")
    reference_atoms = reference[:, atom_index, :] if reference.ndim == 3 else reference
    mobile_atoms = mobile[:, atom_index, :] if mobile.ndim == 3 else mobile
    try:
        return kabsch_rmsd(reference_atoms, mobile_atoms)
    except ValueError:
        return float("nan")


def c_rmsd_cdr(
    reference: torch.Tensor, mobile: torch.Tensor, atom_index: int = 1
) -> float:
    if reference.shape != mobile.shape or reference.shape[0] < 7:
        return float("nan")
    return rmsd_cdr(reference[4:-2], mobile[4:-2], atom_index)


def c_aligned_rmsd_cdr(
    reference: torch.Tensor, mobile: torch.Tensor, atom_index: int = 1
) -> float:
    if reference.shape != mobile.shape or reference.shape[0] < 7:
        return float("nan")
    return aligned_rmsd_cdr(reference[4:-2], mobile[4:-2], atom_index)


def aar(reference: str, designed: str) -> float:
    if not reference or not designed:
        return float("nan")
    length = min(len(reference), len(designed))
    return sum(a == b for a, b in zip(reference, designed)) / length


def c_aar(reference: str, designed: str) -> float:
    if len(reference) < 7 or len(designed) < 7:
        return float("nan")
    return aar(reference[4:-2], designed[4:-2])


def compute_chain_cdr_metrics(
    *,
    chain_type: str,
    reference_coordinates: torch.Tensor,
    aligned_coordinates: torch.Tensor,
    raw_coordinates: torch.Tensor,
    reference_sequence: str,
    designed_sequence: str,
    residue_indices: torch.Tensor,
    cdr_numbers: Iterable[int],
) -> dict[str, float]:
    """Compute RMSD and sequence-recovery metrics for one antibody chain."""
    cdr_mask = get_cdr_mask(chain_type, residue_indices)
    metrics: dict[str, float] = {}
    chain_label = chain_type.upper()
    for cdr_number in cdr_numbers:
        reference_cdr = extract_cdr_coordinates(
            reference_coordinates, cdr_mask, cdr_number
        )
        aligned_cdr = extract_cdr_coordinates(
            aligned_coordinates, cdr_mask, cdr_number
        )
        raw_cdr = extract_cdr_coordinates(raw_coordinates, cdr_mask, cdr_number)
        reference_cdr_sequence = extract_cdr_sequence(
            reference_sequence, cdr_mask, cdr_number
        )
        designed_cdr_sequence = extract_cdr_sequence(
            designed_sequence, cdr_mask, cdr_number
        )
        metrics[f"RMSD_CDR_{chain_label}{cdr_number}"] = rmsd_cdr(
            reference_cdr, aligned_cdr
        )
        metrics[f"C_RMSD_CDR_{chain_label}{cdr_number}"] = c_rmsd_cdr(
            reference_cdr, aligned_cdr
        )
        metrics[f"Aligned_RMSD_CDR_{chain_label}{cdr_number}"] = aligned_rmsd_cdr(
            reference_cdr, raw_cdr
        )
        metrics[f"C_Aligned_RMSD_CDR_{chain_label}{cdr_number}"] = (
            c_aligned_rmsd_cdr(reference_cdr, raw_cdr)
        )
        metrics[f"AAR_{chain_label}{cdr_number}"] = aar(
            reference_cdr_sequence, designed_cdr_sequence
        )
        metrics[f"C_AAR_{chain_label}{cdr_number}"] = c_aar(
            reference_cdr_sequence, designed_cdr_sequence
        )
    return metrics


def calculate_phr(sequence: str) -> float:
    if not sequence:
        return float("nan")
    return sum(residue in HYDROPHOBIC_RESIDUES for residue in sequence) / len(sequence)


def calculate_seqsim(reference: str, designed: str) -> float:
    if not reference or not designed:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from Bio import pairwise2

            alignments = pairwise2.align.globalms(
                reference, designed, 1, 0, 0, 0, penalize_end_gaps=False
            )
    except (ImportError, ValueError) as exc:
        warnings.warn(f"sequence similarity unavailable: {exc}")
        return float("nan")
    if not alignments:
        return 0.0
    aligned_reference, aligned_designed, *_ = alignments[0]
    return sum(a == b for a, b in zip(aligned_reference, aligned_designed)) / len(
        aligned_reference
    )


def calculate_peptide_bond_lengths(coordinates: torch.Tensor) -> torch.Tensor:
    if coordinates.shape[0] < 2:
        return torch.empty(0)
    return torch.norm(coordinates[1:, 0, :] - coordinates[:-1, 2, :], dim=1)


def calculate_cn_score(bond_lengths: torch.Tensor) -> tuple[float, float]:
    if not bond_lengths.numel():
        return float("nan"), float("nan")
    return bond_lengths.mean().item(), bond_lengths.std().item()


def calculate_clashes_inner(cdr_coordinates: torch.Tensor) -> int:
    ca_coordinates = cdr_coordinates[:, 1, :]
    return sum(
        bool(torch.norm(ca_coordinates[i] - ca_coordinates[j]) < CA_CLASH_THRESHOLD)
        for i in range(len(ca_coordinates))
        for j in range(i + 1, len(ca_coordinates))
        if abs(i - j) > 1
    )


def calculate_clashes_outer(
    full_coordinates: torch.Tensor, cdr_mask: torch.Tensor, cdr_number: int
) -> int:
    ca_coordinates = full_coordinates[:, 1, :]
    cdr_indices = torch.where(cdr_mask == cdr_number)[0]
    framework_indices = torch.where(cdr_mask == 0)[0]
    return sum(
        bool(torch.norm(ca_coordinates[i] - ca_coordinates[j]) < CA_CLASH_THRESHOLD)
        for i in cdr_indices
        for j in framework_indices
        if abs((i - j).item()) > 1
    )


def _split_design_id(design_id: str) -> tuple[str, int | None]:
    match = re.fullmatch(r"(.+)_([0-9]+)", design_id)
    if not match:
        return design_id, None
    return match.group(1), int(match.group(2))


def _manifest_design_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("designs", "successes", "generated_designs"):
        rows = manifest.get(key)
        if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
            return rows
    return []


def _read_manifest(designed_dir: Path) -> dict[str, Any]:
    path = designed_dir / "generation_manifest.json"
    if not path.exists():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid generation manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"generation manifest must be a JSON object: {path}")
    if manifest.get("status") not in (None, "complete"):
        raise RuntimeError(
            f"generation manifest is not complete: {manifest.get('status')!r}"
        )
    return manifest


def _metadata_rows(
    designed_dir: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    csv_path = designed_dir / "generation.csv"
    if csv_path.exists():
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        if "id" not in frame.columns:
            raise ValueError("generation.csv is missing required column: id")
        return frame.to_dict(orient="records")
    return _manifest_design_rows(manifest)


def _row_value(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _resolve_original_pdb(
    original_dir: Path, pdb_code: str, row: dict[str, Any]
) -> Path:
    explicit = _row_value(row, "original_pdb", "reference_pdb", "native_pdb")
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if not explicit_path.is_absolute():
            explicit_path = original_dir / explicit_path
        candidates = [explicit_path]
    else:
        candidates = [
            original_dir / f"{pdb_code}_cut.pdb",
            original_dir / f"{pdb_code}.pdb",
        ]
        lower_code = pdb_code.lower()
        if lower_code != pdb_code:
            candidates.extend(
                [
                    original_dir / f"{lower_code}_cut.pdb",
                    original_dir / f"{lower_code}.pdb",
                ]
            )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"native PDB not found for {pdb_code}: expected {candidates[0]}"
    )


def load_design_records(
    designed_dir: Path | str, original_dir: Path | str
) -> list[DesignRecord]:
    """Load and validate the generation-to-evaluation artifact boundary."""
    designed_dir = Path(designed_dir)
    original_dir = Path(original_dir)
    pdb_paths = sorted(path for path in designed_dir.glob("*.pdb") if path.is_file())
    if not pdb_paths:
        raise FileNotFoundError(f"no generated PDB files found in {designed_dir}")

    manifest = _read_manifest(designed_dir)
    rows = _metadata_rows(designed_dir, manifest)
    pdb_by_id = {path.stem: path for path in pdb_paths}
    if len(pdb_by_id) != len(pdb_paths):
        raise RuntimeError("generated PDB files do not have unique design IDs")

    if rows:
        metadata_by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            design_id = _row_value(row, "id", "design_id")
            if not design_id:
                raise ValueError("generation metadata contains an empty design ID")
            if design_id in metadata_by_id:
                raise RuntimeError(f"duplicate generation metadata ID: {design_id}")
            metadata_by_id[design_id] = row
        if set(metadata_by_id) != set(pdb_by_id):
            missing_pdbs = sorted(set(metadata_by_id) - set(pdb_by_id))
            extra_pdbs = sorted(set(pdb_by_id) - set(metadata_by_id))
            raise RuntimeError(
                "generation.csv/PDB mismatch: "
                f"missing PDBs={missing_pdbs}, extra PDBs={extra_pdbs}"
            )
    else:
        warnings.warn(
            "generation.csv has no design metadata; falling back to filename parsing"
        )
        metadata_by_id = {design_id: {} for design_id in pdb_by_id}

    for count_key in ("expected_designs", "successful_designs"):
        count = manifest.get(count_key)
        if count is not None and int(count) != len(pdb_paths):
            raise RuntimeError(
                f"generation manifest {count_key}={count} but found {len(pdb_paths)} PDBs"
            )

    records: list[DesignRecord] = []
    for design_id in sorted(pdb_by_id):
        row = metadata_by_id[design_id]
        derived_target_id, derived_sample_index = _split_design_id(design_id)
        target_id = _row_value(row, "target_id") or derived_target_id
        sample_text = _row_value(row, "sample_index")
        sample_index = int(sample_text) if sample_text else derived_sample_index

        heavy_chain_id = _row_value(row, "h_chain_id", "heavy_chain_id", "H_chain")
        light_chain_id = _row_value(row, "l_chain_id", "light_chain_id", "L_chain")
        pdb_code = _row_value(row, "pdbcode", "pdb_code", "pdb_id")
        if not pdb_code:
            pdb_code = target_id.split("_", 1)[0]

        if not heavy_chain_id or not light_chain_id:
            target_parts = target_id.split("_")
            if len(target_parts) < 3:
                raise ValueError(
                    f"chain metadata missing for {design_id}; cannot parse compatibility filename"
                )
            heavy_chain_id = heavy_chain_id or target_parts[1]
            light_chain_id = light_chain_id or target_parts[2]

        antibody_sequence = _row_value(row, "antibody_seq", "antibody_sequence")
        heavy_length = (
            len(antibody_sequence.split("|", 1)[0]) if antibody_sequence else None
        )
        records.append(
            DesignRecord(
                design_id=design_id,
                target_id=target_id,
                sample_index=sample_index,
                designed_path=pdb_by_id[design_id],
                original_path=_resolve_original_pdb(original_dir, pdb_code, row),
                heavy_chain_id=heavy_chain_id,
                light_chain_id=light_chain_id,
                heavy_length=heavy_length,
            )
        )
    return records


def _run_usalign(
    executable: Path,
    designed_path: Path,
    original_path: Path,
    heavy_chain_id: str,
    output_stem: Path,
) -> tuple[float, float, Path]:
    command = [
        str(executable),
        str(designed_path),
        str(original_path),
        "-chain1",
        "A",
        "-chain2",
        heavy_chain_id,
        "-o",
        str(output_stem),
    ]
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=300
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"US-align timed out for {designed_path.name}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown error").strip()
        raise RuntimeError(
            f"US-align failed for {designed_path.name}: {detail}"
        ) from exc

    tm_scores = [
        float(value)
        for value in re.findall(r"TM-score=\s*([0-9]+(?:\.[0-9]+)?)", process.stdout)
    ]
    rmsd_match = re.search(r"RMSD=\s*([0-9]+(?:\.[0-9]+)?)", process.stdout)
    if not tm_scores or rmsd_match is None:
        raise RuntimeError(f"could not parse US-align metrics for {designed_path.name}")
    aligned_path = Path(f"{output_stem}.pdb")
    if not aligned_path.is_file():
        raise RuntimeError(
            f"US-align did not write aligned PDB for {designed_path.name}: {aligned_path}"
        )
    return tm_scores[-1], float(rmsd_match.group(1)), aligned_path


def _write_ca_pdb(path: Path, coordinates: torch.Tensor, chain_id: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, (x, y, z) in enumerate(coordinates[:, 1, :].cpu().numpy(), 1):
            handle.write(
                f"ATOM  {index:5d}  CA  GLY {chain_id[:1]}{index:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n"
            )
        handle.write("TER\nEND\n")


def calculate_tmscore_cdr(
    coordinates: torch.Tensor,
    reference_coordinates: torch.Tensor,
    executable: Path,
) -> float:
    with tempfile.TemporaryDirectory(prefix="aamfm-cdr-usalign-") as temp_name:
        temp_dir = Path(temp_name)
        designed_path = temp_dir / "designed.pdb"
        reference_path = temp_dir / "reference.pdb"
        _write_ca_pdb(designed_path, coordinates, "A")
        _write_ca_pdb(reference_path, reference_coordinates, "H")
        try:
            process = subprocess.run(
                [
                    str(executable),
                    str(designed_path),
                    str(reference_path),
                    "-mm",
                    "1",
                    "-fast",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=300,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            raise RuntimeError("US-align failed for CDR-H3") from exc
        scores = [
            float(value)
            for value in re.findall(
                r"TM-score=\s*([0-9]+(?:\.[0-9]+)?)", process.stdout
            )
        ]
        if not scores:
            raise RuntimeError("could not parse US-align TM-score for CDR-H3")
        return scores[-1]


def _load_chain(protein_chain_class: Any, path: Path, chain_id: str) -> Any:
    chain = protein_chain_class.from_pdb(path, chain_id)
    if not chain.sequence:
        raise ValueError(f"chain {chain_id!r} is absent from {path}")
    return chain


def _truncate_heavy_chain(
    coordinates: torch.Tensor, sequence: str, heavy_length: int | None
) -> tuple[torch.Tensor, str]:
    if heavy_length is None:
        return coordinates, sequence
    if coordinates.shape[1] < heavy_length or len(sequence) < heavy_length:
        raise ValueError(
            f"generated chain has {coordinates.shape[1]} residues; expected {heavy_length} heavy residues"
        )
    return coordinates[:, :heavy_length], sequence[:heavy_length]


def _load_generated_light_chain(
    protein_chain_class: Any,
    path: Path,
    full_chain: Any,
    full_coordinates: torch.Tensor,
    heavy_length: int,
    expected_light_length: int,
) -> tuple[torch.Tensor, str]:
    """Load generated light chain B, or split it from a single-chain output."""
    try:
        light_chain = _load_chain(protein_chain_class, path, "B")
    except (ValueError, IndexError):
        start = heavy_length
        stop = start + expected_light_length
        if full_coordinates.shape[1] < stop or len(full_chain.sequence) < stop:
            raise ValueError(
                f"generated light chain is absent from {path}; expected chain B or "
                f"{expected_light_length} residues after the heavy chain"
            )
        return full_coordinates[:, start:stop], full_chain.sequence[start:stop]

    coordinates, _, _ = to_structure_encoder_inputs(
        light_chain, should_normalize_coordinates=False
    )
    if coordinates.shape[1] != expected_light_length:
        raise ValueError(
            f"generated light-chain length mismatch for {path.name}: "
            f"expected={expected_light_length}, observed={coordinates.shape[1]}"
        )
    return coordinates, light_chain.sequence


def evaluate_record(
    record: DesignRecord,
    mode: str,
    usalign_executable: Path,
    protein_chain_class: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "design_id": record.design_id,
        "target_id": record.target_id,
        "sample_index": record.sample_index,
        "designed_file": record.designed_path.name,
        "original_file": record.original_path.name,
        "heavy_chain_id": record.heavy_chain_id,
        "light_chain_id": record.light_chain_id,
    }

    with tempfile.TemporaryDirectory(prefix="aamfm-chain-usalign-") as temp_name:
        temp_dir = Path(temp_name)
        original_heavy = _load_chain(
            protein_chain_class, record.original_path, record.heavy_chain_id
        )
        raw_full_chain = _load_chain(protein_chain_class, record.designed_path, "A")
        heavy_length = record.heavy_length or len(original_heavy.sequence)
        if len(raw_full_chain.sequence) < heavy_length:
            raise ValueError(
                f"generated chain is shorter than the native heavy chain for {record.design_id}: "
                f"native={heavy_length}, generated={len(raw_full_chain.sequence)}"
            )

        raw_heavy = raw_full_chain[:heavy_length]
        raw_heavy_path = temp_dir / f"{record.design_id}_H.pdb"
        raw_heavy.to_pdb(raw_heavy_path)
        aligned_heavy_stem = temp_dir / f"{record.design_id}_H_aligned"
        tm_score, chain_rmsd, aligned_heavy_path = _run_usalign(
            usalign_executable,
            raw_heavy_path,
            record.original_path,
            record.heavy_chain_id,
            aligned_heavy_stem,
        )
        row.update({"TM_score_H": tm_score, "RMSD_H": chain_rmsd})
        aligned_heavy = _load_chain(
            protein_chain_class, aligned_heavy_path, "A"
        )

        original_coordinates, _, original_indices = to_structure_encoder_inputs(
            original_heavy, should_normalize_coordinates=False
        )
        raw_coordinates, _, _ = to_structure_encoder_inputs(
            raw_heavy, should_normalize_coordinates=False
        )
        aligned_coordinates, _, _ = to_structure_encoder_inputs(
            aligned_heavy, should_normalize_coordinates=False
        )
        raw_sequence = raw_heavy.sequence

        expected_length = len(original_heavy.sequence)
        if raw_coordinates.shape[1] != expected_length:
            raise ValueError(
                f"heavy-chain length mismatch for {record.design_id}: "
                f"native={expected_length}, designed={raw_coordinates.shape[1]}"
            )
        if aligned_coordinates.shape[1] != expected_length:
            raise ValueError(
                f"aligned heavy-chain length mismatch for {record.design_id}: "
                f"native={expected_length}, aligned={aligned_coordinates.shape[1]}"
            )

        cdr_mask = get_cdr_mask("H", original_indices)
        cdr_numbers = [3] if mode == "h3only" else [1, 2, 3]
        for cdr_number in cdr_numbers:
            original_cdr = extract_cdr_coordinates(
                original_coordinates, cdr_mask, cdr_number
            )
            aligned_cdr = extract_cdr_coordinates(
                aligned_coordinates, cdr_mask, cdr_number
            )
            raw_cdr = extract_cdr_coordinates(raw_coordinates, cdr_mask, cdr_number)
            original_sequence = extract_cdr_sequence(
                original_heavy.sequence, cdr_mask, cdr_number
            )
            designed_sequence = extract_cdr_sequence(raw_sequence, cdr_mask, cdr_number)

            row[f"RMSD_CDR_H{cdr_number}"] = rmsd_cdr(original_cdr, aligned_cdr)
            row[f"C_RMSD_CDR_H{cdr_number}"] = c_rmsd_cdr(original_cdr, aligned_cdr)
            row[f"Aligned_RMSD_CDR_H{cdr_number}"] = aligned_rmsd_cdr(
                original_cdr, raw_cdr
            )
            row[f"C_Aligned_RMSD_CDR_H{cdr_number}"] = c_aligned_rmsd_cdr(
                original_cdr, raw_cdr
            )
            row[f"AAR_H{cdr_number}"] = aar(original_sequence, designed_sequence)
            row[f"C_AAR_H{cdr_number}"] = c_aar(original_sequence, designed_sequence)

            if cdr_number == 3:
                row["PHR_H3"] = calculate_phr(designed_sequence)
                row["SeqSim_H3"] = calculate_seqsim(
                    original_sequence, designed_sequence
                )
                bond_lengths = calculate_peptide_bond_lengths(raw_cdr)
                row["CN_Score_H3_Mean"], row["CN_Score_H3_Std"] = calculate_cn_score(
                    bond_lengths
                )
                row["Clashes_H3_inner"] = calculate_clashes_inner(raw_cdr)
                row["Clashes_H3_outer"] = calculate_clashes_outer(
                    raw_coordinates[0], cdr_mask, 3
                )
                row["TM_score_H3"] = calculate_tmscore_cdr(
                    aligned_cdr, original_cdr, usalign_executable
                )

        if mode == "full":
            original_light = _load_chain(
                protein_chain_class, record.original_path, record.light_chain_id
            )
            original_light_coordinates, _, original_light_indices = (
                to_structure_encoder_inputs(
                    original_light, should_normalize_coordinates=False
                )
            )
            expected_light_length = len(original_light.sequence)
            light_stop = heavy_length + expected_light_length
            if len(raw_full_chain.sequence) < light_stop:
                raise ValueError(
                    f"generated chain is shorter than the native antibody for {record.design_id}: "
                    f"expected={light_stop}, generated={len(raw_full_chain.sequence)}"
                )
            raw_light = raw_full_chain[heavy_length:light_stop]
            raw_light_path = temp_dir / f"{record.design_id}_L.pdb"
            raw_light.to_pdb(raw_light_path)
            aligned_light_stem = temp_dir / f"{record.design_id}_L_aligned"
            light_tm_score, light_chain_rmsd, aligned_light_path = _run_usalign(
                usalign_executable,
                raw_light_path,
                record.original_path,
                record.light_chain_id,
                aligned_light_stem,
            )
            row.update(
                {"TM_score_L": light_tm_score, "RMSD_L": light_chain_rmsd}
            )
            aligned_light = _load_chain(
                protein_chain_class, aligned_light_path, "A"
            )
            raw_light_coordinates, _, _ = to_structure_encoder_inputs(
                raw_light, should_normalize_coordinates=False
            )
            aligned_light_coordinates, _, _ = to_structure_encoder_inputs(
                aligned_light, should_normalize_coordinates=False
            )
            row.update(
                compute_chain_cdr_metrics(
                    chain_type="L",
                    reference_coordinates=original_light_coordinates,
                    aligned_coordinates=aligned_light_coordinates,
                    raw_coordinates=raw_light_coordinates,
                    reference_sequence=original_light.sequence,
                    designed_sequence=raw_light.sequence,
                    residue_indices=original_light_indices,
                    cdr_numbers=(1, 2, 3),
                )
            )
    return row


def require_complete_results(
    results: pd.DataFrame, designed_paths: Iterable[Path], mode: str
) -> None:
    expected_files = [Path(path).name for path in designed_paths]
    if len(results) != len(expected_files):
        raise RuntimeError(
            f"expected one result per generated PDB: {len(expected_files)} PDBs, "
            f"{len(results)} result rows"
        )
    if "designed_file" not in results:
        raise RuntimeError("results are missing designed_file")
    actual_files = results["designed_file"].astype(str).tolist()
    if len(set(actual_files)) != len(actual_files) or set(actual_files) != set(
        expected_files
    ):
        raise RuntimeError(
            "result/generated PDB mismatch: "
            f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
        )
    if mode == "h3only":
        if "RMSD_CDR_H3" not in results:
            raise RuntimeError("results are missing required RMSD_CDR_H3")
        values = pd.to_numeric(results["RMSD_CDR_H3"], errors="coerce")
        invalid = ~np.isfinite(values.to_numpy(dtype=float))
        if invalid.any():
            bad_files = results.loc[invalid, "designed_file"].astype(str).tolist()
            raise RuntimeError(
                "h3only evaluation produced non-finite RMSD_CDR_H3 for: "
                + ", ".join(bad_files)
            )


def summarize_rmsd_by_target(results: pd.DataFrame) -> pd.DataFrame:
    required = {"target_id", "RMSD_CDR_H3"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"target summary missing columns: {sorted(missing)}")
    frame = results.copy()
    frame["RMSD_CDR_H3"] = pd.to_numeric(frame["RMSD_CDR_H3"], errors="raise")
    return frame.groupby("target_id")["RMSD_CDR_H3"].agg(
        mean_rmsd_h3="mean",
        median_rmsd_h3="median",
        best_rmsd_h3="min",
        worst_rmsd_h3="max",
    )


def build_statistics(
    results: pd.DataFrame, stat_columns: Iterable[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in stat_columns:
        if column not in results:
            continue
        values = pd.to_numeric(results[column], errors="coerce")
        rows.append(
            {
                "Metric": column,
                "Mean": values.mean(skipna=True),
                "Std": values.std(skipna=True),
                "Median": values.median(skipna=True),
                "Min": values.min(skipna=True),
                "Max": values.max(skipna=True),
                "Count": values.count(),
            }
        )

    if "RMSD_CDR_H3" in results:
        rmsd = pd.to_numeric(results["RMSD_CDR_H3"], errors="coerce").dropna()
        for threshold in (1, 2, 3, 4):
            fraction = float((rmsd <= threshold).mean()) if len(rmsd) else float("nan")
            rows.append(
                {
                    "Metric": f"RMSD_CDR_H3_fraction_le_{threshold}A",
                    "Mean": fraction,
                    "Std": 0.0,
                    "Median": fraction,
                    "Min": fraction,
                    "Max": fraction,
                    "Count": len(rmsd),
                }
            )
    return pd.DataFrame(
        rows, columns=["Metric", "Mean", "Std", "Median", "Min", "Max", "Count"]
    )


def _resolve_cli_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def run_evaluation(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    project_root = Path(__file__).resolve().parents[2]
    usalign_executable = _resolve_cli_path(args.usalign_executable, project_root)
    if not usalign_executable.is_file():
        raise FileNotFoundError(f"US-align executable not found: {usalign_executable}")
    if not os.access(usalign_executable, os.X_OK):
        raise PermissionError(
            f"US-align executable is not executable: {usalign_executable}"
        )

    designed_dir = _resolve_cli_path(args.designed_dir, project_root)
    original_dir = _resolve_cli_path(args.original_dir, project_root)
    results_dir = _resolve_cli_path(args.results_dir, project_root)
    if not designed_dir.is_dir():
        raise FileNotFoundError(f"designed directory not found: {designed_dir}")
    if not original_dir.is_dir():
        raise FileNotFoundError(f"original directory not found: {original_dir}")
    if results_dir.exists() and not results_dir.is_dir():
        raise NotADirectoryError(f"results directory is not a directory: {results_dir}")

    records = load_design_records(designed_dir, original_dir)
    try:
        from esm.utils.structure.protein_chain import ProteinChain
    except ImportError as exc:
        raise ImportError(
            "ESM ProteinChain is required for structure evaluation"
        ) from exc

    all_results: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        print(
            f"[{index}/{len(records)}] evaluating {record.design_id}",
            flush=True,
        )
        try:
            all_results.append(
                evaluate_record(
                    record,
                    mode=args.mode,
                    usalign_executable=usalign_executable,
                    protein_chain_class=ProteinChain,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"evaluation failed for {record.design_id}: {exc}"
            ) from exc

    results = pd.DataFrame(all_results)
    require_complete_results(
        results, [record.designed_path for record in records], args.mode
    )
    stat_columns = STAT_COLS_FULL if args.mode == "full" else STAT_COLS_H3ONLY
    statistics = build_statistics(results, stat_columns)
    target_summary = summarize_rmsd_by_target(results)

    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"results_{args.mode}.csv"
    statistics_path = results_dir / f"statistics_{args.mode}.csv"
    target_summary_path = results_dir / "rmsd_by_target.csv"
    results.to_csv(results_path, index=False, float_format="%.4f")
    statistics.to_csv(statistics_path, index=False, float_format="%.4f")
    target_summary.to_csv(target_summary_path, index=True, float_format="%.4f")

    print(f"Results -> {results_path}")
    print(f"Statistics -> {statistics_path}")
    print(f"Target RMSD -> {target_summary_path}")
    return results_path, statistics_path, target_summary_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_evaluation(args)
    except Exception as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
