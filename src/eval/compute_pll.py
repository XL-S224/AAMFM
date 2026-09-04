#!/usr/bin/env python3
"""Recompute AntiBERTy PLL metrics from an antibody generation CSV.

The default metric scores complete heavy and light chains independently and
adds their mean per-residue PLL values.  ``--mode cdr`` keeps the complete
heavy chain as model context but averages only the positions listed by the
zero-based, end-inclusive ``cdrh3_pos`` interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from tqdm import tqdm


def sum_chain_pll(heavy_pll: float, light_pll: float) -> float:
    """Return the canonical whole-antibody score without dividing by two."""
    return float(heavy_pll) + float(light_pll)


def split_antibody(antibody_sequence: str) -> tuple[str, str]:
    """Split the required ``heavy|light`` CSV representation."""
    try:
        heavy, light = str(antibody_sequence).split("|", 1)
    except ValueError as exc:
        raise ValueError("antibody_seq must use the heavy|light format") from exc
    if not heavy or not light:
        raise ValueError("antibody_seq must contain non-empty heavy|light sequences")
    return heavy, light


def target_id_from_design_id(design_id: str) -> str:
    """Remove the final sample-index suffix from a generation ID."""
    design_id = str(design_id)
    try:
        target_id, sample_index = design_id.rsplit("_", 1)
    except ValueError as exc:
        raise ValueError(f"design ID has no sample suffix: {design_id}") from exc
    if not target_id or not sample_index.isdigit():
        raise ValueError(f"design ID must end in an integer sample suffix: {design_id}")
    return target_id


def inclusive_positions(interval: Sequence[int]) -> list[int]:
    """Expand a zero-based ``[start, end]`` interval, including ``end``."""
    if len(interval) != 2:
        raise ValueError(f"cdrh3_pos must contain [start, end], got {interval}")
    start, end = (int(interval[0]), int(interval[1]))
    if start < 0 or end < start:
        raise ValueError(f"invalid cdrh3_pos interval: {interval}")
    return list(range(start, end + 1))


def _target_id_from_rabd_entry(entry: dict) -> str:
    if entry.get("id"):
        return str(entry["id"])
    pdb_code = str(entry.get("pdb", ""))
    heavy_chain = str(entry.get("heavy_chain", ""))
    light_chain = str(entry.get("light_chain", ""))
    antigen_chains = entry.get("antigen_chains", [])
    if not pdb_code or not heavy_chain or not light_chain or not antigen_chains:
        raise ValueError(f"cannot build target ID from RAbD entry: {entry}")
    return f"{pdb_code}_{heavy_chain}_{light_chain}_{''.join(antigen_chains)}"


def _read_json_records(path: Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"RAbD metadata is empty: {path}")
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"RAbD metadata must contain JSON objects: {path}")
    return records


def load_cdr_positions(rabd_json: Path) -> dict[str, list[int]]:
    """Map target IDs to zero-based CDRH3 sequence offsets from RAbD metadata."""
    positions: dict[str, list[int]] = {}
    for entry in _read_json_records(Path(rabd_json)):
        if "cdrh3_pos" not in entry:
            raise ValueError(f"RAbD entry is missing cdrh3_pos: {entry}")
        target_id = _target_id_from_rabd_entry(entry)
        if target_id in positions:
            raise ValueError(f"duplicate RAbD target ID: {target_id}")
        positions[target_id] = inclusive_positions(entry["cdrh3_pos"])
    return positions


def _parse_csv_position(value: object) -> list[int]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"cdrh3_pos must be a JSON [start, end] interval, got {value}")
    return inclusive_positions(value)


def pseudo_log_likelihood_at_positions(
    runner,
    sequence: str,
    positions: Iterable[int],
    *,
    batch_size: int,
) -> torch.Tensor:
    """Score selected residues while retaining the complete sequence context."""
    positions = [int(position) for position in positions]
    if not sequence:
        raise ValueError("sequence must not be empty")
    if not positions:
        raise ValueError("PLL positions must not be empty")
    if min(positions) < 0 or max(positions) >= len(sequence):
        raise IndexError(
            f"PLL positions [{min(positions)}, {max(positions)}] exceed sequence "
            f"length {len(sequence)}"
        )

    masked_sequences = []
    for position in positions:
        masked = (
            list(sequence[:position])
            + ["[MASK]"]
            + list(sequence[position + 1 :])
        )
        masked_sequences.append(" ".join(masked))

    tokenized = runner.tokenizer(
        masked_sequences,
        return_tensors="pt",
        padding=True,
    )
    tokens = tokenized["input_ids"].to(runner.device)
    attention_mask = tokenized["attention_mask"].to(runner.device)

    logits = []
    with torch.no_grad():
        for start in range(0, len(masked_sequences), batch_size):
            stop = min(start + batch_size, len(masked_sequences))
            output = runner.model(
                input_ids=tokens[start:stop],
                attention_mask=attention_mask[start:stop],
            )
            logits.append(output.prediction_logits)

    logits_tensor = torch.cat(logits, dim=0)
    logits_tensor[:, :, runner.tokenizer.all_special_ids] = -float("inf")
    logits_tensor = logits_tensor[:, 1:-1]
    row_indices = torch.arange(len(positions), device=runner.device)
    position_indices = torch.tensor(positions, device=runner.device)
    selected_logits = logits_tensor[row_indices, position_indices]

    labels = runner.tokenizer.encode(
        " ".join(sequence),
        return_tensors="pt",
    )[0, 1:-1].to(runner.device)
    selected_labels = labels[position_indices]
    negative_log_likelihood = torch.nn.functional.cross_entropy(
        selected_logits,
        selected_labels,
        reduction="mean",
    )
    return -negative_log_likelihood


def _load_antiberty(device: str):
    from antiberty import AntiBERTyRunner

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    use_cpu = device == "cpu" or (device == "auto" and not torch.cuda.is_available())
    if not use_cpu:
        return AntiBERTyRunner()

    original_cuda_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        return AntiBERTyRunner()
    finally:
        torch.cuda.is_available = original_cuda_is_available


def _score(
    runner,
    sequence: str,
    positions: Iterable[int],
    batch_size: int,
) -> float:
    return float(
        pseudo_log_likelihood_at_positions(
            runner,
            sequence,
            positions,
            batch_size=batch_size,
        ).item()
    )


def _write_outputs(
    frame: pd.DataFrame,
    *,
    metric: str,
    output_dir: Path,
    stem: str,
    manifest: dict,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / f"generation_with_{stem}.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    targets_path = output_dir / f"{stem}_by_target.csv"
    manifest_path = output_dir / f"{stem}_manifest.json"

    frame.to_csv(details_path, index=False)
    values = pd.to_numeric(frame[metric], errors="raise")
    summary = pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
                "count": int(values.count()),
            }
        ]
    )
    summary.to_csv(summary_path, index=False)
    target_summary = frame.groupby("target_id")[metric].agg(
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max",
        count="count",
    )
    target_summary.to_csv(targets_path)

    manifest.update(
        {
            "design_count": len(frame),
            "target_count": len(target_summary),
            "micro_mean": float(values.mean()),
            "macro_target_mean": float(target_summary["mean"].mean()),
            "details_csv": str(details_path.resolve()),
            "summary_csv": str(summary_path.resolve()),
            "target_summary_csv": str(targets_path.resolve()),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return details_path, summary_path, targets_path, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute whole-antibody or CDRH3 AntiBERTy PLL from CSV"
    )
    parser.add_argument("generation_csv", type=Path)
    parser.add_argument(
        "--mode",
        choices=("cdr",),
        default=None,
        help="Score only cdrh3_pos while retaining the full heavy-chain context",
    )
    parser.add_argument(
        "--rabd-json",
        type=Path,
        help="Metadata containing cdrh3_pos; required for cdr mode unless the CSV has that column",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not args.generation_csv.is_file():
        parser.error(f"generation CSV not found: {args.generation_csv}")

    frame = pd.read_csv(args.generation_csv)
    required_columns = {"id", "antibody_seq"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        parser.error(f"generation CSV is missing columns: {sorted(missing_columns)}")
    if frame.empty:
        parser.error("generation CSV is empty")
    if frame["id"].astype(str).duplicated().any():
        parser.error("generation CSV contains duplicate IDs")
    frame["target_id"] = frame["id"].map(target_id_from_design_id)

    output_dir = args.output_dir or args.generation_csv.parent
    runner = _load_antiberty(args.device)
    print(f"AntiBERTy device: {runner.device}", flush=True)

    if args.mode == "cdr":
        metadata_positions = None
        if "cdrh3_pos" not in frame.columns:
            if args.rabd_json is None:
                parser.error(
                    "cdr mode requires a cdrh3_pos CSV column or --rabd-json"
                )
            metadata_positions = load_cdr_positions(args.rabd_json)

        scores = []
        starts = []
        ends = []
        for _, row in tqdm(frame.iterrows(), total=len(frame), desc="cdr PLL"):
            heavy, _ = split_antibody(row["antibody_seq"])
            if metadata_positions is None:
                positions = _parse_csv_position(row["cdrh3_pos"])
            else:
                target_id = str(row["target_id"])
                if target_id not in metadata_positions:
                    raise KeyError(f"cdrh3_pos metadata missing for {target_id}")
                positions = metadata_positions[target_id]
            if positions[-1] >= len(heavy):
                raise IndexError(
                    f"cdrh3_pos {positions[0]}-{positions[-1]} exceeds heavy-chain "
                    f"length {len(heavy)} for {row['id']}"
                )
            scores.append(_score(runner, heavy, positions, args.batch_size))
            starts.append(positions[0])
            ends.append(positions[-1])

        frame["cdrh3_start"] = starts
        frame["cdrh3_end"] = ends
        frame["cdrh3_pll"] = scores
        paths = _write_outputs(
            frame,
            metric="cdrh3_pll",
            output_dir=output_dir,
            stem="cdr_pll",
            manifest={
                "mode": "cdr",
                "formula": "mean log-likelihood over zero-based end-inclusive cdrh3_pos",
                "context": "complete heavy chain",
                "light_chain_scored": False,
                "generation_csv": str(args.generation_csv.resolve()),
                "rabd_json": str(args.rabd_json.resolve()) if args.rabd_json else None,
            },
        )
    else:
        heavy_scores = []
        light_scores = []
        for _, row in tqdm(frame.iterrows(), total=len(frame), desc="antibody PLL"):
            heavy, light = split_antibody(row["antibody_seq"])
            heavy_scores.append(
                _score(runner, heavy, range(len(heavy)), args.batch_size)
            )
            light_scores.append(
                _score(runner, light, range(len(light)), args.batch_size)
            )
        frame["heavy_pll"] = heavy_scores
        frame["light_pll"] = light_scores
        frame["pll_sum"] = [
            sum_chain_pll(heavy, light)
            for heavy, light in zip(heavy_scores, light_scores)
        ]
        paths = _write_outputs(
            frame,
            metric="pll_sum",
            output_dir=output_dir,
            stem="pll",
            manifest={
                "mode": "antibody",
                "formula": "heavy_pll + light_pll (no division by two)",
                "context": "each complete antibody chain scored independently",
                "generation_csv": str(args.generation_csv.resolve()),
            },
        )

    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
