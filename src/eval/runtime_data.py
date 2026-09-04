"""Small, dependency-light readers used by evaluation runtimes.

This module deliberately contains no model imports (in particular, no ESM or
CUDA initialization).  Evaluation entrypoints can therefore inspect assets in
CPU-only processes.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch


def normalize_sample_id(value: str) -> str:
    """Return a canonical sample identifier while retaining chain suffixes."""
    if not isinstance(value, str):
        raise TypeError("sample id must be a string")
    return value.strip().lower()


def pdb_id_from_sample_id(value: str) -> str:
    """Extract the PDB portion from either a bare or chain-qualified id."""
    return normalize_sample_id(value).split("_", 1)[0]


def load_rabd_entries(path: Path) -> list[dict]:
    """Read RAbD JSONL and derive the compact entries consumed at runtime."""
    entries: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            pdb = str(record["pdb"])
            heavy = record.get("heavy_chain", "")
            light = record.get("light_chain", "")
            antigen = record.get("antigen_chains", [])
            entry_id = f"{pdb}_{heavy}_{light}_{''.join(antigen)}"
            entries.append(
                {
                    "id": entry_id,
                    "pdbcode": pdb,
                    "H_chain": heavy,
                    "L_chain": light,
                    "ag_chains": antigen,
                }
            )
    return entries


def _feature_records(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("features", payload))
    if isinstance(payload, dict):
        records = []
        for key, value in payload.items():
            if isinstance(value, dict):
                record = dict(value)
                record.setdefault("pdbid", key)
                records.append(record)
            else:
                records.append({"pdbid": key, "node_feature": value})
        return records
    if not isinstance(payload, (list, tuple)):
        raise ValueError("feature payload must be a list or dictionary")
    return list(payload)


def load_antigen_features(
    path: Path, required_ids: list[str], allow_missing: bool = False
) -> dict[str, torch.Tensor]:
    """Load compact antigen features, resolving full and bare identifiers.

    Missing requested IDs are an error unless ``allow_missing`` is explicitly
    enabled for an evaluation fallback.
    """
    payload = torch.load(path, map_location="cpu")
    records = _feature_records(payload)
    full: dict[str, torch.Tensor] = {}
    by_pdb: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for record in records:
        if not isinstance(record, dict) or "node_feature" not in record:
            raise ValueError("each feature record requires node_feature")
        raw_id = record.get("pdbid", record.get("id"))
        if raw_id is None:
            raise ValueError("each feature record requires pdbid")
        feature = torch.as_tensor(record["node_feature"])
        if feature.ndim == 0 or feature.shape[-1] != 3072:
            raise ValueError("antigen feature final dimension 3072 required")
        identifier = normalize_sample_id(str(raw_id))
        full[identifier] = feature
        by_pdb.setdefault(pdb_id_from_sample_id(identifier), []).append((identifier, feature))

    result: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for requested in required_ids:
        key = normalize_sample_id(requested)
        if key in full:
            result[requested] = full[key]
            continue
        matches = by_pdb.get(pdb_id_from_sample_id(key), [])
        if len(matches) > 1:
            raise ValueError(f"ambiguous bare PDB id: {requested}")
        if matches:
            result[requested] = matches[0][1]
        else:
            missing.append(str(requested))
    if missing and not allow_missing:
        raise ValueError(f"missing antigen features: {', '.join(missing)}")
    return result


@dataclass(frozen=True)
class AssetValidationReport:
    entry_count: int
    pdb_count: int
    feature_count: int
    missing_feature_ids: tuple[str, ...]


def validate_runtime_assets(
    json_path: Path,
    pdb_dir: Path,
    feature_path: Path,
    require_features: bool = True,
) -> AssetValidationReport:
    entries = load_rabd_entries(json_path)
    pdb_count = sum(1 for path in Path(pdb_dir).rglob("*.pdb") if path.is_file())
    requested = [entry["id"] for entry in entries]
    records = _feature_records(torch.load(feature_path, map_location="cpu"))
    feature_ids = [str(item.get("pdbid", item.get("id", ""))) for item in records if isinstance(item, dict)]
    available = {normalize_sample_id(value) for value in feature_ids}
    available_bare = {pdb_id_from_sample_id(value) for value in available}
    missing = tuple(
        entry_id for entry_id in requested
        if normalize_sample_id(entry_id) not in available
        and pdb_id_from_sample_id(entry_id) not in available_bare
    )
    if missing and require_features:
        raise ValueError(f"missing antigen features: {', '.join(missing)}")
    return AssetValidationReport(
        entry_count=len(entries),
        pdb_count=pdb_count,
        feature_count=len(records),
        missing_feature_ids=missing,
    )
