import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import SabdabConfig
from .utils import (
    build_entry_id,
    nan_to_empty_string,
    nan_to_none,
    parse_date,
    parse_sabdab_resolution,
    split_sabdab_delimited_str,
)


@dataclass
class SabdabEntry:
    id: str
    pdbcode: str
    H_chain: Optional[str]
    L_chain: Optional[str]
    ag_chains: List[str]
    ag_type: Optional[str]
    ag_name: Optional[str]
    date: Optional[pd.Timestamp]
    resolution: Optional[float]
    method: Optional[str]
    scfv: Optional[str]

    def as_dict(self) -> dict:
        return asdict(self)


class SabdabEntryLoader:
    def __init__(self, config: SabdabConfig) -> None:
        self.config = config

    def _row_to_entry(self, row: dict) -> SabdabEntry:
        ag_chains = split_sabdab_delimited_str(nan_to_empty_string(row.get("antigen_chain")))
        h_chain = nan_to_empty_string(row.get("Hchain"))
        l_chain = nan_to_empty_string(row.get("Lchain"))
        entry_id = build_entry_id(row["pdb"], h_chain, l_chain, ag_chains)

        return SabdabEntry(
            id=entry_id,
            pdbcode=row["pdb"],
            H_chain=nan_to_none(row.get("Hchain")),
            L_chain=nan_to_none(row.get("Lchain")),
            ag_chains=ag_chains,
            ag_type=nan_to_none(row.get("antigen_type")),
            ag_name=nan_to_none(row.get("antigen_name")),
            date=parse_date(row.get("date")),
            resolution=parse_sabdab_resolution(row.get("resolution")),
            method=row.get("method"),
            scfv=row.get("scfv"),
        )

    def _keep_entry(self, entry: SabdabEntry) -> bool:
        cfg = self.config.filtering
        return (
            (entry.ag_type in cfg.allowed_antigen_types or entry.ag_type is None)
            and entry.resolution is not None
            and entry.resolution <= cfg.resolution_threshold
        )

    def load(self, summary_path: Optional[str] = None) -> List[SabdabEntry]:
        path = Path(summary_path or self.config.paths.summary_path)
        if not path.exists():
            raise FileNotFoundError(f"Summary file not found: {path}")

        df = pd.read_csv(path, sep="\t")
        entries: List[SabdabEntry] = []

        for _, row in df.iterrows():
            entry = self._row_to_entry(row)
            if self._keep_entry(entry):
                entries.append(entry)
        logging.info("Loaded %d entries from %s", len(entries), path)
        return entries
