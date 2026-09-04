import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from .config import SabdabConfig
from .entries import SabdabEntry


@dataclass
class StructureTask:
    id: str
    entry: SabdabEntry
    pdb_path: Path

    def to_dict(self) -> dict:
        return {"id": self.id, "entry": self.entry.as_dict(), "pdb_path": str(self.pdb_path)}


class SabdabTaskBuilder:
    def __init__(self, config: SabdabConfig) -> None:
        self.config = config

    def build_tasks(self, entries: Iterable[SabdabEntry]) -> List[StructureTask]:
        tasks: List[StructureTask] = []
        imgt_dir = Path(self.config.paths.imgt_dir)
        for entry in entries:
            pdb_path = imgt_dir / f"{entry.pdbcode}.pdb"
            tasks.append(StructureTask(id=entry.id, entry=entry, pdb_path=pdb_path))
        return tasks


class StructurePreprocessor:
    """Thin orchestrator that wires cut/preprocess callables for easier testing."""

    def __init__(
        self,
        cut_fn: Optional[Callable[[dict], str]] = None,
        preprocess_fn: Optional[Callable[[dict], Optional[dict]]] = None,
    ) -> None:
        self.cut_fn = cut_fn
        self.preprocess_fn = preprocess_fn

    def run(self, task: StructureTask) -> Optional[dict]:
        payload = task.to_dict()

        if self.cut_fn:
            payload["pdb_path"] = self.cut_fn(payload)

        if self.preprocess_fn:
            return self.preprocess_fn(payload)

        return payload

    def process_tasks(self, tasks: Iterable[StructureTask]) -> List[dict]:
        processed: List[dict] = []
        for task in tasks:
            try:
                output = self.run(task)
            except Exception as exc:
                logging.warning("Failed to preprocess task %s: %s", task.id, exc)
                continue
            if output is not None:
                processed.append(output)
        return processed
