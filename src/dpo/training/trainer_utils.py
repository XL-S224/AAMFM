from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from accelerate import Accelerator


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    step_in_epoch: int = 0
    best_metric: Optional[float] = None

    @classmethod
    def from_json(cls, path: Path) -> "TrainerState":
        data = json.loads(path.read_text())
        return cls(**data)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))


class CheckpointManager:
    def __init__(
        self,
        output_dir: str,
        accelerator: Accelerator,
        metric_name: str,
        greater_is_better: bool,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.accelerator = accelerator
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_dir(self, tag: str) -> Path:
        return self.output_dir / tag

    def save(self, tag: str, state: TrainerState) -> None:
        ckpt_dir = self._checkpoint_dir(tag)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.save_state(str(ckpt_dir))
        if self.accelerator.is_main_process:
            state.to_json(ckpt_dir / "trainer_state.json")
        self.accelerator.wait_for_everyone()

    def load(self, path: str, state: TrainerState) -> TrainerState:
        ckpt_dir = Path(path)
        self.accelerator.load_state(str(ckpt_dir))
        state_path = ckpt_dir / "trainer_state.json"
        if state_path.exists():
            return TrainerState.from_json(state_path)
        return state

    def best_dir(self) -> Path:
        return self._checkpoint_dir("best")

    def has_best(self) -> bool:
        return (self.best_dir() / "trainer_state.json").is_file()

    def is_better(self, new_value: float, best_value: Optional[float]) -> bool:
        if best_value is None:
            return True
        if self.greater_is_better:
            return new_value > best_value
        return new_value < best_value
