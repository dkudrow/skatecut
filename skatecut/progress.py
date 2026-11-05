"""Progress events shared by local and future hosted callers."""

from dataclasses import dataclass
from typing import Callable, Literal


Stage = Literal["probe", "background", "analysis", "tracking", "calibration", "selection", "render", "joining", "complete"]


@dataclass(frozen=True)
class ProgressEvent:
    stage: Stage
    message: str
    completed: float | None = None
    total: float | None = None
    unit: Literal["seconds", "clips"] | None = None


ProgressCallback = Callable[[ProgressEvent], None]


def emit(callback: ProgressCallback | None, stage: Stage, message: str,
         completed: float | None = None, total: float | None = None,
         unit: Literal["seconds", "clips"] | None = None):
    if callback is not None:
        callback(ProgressEvent(stage, message, completed, total, unit))
