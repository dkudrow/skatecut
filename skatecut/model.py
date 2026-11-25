"""Configuration and time-based interval selection, independent of video decoding."""

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class Config:
    sample_fps: float = 10.0
    analysis_width: int = 640
    sensitivity: float = 1.0
    start_threshold: float = 0.008
    stop_ratio: float = 0.5
    min_motion: float = 0.3
    quiet_time: float = 0.6
    pre_roll: float = 1.5
    post_roll: float = 2.0
    merge_gap: float = 1.0
    min_component: float = 0.001
    background_seconds: float = 20.0
    # Normalized coordinates in the displayed (rotated) image: x, y, width, height.
    roi: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        for field in fields(self):
            name, value = field.name, getattr(self, field.name)
            if name != "roi" and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite, non-negative number")
        if not isinstance(self.analysis_width, int):
            raise ValueError("analysis_width must be an integer")
        if not 0 < self.sample_fps <= 120:
            raise ValueError("sample_fps must be between 0 and 120")
        if not 32 <= self.analysis_width <= 1920:
            raise ValueError("analysis_width must be between 32 and 1920")
        if self.sensitivity <= 0 or self.background_seconds <= 0:
            raise ValueError("sensitivity and background_seconds must be positive")
        if not 0 < self.start_threshold <= 1 or not 0 < self.stop_ratio <= 1:
            raise ValueError("start_threshold and stop_ratio must be in (0, 1]")
        if self.start_threshold / self.sensitivity > 1:
            raise ValueError("effective motion threshold must not exceed 1")
        if not 0 <= self.min_component <= 1:
            raise ValueError("min_component must be between 0 and 1")
        if self.roi is not None:
            if (not isinstance(self.roi, (tuple, list)) or len(self.roi) != 4
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               and math.isfinite(v) for v in self.roi)):
                raise ValueError("ROI must contain four finite numbers")
            x, y, width, height = self.roi
            if min(x, y) < 0 or min(width, height) <= 0 or x + width > 1 or y + height > 1:
                raise ValueError("ROI must fit inside the image using normalized coordinates")
            # JSON arrays are accepted, but configuration snapshots stay immutable.
            object.__setattr__(self, "roi", tuple(self.roi))

    @property
    def on_threshold(self) -> float:
        return self.start_threshold / self.sensitivity

    @property
    def off_threshold(self) -> float:
        return self.on_threshold * self.stop_ratio


ENCODING_PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow")


@dataclass(frozen=True)
class RenderOptions:
    crf: int = 18
    preset: str = "medium"
    attempt_counter: bool = False

    def __post_init__(self):
        if isinstance(self.crf, bool) or not isinstance(self.crf, int) or not 0 <= self.crf <= 51:
            raise ValueError("CRF must be an integer between 0 and 51")
        if self.preset not in ENCODING_PRESETS:
            raise ValueError(f"preset must be one of: {', '.join(ENCODING_PRESETS)}")
        if not isinstance(self.attempt_counter, bool):
            raise ValueError("attempt_counter must be a boolean")


@dataclass(frozen=True)
class Score:
    time: float
    motion: float


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def select_intervals(scores: list[Score], duration: float, config: Config) -> list[Interval]:
    """Require sustained starts; tolerate quiet gaps; then pad, clamp, and merge.

    Each score describes a sample bin beginning at ``time``. Motion confirmation
    counts bins (including the last bin at EOF), rather than only their gaps.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be a positive, finite number")
    step = 1 / config.sample_fps
    candidate = None
    active = None
    last_motion_end = None
    raw = []
    previous = -1.0
    for score in scores:
        if not math.isfinite(score.time) or not previous < score.time < duration:
            raise ValueError("score timestamps must increase and lie inside the video")
        if score.time < 0 or not math.isfinite(score.motion) or not 0 <= score.motion <= 1:
            raise ValueError("invalid motion score")
        previous = score.time
        bin_end = min(duration, score.time + step)
        if active is None:
            if score.motion >= config.on_threshold:
                if candidate is None:
                    candidate = score.time
                if bin_end - candidate + 1e-9 >= config.min_motion:
                    active = candidate
                    last_motion_end = bin_end
            else:
                candidate = None
        elif score.motion >= config.off_threshold:
            last_motion_end = bin_end
        elif bin_end - last_motion_end + 1e-9 >= config.quiet_time:
            raw.append(Interval(active, last_motion_end))
            active = candidate = last_motion_end = None
    if active is not None:
        raw.append(Interval(active, last_motion_end))

    merged: list[Interval] = []
    for item in raw:
        padded = Interval(max(0.0, item.start - config.pre_roll),
                          min(duration, item.end + config.post_roll))
        if merged and padded.start <= merged[-1].end + config.merge_gap + 1e-9:
            merged[-1] = Interval(merged[-1].start, max(merged[-1].end, padded.end))
        else:
            merged.append(padded)
    return merged

