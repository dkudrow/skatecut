"""Two-pass motion analysis with a sampled background and adaptive MOG2."""

from contextlib import closing
import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .media import MediaError, VideoInfo, analysis_size, frames
from .model import Config, Score
from .progress import ProgressCallback, emit
from .sequences import Observation, Region


def roi_bounds(shape: tuple[int, int], roi):
    height, width = shape
    if roi is None:
        return 0, 0, width, height
    x, y, w, h = roi
    x0, y0 = round(x * width), round(y * height)
    x1, y1 = round((x + w) * width), round((y + h) * height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI is smaller than one analysis pixel; increase --analysis-width")
    return x0, y0, x1, y1


def foreground_fraction(mask: np.ndarray, min_component: float) -> float:
    """Ignore disconnected small regions; score remaining foreground by area."""
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    minimum = max(9, int(np.ceil(mask.size * min_component)))
    return float(areas[areas >= minimum].sum() / mask.size)


def regions_from_mask(mask: np.ndarray, min_component: float) -> tuple[Region, ...]:
    # Close small gaps between clothing/limbs before locating subjects. Motion
    # scoring deliberately retains its original mask and thresholds.
    height, width = mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (max(3, round(width * 0.02)) | 1,
                                       max(3, round(height * 0.02)) | 1))
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(joined, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    minimum = max(9, math.ceil(mask.size * min_component))
    components = sorted((s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= minimum),
                        key=lambda s: s[cv2.CC_STAT_AREA], reverse=True)[:12]
    return tuple(Region(float(s[0] / width), float(s[1] / height), float(s[2] / width),
                        float(s[3] / height), float(s[4] / mask.size)) for s in components)


def analyze(source: Path, info: VideoInfo, config: Config,
            progress: ProgressCallback | None = None,
            scratch_dir: Path | None = None,
            observation_callback: Callable[[Observation], None] | None = None) -> list[Score]:
    width, height = analysis_size(info, config.analysis_width)
    x0, y0, x1, y1 = roi_bounds((height, width), config.roi)

    def prepare(frame):
        return cv2.GaussianBlur(frame[y0:y1, x0:x1], (5, 5), 0)

    emit(progress, "background", "Sampling the recording to estimate its background…")
    # A time-distributed median avoids a blind warm-up window at the start.
    # At most 31 grayscale images are held, regardless of recording length.
    samples = []
    full_samples = [] if observation_callback is not None and config.roi is not None else None
    with closing(frames(source, width, height, min(config.sample_fps, 31 / info.duration),
                        scratch_dir=scratch_dir)) as stream:
        for frame in stream:
            samples.append(prepare(frame))
            if full_samples is not None:
                full_samples.append(cv2.GaussianBlur(frame, (5, 5), 0))
            if len(samples) >= 31:
                break
    if not samples:
        raise MediaError("No video frames could be decoded")
    background = np.median(np.stack(samples), axis=0).astype(np.uint8)
    del samples
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=max(2, round(config.background_seconds * config.sample_fps)),
        varThreshold=25, detectShadows=False
    )
    subtractor.apply(background, learningRate=1)
    full_subtractor = None
    if full_samples is not None:
        full_background = np.median(np.stack(full_samples), axis=0).astype(np.uint8)
        del full_samples
        full_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=max(2, round(config.background_seconds * config.sample_fps)),
            varThreshold=25, detectShadows=False)
        full_subtractor.apply(full_background, learningRate=1)
    learning_rate = 1 - np.exp(-1 / (config.background_seconds * config.sample_fps))
    scores = []
    emit(progress, "analysis", f"Analyzing motion at {config.sample_fps:g} fps ({width} × {height})…",
         0, info.duration, "seconds")
    next_update = 30.0
    with closing(frames(source, width, height, config.sample_fps, scratch_dir=scratch_dir)) as stream:
        for index, frame in enumerate(stream):
            time = index / config.sample_fps
            if time >= info.duration:
                break
            image = prepare(frame)
            # Compensate modest global exposure changes without following a small
            # moving subject. The median has a 50% spatial breakdown point.
            reference = subtractor.getBackgroundImage()
            shift = float(np.median(image.astype(np.int16) - reference.astype(np.int16)))
            normalized = np.clip(image.astype(np.float32) - shift, 0, 255).astype(np.uint8)
            mask = subtractor.apply(normalized, learningRate=float(learning_rate))
            motion = foreground_fraction(mask, config.min_component)
            scores.append(Score(time, motion))
            if observation_callback is not None:
                full_mask = mask
                if full_subtractor is not None:
                    full_image = cv2.GaussianBlur(frame, (5, 5), 0)
                    reference = full_subtractor.getBackgroundImage()
                    shift = float(np.median(full_image.astype(np.int16) - reference.astype(np.int16)))
                    normalized = np.clip(full_image.astype(np.float32) - shift, 0, 255).astype(np.uint8)
                    full_mask = full_subtractor.apply(normalized, learningRate=float(learning_rate))
                observation_callback(Observation(time, motion, regions_from_mask(full_mask, config.min_component),
                                                 np.count_nonzero(full_mask) / full_mask.size > 0.6))
            if time >= next_update:
                emit(progress, "analysis", f"  {time:.0f} / {info.duration:.0f} seconds analyzed",
                     time, info.duration, "seconds")
                next_update += 30
    if not scores:
        raise MediaError("No analysis frames were decoded")
    if scores[-1].time + 1 / config.sample_fps < info.duration - max(0.5, 2 / config.sample_fps):
        raise MediaError("Video ended well before its declared duration; input may be truncated")
    emit(progress, "analysis", "Motion analysis complete.", info.duration, info.duration, "seconds")
    return scores

