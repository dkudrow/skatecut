"""Spatial motion-sequence observations and deterministic edit planning.

All geometry is normalized to the displayed, full video frame. No model weights,
network access, video decoding, or terminal behavior live in this module.
"""

from dataclasses import asdict, dataclass, replace
import math
from statistics import median
from typing import Literal

from .model import Config, Interval, Score, select_intervals

Side = Literal["left", "right"]


@dataclass(frozen=True)
class SequenceFilterOptions:
    approach_from: Side
    attempt_duration: float | None = None
    same_side: Literal["trim", "keep", "omit"] = "trim"
    trick_roi: tuple[float, float, float, float] | None = None
    min_references: int = 5

    def __post_init__(self):
        if self.approach_from not in ("left", "right"):
            raise ValueError("approach_from must be left or right")
        if self.same_side not in ("trim", "keep", "omit"):
            raise ValueError("same_side must be trim, keep, or omit")
        if self.attempt_duration is not None:
            value = self.attempt_duration
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError("attempt_duration must be a positive, finite number")
        if (isinstance(self.min_references, bool) or not isinstance(self.min_references, int)
                or self.min_references < 3):
            raise ValueError("min_references must be an integer of at least 3")
        if self.trick_roi is not None:
            # Share coordinate validation with the motion ROI, not its meaning.
            object.__setattr__(self, "trick_roi", Config(roi=self.trick_roi).roi)


@dataclass(frozen=True)
class Region:
    x: float
    y: float
    width: float
    height: float
    area: float

    @property
    def cx(self):
        return self.x + self.width / 2

    @property
    def cy(self):
        return self.y + self.height / 2


@dataclass(frozen=True)
class Observation:
    time: float
    motion: float
    regions: tuple[Region, ...]
    scene_change: bool = False


@dataclass(frozen=True)
class TrackPoint:
    time: float
    region: Region
    motion: float


@dataclass(frozen=True)
class MotionSequence:
    sequence_id: int
    start: float
    end: float
    entry_side: Side | None
    exit_side: Side | None
    entry_time: float | None
    exit_time: float | None
    entry_origin: Literal["edge", "staging", "unknown"]
    active_duration: float
    pause_duration: float
    trajectory: tuple[TrackPoint, ...]
    quality_flags: tuple[str, ...] = ()

    @property
    def duration(self):
        return self.end - self.start


@dataclass(frozen=True)
class DurationModel:
    source: Literal["learned", "manual", "unavailable"]
    reliable: bool
    target_seconds: float | None
    median_seconds: float | None
    robust_spread_seconds: float | None
    reference_ids: tuple[int, ...]
    excluded_outlier_ids: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class SequenceDecision:
    sequence_id: int
    classification: str
    action: Literal["keep", "trim", "omit"]
    reason: str
    estimated_end: float | None
    retained_intervals: tuple[Interval, ...]
    max_progress: float
    return_start: float | None
    visited_trick_roi: bool | None


@dataclass(frozen=True)
class SequencePlan:
    sequences: tuple[MotionSequence, ...]
    decisions: tuple[SequenceDecision, ...]
    duration_model: DurationModel
    intervals: tuple[Interval, ...]


def _iou(a: Region, b: Region) -> float:
    width = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    height = max(0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    intersection = width * height
    return intersection / max(1e-9, a.width * a.height + b.width * b.height - intersection)


def _distance(a: Region, b: Region) -> float:
    return math.hypot(a.cx - b.cx, a.cy - b.cy)


def _seed(regions: tuple[Region, ...]) -> Region | None:
    # Small detached boards and leaves must not become primary subjects.
    candidates = [r for r in regions if r.area >= 0.002 and r.height >= 0.15]
    return max(candidates, key=lambda r: r.area * min(r.height, 0.6), default=None)


def _match(points: list[TrackPoint], observation: Observation) -> Region | None:
    previous = points[-1]
    elapsed = observation.time - previous.time
    vx = vy = 0.0
    if len(points) >= 2:
        earlier = points[max(0, len(points) - 4)]
        dt = max(0.01, previous.time - earlier.time)
        vx = (previous.region.cx - earlier.region.cx) / dt
        vy = (previous.region.cy - earlier.region.cy) / dt
    # Extrapolate only briefly; long extrapolation is unreliable around a fall.
    predicted_x = previous.region.cx + vx * min(elapsed, 0.25)
    predicted_y = previous.region.cy + vy * min(elapsed, 0.25)
    matches = []
    for region in observation.regions:
        ratio = region.area / max(previous.region.area, 1e-9)
        distance = math.hypot(region.cx - predicted_x, region.cy - predicted_y)
        if (distance > min(0.32, 0.12 + elapsed * 0.45) or ratio < 0.16 or ratio > 7):
            continue
        cost = distance + 0.035 * abs(math.log(ratio)) - 0.08 * _iou(previous.region, region)
        matches.append((cost, region))
    return min(matches, key=lambda item: item[0])[1] if matches else None


def _edge_event(points: tuple[TrackPoint, ...], entry: bool, step: float) -> tuple[Side | None, float | None]:
    if len(points) < 3:
        return None, None
    window = points[:max(3, math.ceil(0.6 / step))] if entry else points[-max(3, math.ceil(0.6 / step)):]
    if window[-1].time - window[0].time < 0.15:
        return None, None
    delta = window[-1].region.cx - window[0].region.cx
    boundary = points[0] if entry else points[-1]
    # A bounding box touching an edge is insufficient: require inward/outward travel.
    if boundary.region.x <= 0.12 and ((delta >= 0.025) if entry else (delta <= -0.025)):
        return "left", boundary.time if entry else boundary.time + step
    if boundary.region.x + boundary.region.width >= 0.88 and ((delta <= -0.025) if entry else (delta >= 0.025)):
        return "right", boundary.time if entry else boundary.time + step
    return None, None


def _make_sequence(points: list[TrackPoint], flags: set[str], duration: float,
                   config: Config, *, exit_observed: bool) -> MotionSequence:
    step = 1 / config.sample_fps
    trajectory = tuple(points)
    entry_side, entry_time = _edge_event(trajectory, True, step)
    exit_side, exit_time = _edge_event(trajectory, False, step) if exit_observed else (None, None)
    flags = set(flags)
    if points[0].time < step / 2:
        flags.add("recording_start")
        entry_side = entry_time = None
    if points[-1].time + step >= duration - step / 2:
        flags.add("recording_end")
        exit_side = exit_time = None
    active = sum(step for point in points if point.motion >= config.off_threshold)
    # Horizontal pauses are diagnostic; they do not end a sequence.
    pauses = sum(min(b.time - a.time, step) for a, b in zip(points, points[1:])
                 if abs(b.region.cx - a.region.cx) < 0.003)
    end = min(duration, points[-1].time + step)
    return MotionSequence(0, points[0].time, end, entry_side, exit_side, entry_time,
                          exit_time, "edge" if entry_side else "unknown",
                          min(end - points[0].time, active), pauses, trajectory, tuple(sorted(flags)))


def _progress(point: TrackPoint, options: SequenceFilterOptions) -> float:
    return point.region.cx if options.approach_from == "left" else 1 - point.region.cx


def _split_staging(sequence: MotionSequence, config: Config,
                   options: SequenceFilterOptions) -> list[MotionSequence]:
    """Split a new inward approach after a return and pause in visible staging."""
    points = sequence.trajectory
    if not points:
        return [sequence]
    step = 1 / config.sample_fps
    values = [_progress(p, options) for p in points]
    maximum = values[0]
    stationary_start = None
    restart = None
    for i in range(1, len(points)):
        maximum = max(maximum, values[i])
        near_staging = values[i] <= 0.25 and maximum - values[i] >= 0.18
        if near_staging and abs(values[i] - values[i - 1]) < 0.004:
            if stationary_start is None:
                stationary_start = i - 1
        elif stationary_start is not None:
            if (near_staging and points[i - 1].time - points[stationary_start].time >= config.quiet_time):
                restart = i - 1
            stationary_start = None
        if restart is not None:
            if values[i] - values[restart] >= 0.06 and points[i].time - points[restart].time >= 0.2:
                left_points, right_points = points[:restart], points[restart:]
                if len(left_points) < 3:
                    break
                left = replace(sequence, end=right_points[0].time, exit_side=None, exit_time=None,
                               trajectory=left_points,
                               active_duration=sum(step for p in left_points if p.motion >= config.off_threshold),
                               pause_duration=min(sequence.pause_duration, right_points[0].time - sequence.start))
                right = _make_sequence(list(right_points), set(sequence.quality_flags) - {"recording_start"},
                                       sequence.end, config, exit_observed=sequence.exit_side is not None)
                right = replace(right, entry_side=options.approach_from,
                                entry_time=right.start, entry_origin="staging",
                                exit_side=sequence.exit_side, exit_time=sequence.exit_time,
                                quality_flags=tuple(f for f in right.quality_flags if f != "recording_end")
                                + tuple(f for f in sequence.quality_flags if f == "recording_end"))
                return [left, *_split_staging(right, config, options)]
            if values[i] < values[restart] - 0.03:
                restart = None
    return [sequence]


def build_sequences(observations: tuple[Observation, ...], duration: float,
                    config: Config, options: SequenceFilterOptions) -> tuple[MotionSequence, ...]:
    """Track a dominant foreground region, retaining ambiguous/missed activity later."""
    sequences = []
    points: list[TrackPoint] = []
    flags: set[str] = set()
    step = 1 / config.sample_fps
    max_gap = max(0.75, 2 * step)

    def finish(exit_observed):
        nonlocal points, flags
        if points and points[-1].time + step - points[0].time + 1e-9 >= config.min_motion:
            activity = select_intervals([Score(p.time, p.motion) for p in points], duration,
                                        replace(config, pre_roll=0, post_roll=0, merge_gap=0))
            if activity:
                sequences.append(_make_sequence(points, flags, duration, config, exit_observed=exit_observed))
        points, flags = [], set()

    for observation in observations:
        if points:
            gap = observation.time - points[-1].time
            edge_side, _ = _edge_event(tuple(points), False, step)
            if gap > max_gap:
                if edge_side is None:
                    flags.add("unobserved_departure")
                finish(True)
            elif gap > 1.5 * step and edge_side is not None:
                # An actual disappearance at the edge separates passes, including
                # a quick return before the usual motion quiet-time expires.
                if _seed(observation.regions) is not None:
                    finish(True)
        region = _match(points, observation) if points else _seed(observation.regions)
        challenger = _seed(observation.regions)
        if points and challenger is not None:
            current = region or points[-1].region
            if challenger.area > current.area * 3 and _distance(challenger, current) > 0.15:
                # A small background patch must not monopolize the primary track
                # when a substantial subject enters elsewhere.
                flags.add("subject_change")
                finish(False)
                region = challenger
        if region is None:
            continue
        if points and observation.time - points[-1].time > max(0.3, 2 * step):
            flags.add("tracking_gap")
        others = [r for r in observation.regions if r is not region
                  and r.area >= max(0.015, region.area * 0.8) and r.height >= 0.2
                  and abs(r.cx - region.cx) > 0.25 and _iou(r, region) == 0]
        if others:
            flags.add("multiple_subjects")
        if observation.scene_change:
            flags.add("scene_change")
        points.append(TrackPoint(observation.time, region, observation.motion))
    if points:
        finish(duration - points[-1].time - step >= 0.5)
    return tuple(replace(item, sequence_id=i + 1) for i, item in enumerate(sequences))


def _metrics(sequence: MotionSequence, options: SequenceFilterOptions):
    points = sequence.trajectory
    if not points:
        return 0.0, 0.0, None, False, None
    values = [_progress(p, options) for p in points]
    # A short median suppresses bounding-box jitter and detached limb motion.
    smooth = [median(values[max(0, i - 1):min(len(values), i + 2)]) for i in range(len(values))]
    maximum = max(smooth)
    peak_index = smooth.index(maximum)
    peak = smooth[0]
    backwards = 0.0
    for value in smooth:
        peak = max(peak, value)
        backwards = max(backwards, peak - value)
    return_start = None
    for i in range(peak_index + 1, len(points)):
        if (maximum - smooth[i] >= 0.12 and maximum - smooth[-1] >= 0.18
                and points[i].time - points[peak_index].time >= 0.5):
            return_start = points[peak_index].time
            break
    multiple_approaches = False
    running_peak, trough, retreating = smooth[0], smooth[0], False
    for value in smooth[1:]:
        if running_peak - value >= 0.18:
            retreating = True
            trough = min(trough, value)
        if retreating and value - trough >= 0.18:
            multiple_approaches = True
        running_peak = max(running_peak, value)
        if not retreating:
            trough = value
    visited = None
    if options.trick_roi is not None:
        x, y, w, h = options.trick_roi
        visited = any(p.region.x < x + w and p.region.x + p.region.width > x
                      and p.region.y < y + h and p.region.y + p.region.height > y for p in points)
    return maximum, backwards, return_start, multiple_approaches, visited


def learn_duration(sequences: tuple[MotionSequence, ...], options: SequenceFilterOptions) -> DurationModel:
    opposite = "right" if options.approach_from == "left" else "left"
    references = []
    for sequence in sequences:
        _, backwards, _, multiple, visited = _metrics(sequence, options)
        if (sequence.entry_side == options.approach_from and sequence.exit_side == opposite
                and sequence.entry_origin == "edge" and not sequence.quality_flags
                and not multiple and backwards <= 0.15 and visited is not False
                and sequence.pause_duration <= sequence.duration * 0.35):
            references.append((sequence.sequence_id, sequence.exit_time - sequence.entry_time))
    center = median(d for _, d in references) if references else None
    spread = 1.4826 * median(abs(d - center) for _, d in references) if references else None
    excluded = ()
    if references:
        cutoff = max(1.0, 3 * spread)
        excluded = tuple(i for i, d in references if abs(d - center) > cutoff)
        references = [(i, d) for i, d in references if i not in excluded]
        center = median(d for _, d in references)
        spread = 1.4826 * median(abs(d - center) for _, d in references)
    ids = tuple(i for i, _ in references)
    if options.attempt_duration is not None:
        return DurationModel("manual", True, options.attempt_duration, center, spread, ids, excluded,
                             "manual_duration_override")
    if len(references) < options.min_references:
        return DurationModel("unavailable", False, None, center, spread, ids, excluded,
                             "insufficient_reference_passes")
    if spread / center > 0.35:
        return DurationModel("unavailable", False, None, center, spread, ids, excluded,
                             "reference_durations_too_variable")
    target = center + max(0.5, 2 * spread)
    return DurationModel("learned", True, target, center, spread, ids, excluded,
                         "robust_reference_duration")


def subtract_intervals(intervals, barriers):
    result = list(intervals)
    for barrier in sorted(barriers, key=lambda x: x.start):
        remaining = []
        for item in result:
            if item.end <= barrier.start or item.start >= barrier.end:
                remaining.append(item)
            else:
                if item.start < barrier.start:
                    remaining.append(Interval(item.start, barrier.start))
                if item.end > barrier.end:
                    remaining.append(Interval(barrier.end, item.end))
        result = remaining
    return result


def _merge(intervals, barriers, gap):
    result = []
    for interval in sorted(intervals, key=lambda i: (i.start, i.end)):
        if interval.end - interval.start <= 1e-9:
            continue
        crosses = result and any(b.start < interval.start and b.end > result[-1].end for b in barriers)
        if result and interval.start <= result[-1].end + gap + 1e-9 and not crosses:
            result[-1] = Interval(result[-1].start, max(result[-1].end, interval.end))
        else:
            result.append(interval)
    return result


def plan_sequences(sequences: tuple[MotionSequence, ...], scores: tuple[Score, ...],
                   duration: float, config: Config, options: SequenceFilterOptions) -> SequencePlan:
    """Classify raw sequences and create padded edits, with explicit removal barriers."""
    expanded = [part for sequence in sequences for part in _split_staging(sequence, config, options)]
    raw_config = replace(config, pre_roll=0, post_roll=0, merge_gap=0)
    raw = select_intervals(scores, duration, raw_config)
    covered = [Interval(s.start, s.end) for s in expanded]
    # A one-sample residual at a track boundary must not acquire several seconds
    # of padding and turn into an empty clip after a removed return.
    uncovered = [item for item in subtract_intervals(raw, covered)
                 if item.duration + 1e-9 >= config.min_motion]
    expanded.extend(MotionSequence(0, item.start, item.end, None, None, None, None, "unknown",
                                    item.duration, 0, (), ("untracked_activity",)) for item in uncovered)
    sequences = tuple(replace(s, sequence_id=i + 1) for i, s in enumerate(sorted(expanded, key=lambda s: s.start)))
    model = learn_duration(sequences, options)
    opposite = "right" if options.approach_from == "left" else "left"
    decisions = []
    barriers = []
    for sequence in sequences:
        maximum, _, return_start, multiple, visited = _metrics(sequence, options)
        classification, action, reason, estimate = "uncertain", "keep", "uncertain_sequence", None
        flags = set(sequence.quality_flags)
        if ("unobserved_departure" in flags and return_start is not None
                and _progress(sequence.trajectory[-1], options) <= 0.25):
            # Returning to visible staging and stopping is not a frame exit.
            # It can still provide a well-observed approach/return trajectory.
            flags.remove("unobserved_departure")
        if flags or multiple:
            reason = "multiple_approaches" if multiple else "uncertain_tracking_or_boundary"
        elif sequence.entry_side == opposite and sequence.exit_side == options.approach_from:
            classification, action, reason = "return", "omit", "opposite_direction_return"
        elif sequence.entry_side == options.approach_from and sequence.exit_side == opposite:
            classification, reason = "forward_pass", "expected_traversal"
        elif sequence.entry_side == options.approach_from:
            same_side = sequence.exit_side == options.approach_from
            classification = "same_side" if same_side else "approach_and_return" if return_start is not None else "uncertain"
            has_return = same_side or return_start is not None
            if options.same_side == "keep" and has_return:
                reason = "same_side_keep_policy"
            elif same_side and options.same_side == "omit":
                action, reason = "omit", "same_side_omit_policy"
            elif has_return and visited is False:
                classification, action, reason = "aborted_approach", "omit", "turnback_before_trick_roi"
            elif has_return and model.reliable:
                anchor = sequence.entry_time if sequence.entry_time is not None else sequence.start
                target = model.target_seconds
                expected_end = anchor + target
                if sequence.end - anchor > target + max(1.0, 0.2 * target):
                    # A later observed reversal extends the estimate; an early
                    # fall/reversal must not prematurely truncate an attempt.
                    estimate = max(expected_end, return_start or expected_end)
                    padded_end = min(sequence.end, estimate + config.post_roll)
                    if sequence.end - padded_end >= 0.5:
                        action, reason = "trim", "return_exceeds_reference_duration"
                    else:
                        estimate, reason = None, "insufficient_tail_to_trim"
                else:
                    reason = "within_expected_duration"
            elif has_return:
                reason = model.reason
        else:
            reason = "unknown_approach_start"
        if action == "omit":
            retained = ()
            barriers.append(Interval(sequence.start, sequence.end))
        else:
            end = min(duration, (estimate if action == "trim" else sequence.end) + config.post_roll)
            if classification == "uncertain":
                retained = tuple(Interval(max(0, max(sequence.start, item.start) - config.pre_roll),
                                          min(duration, min(sequence.end, item.end) + config.post_roll))
                                 for item in raw if item.start < sequence.end and item.end > sequence.start)
            else:
                retained = (Interval(max(0, sequence.start - config.pre_roll), end),)
            if action == "trim":
                barriers.append(Interval(end, sequence.end))
        decisions.append(SequenceDecision(sequence.sequence_id, classification, action, reason,
                                           estimate, retained, maximum, return_start, visited))

    # Untracked activity has explicit uncertain records too; it participates in
    # exactly the same padding/barrier rules as every other retained sequence.
    updated = []
    for sequence, decision in zip(sequences, decisions):
        retained = []
        core_end = decision.estimated_end if decision.action == "trim" else sequence.end
        for interval in decision.retained_intervals:
            start, end = interval.start, interval.end
            for barrier in barriers:
                if barrier.end <= sequence.start + 1e-9 and barrier.end > start:
                    start = barrier.end
                elif barrier.start >= core_end - 1e-9 and barrier.start < end:
                    end = barrier.start
            if end > start + 1e-9:
                retained.extend(subtract_intervals([Interval(start, end)], barriers))
        updated.append(replace(decision, retained_intervals=tuple(retained)))
    updated = tuple(updated)
    candidates = [item for decision in updated for item in decision.retained_intervals]
    intervals = tuple(_merge(candidates, barriers, config.merge_gap))
    return SequencePlan(sequences, updated, model, intervals)


def sequence_report(sequence: MotionSequence, decision: SequenceDecision) -> dict:
    summary = {key: value for key, value in asdict(sequence).items() if key != "trajectory"}
    summary.update({"observed_start": sequence.start, "observed_end": sequence.end,
                    "duration": sequence.duration, **asdict(decision)})
    summary["retained_start"] = decision.retained_intervals[0].start if decision.retained_intervals else None
    summary["retained_end"] = decision.retained_intervals[-1].end if decision.retained_intervals else None
    summary["trajectory_fields"] = ["time", "x", "y", "width", "height", "area", "motion"]
    summary["trajectory"] = [[p.time, p.region.x, p.region.y, p.region.width, p.region.height,
                               p.region.area, p.motion] for p in sequence.trajectory]
    return summary
