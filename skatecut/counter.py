"""Map retained attempts to non-overlapping labels on the source timeline."""

from dataclasses import dataclass, replace
from collections.abc import Sequence
import math

from .model import Config, Interval, Score, select_intervals
from .sequences import SequencePlan


@dataclass(frozen=True)
class AttemptSpan:
    number: int
    start: float
    end: float

    def __post_init__(self):
        if isinstance(self.number, bool) or not isinstance(self.number, int) or self.number < 1:
            raise ValueError("Attempt number must be a positive integer")
        if (any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
                for t in (self.start, self.end)) or not 0 <= self.start < self.end):
            raise ValueError("Attempt span must have finite times with 0 <= start < end")


def plan_counter(intervals: Sequence[Interval], scores: Sequence[Score], duration: float,
                 config: Config, sequence_plan: SequencePlan | None = None) -> tuple[AttemptSpan, ...]:
    """Count before clip merging; one split sequence retains the same number.

    Labels include lead/trail footage. Where padding overlaps, the next label
    starts with its lead-in, but never before the previous attempt's core ends.
    Uncertain sequences count as one detected attempt; this is not trick recognition.
    """
    if not intervals:
        return ()
    candidates = []
    if sequence_plan is not None:
        for sequence, decision in zip(sequence_plan.sequences, sequence_plan.decisions):
            windows = decision.retained_intervals
            if decision.action != "omit" and any(
                    w.start < i.end and w.end > i.start for w in windows for i in intervals):
                core_end = decision.estimated_end if decision.action == "trim" else sequence.end
                candidates.append((sequence.start, core_end, windows[0].start))
    else:
        # Removing padding and merge-gap preserves distinct detected events even
        # when the eventual export joins them into a single continuous clip.
        raw_config = replace(config, pre_roll=0, post_roll=0, merge_gap=0)
        for event in select_intervals(scores, duration, raw_config):
            if any(event.start < i.end and event.end > i.start for i in intervals):
                candidates.append((event.start, event.end, max(0, event.start - config.pre_roll)))
    if not candidates:
        # Support callers constructing an AnalysisResult with explicit intervals.
        return tuple(AttemptSpan(n, i.start, i.end) for n, i in enumerate(intervals, 1))
    candidates.sort()
    starts = [intervals[0].start]
    for previous, current in zip(candidates, candidates[1:]):
        starts.append(max(starts[-1], min(current[0], max(previous[1], current[2]))))
    spans = []
    for interval in intervals:
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else duration
            left, right = max(interval.start, start), min(interval.end, end)
            if right > left:
                spans.append(AttemptSpan(index + 1, left, right))
    return tuple(spans)
