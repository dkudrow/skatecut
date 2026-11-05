from dataclasses import replace
import json

import numpy as np
import pytest

from skatecut import Config, SequenceFilterOptions
from skatecut.model import Interval, Score, select_intervals
from skatecut.sequences import (MotionSequence, Observation, Region, TrackPoint,
                               build_sequences, learn_duration, plan_sequences, sequence_report)


def region(center, height=0.4):
    left, right = max(0.0, center - 0.04), min(1.0, center + 0.04)
    return Region(left, 0.2, right - left, height, (right - left) * height)


def sequence(identifier, start, seconds=3, entry="left", exit="right", *, positions=None,
             flags=(), origin="edge", pauses=0):
    count = round(seconds * 10)
    if positions is None:
        positions = np.linspace(0.03, 0.97, count)
        if entry == "right":
            positions = 1 - positions
        if exit == entry:
            split = max(3, count // 3)
            positions = np.r_[np.linspace(0.03, 0.75, split), np.linspace(0.75, 0.03, count - split)]
            if entry == "right":
                positions = 1 - positions
    points = tuple(TrackPoint(start + i / 10, region(float(x)), 0.04) for i, x in enumerate(positions))
    return MotionSequence(identifier, start, start + len(points) / 10, entry, exit,
                          start if entry else None, start + len(points) / 10 if exit else None,
                          origin, len(points) / 10 - pauses, pauses, points, flags)


def scores_for(sequences, duration=150):
    return tuple(Score(i / 10, 0.04 if any(s.start <= i / 10 < s.end for s in sequences) else 0)
                 for i in range(round(duration * 10)))


def references(durations=(3, 3, 3, 3, 3), side="left"):
    return tuple(sequence(i + 1, 2 + i * 12, d, side, "right" if side == "left" else "left")
                 for i, d in enumerate(durations))


def test_duration_model_uses_clean_passes_and_excludes_outliers():
    values = references((3, 3.1, 3.2, 3.1, 3.2, 12))
    model = learn_duration(values, SequenceFilterOptions("left"))
    assert model.reliable and model.source == "learned"
    assert len(model.reference_ids) == 5 and model.excluded_outlier_ids == (6,)
    assert model.median_seconds == pytest.approx(3.1)
    assert model.target_seconds >= model.median_seconds + 0.5


def test_insufficient_or_variable_references_do_not_guess():
    assert not learn_duration(references((3, 3)), SequenceFilterOptions("left")).reliable
    model = learn_duration(references((2, 4, 6, 8, 10)), SequenceFilterOptions("left"))
    assert not model.reliable and model.reason == "reference_durations_too_variable"


@pytest.mark.parametrize("side", ["left", "right"])
def test_forward_passes_kept_returns_omitted_and_turnbacks_shortened(side):
    opposite = "right" if side == "left" else "left"
    seqs = (*references(side=side), sequence(6, 70, 3, opposite, side),
            sequence(7, 80, 12, side, side))
    config = Config(pre_roll=0.2, post_roll=0.5, merge_gap=0)
    plan = plan_sequences(seqs, scores_for(seqs), 150, config, SequenceFilterOptions(side))
    assert [d.action for d in plan.decisions] == ["keep"] * 5 + ["omit", "trim"]
    trimmed = plan.decisions[-1]
    assert trimmed.estimated_end >= 80 + plan.duration_model.target_seconds
    assert trimmed.retained_intervals[-1].end < 92
    assert not any(i.start < 73 and i.end > 70 for i in plan.intervals)


@pytest.mark.parametrize("policy,action", [("keep", "keep"), ("omit", "omit"), ("trim", "keep")])
def test_same_side_policy_and_missing_duration(policy, action):
    seqs = (sequence(1, 2, 12, "left", "left"),)
    plan = plan_sequences(seqs, scores_for(seqs), 150, Config(),
                          SequenceFilterOptions("left", same_side=policy))
    assert plan.decisions[0].action == action


def test_manual_duration_is_used_without_training_and_keeps_fall_buffer():
    seqs = (sequence(1, 2, 12, "left", "left"),)
    plan = plan_sequences(seqs, scores_for(seqs), 150, Config(post_roll=1),
                          SequenceFilterOptions("left", attempt_duration=5))
    decision = plan.decisions[0]
    assert plan.duration_model.source == "manual"
    assert decision.action == "trim"
    assert decision.estimated_end >= 7
    assert decision.retained_intervals[-1].end >= 8


def test_duration_does_not_cut_a_forward_sequence_without_return_evidence():
    seqs = (*references(), sequence(6, 80, 12, "left", None,
                                   positions=np.linspace(0.03, 0.7, 120)))
    plan = plan_sequences(seqs, scores_for(seqs), 150, Config(), SequenceFilterOptions("left"))
    assert plan.decisions[-1].action == "keep"


def test_late_reversal_extends_estimate():
    positions = np.r_[np.linspace(0.03, 0.8, 80), np.linspace(0.8, 0.03, 60)]
    seqs = (*references(), sequence(6, 80, entry="left", exit="left", positions=positions))
    plan = plan_sequences(seqs, scores_for(seqs), 150, Config(post_roll=0.5), SequenceFilterOptions("left"))
    assert plan.decisions[-1].action == "trim"
    assert plan.decisions[-1].estimated_end >= 87.8


def test_return_to_visible_staging_without_frame_exit_can_be_trimmed():
    positions = np.r_[np.linspace(.03, .75, 30), np.linspace(.75, .12, 90)]
    seqs = (*references(), sequence(6, 80, entry="left", exit=None, positions=positions,
                                   flags=("unobserved_departure",)))
    plan = plan_sequences(seqs, scores_for(seqs), 150, Config(post_roll=.5), SequenceFilterOptions("left"))
    assert plan.decisions[-1].action == "trim"
    assert plan.decisions[-1].classification == "approach_and_return"


@pytest.mark.parametrize("flag", ["recording_start", "recording_end", "tracking_gap", "multiple_subjects"])
def test_uncertain_sequences_are_kept_even_under_omit_policy(flag):
    seqs = (sequence(1, 2, 12, "left", "left", flags=(flag,)),)
    plan = plan_sequences(seqs, scores_for(seqs), 150, Config(),
                          SequenceFilterOptions("left", same_side="omit"))
    assert plan.decisions[0].classification == "uncertain"
    assert plan.decisions[0].action == "keep"


def test_trick_roi_allows_aborted_approach_omission_but_keep_policy_wins():
    seqs = (sequence(1, 2, 4, "left", "left", positions=np.r_[np.linspace(.03, .15, 20), np.linspace(.15, .03, 20)]),)
    options = SequenceFilterOptions("left", trick_roi=(0.7, 0.2, 0.2, 0.5))
    plan = plan_sequences(seqs, scores_for(seqs), 150, Config(), options)
    assert plan.decisions[0].classification == "aborted_approach"
    assert plan.decisions[0].action == "omit"
    plan = plan_sequences(seqs, scores_for(seqs), 150, Config(), replace(options, same_side="keep"))
    assert plan.decisions[0].action == "keep"


def test_padding_and_large_merge_gap_cannot_restore_a_return():
    seqs = (sequence(1, 10), sequence(2, 14, 3, "right", "left"), sequence(3, 18))
    config = Config(pre_roll=5, post_roll=5, merge_gap=50)
    plan = plan_sequences(seqs, scores_for(seqs, 30), 30, config, SequenceFilterOptions("left"))
    assert len(plan.intervals) == 2
    assert plan.intervals[0].end == 14 and plan.intervals[1].start == 17


def test_padding_stops_at_return_instead_of_resuming_as_orphan_clip():
    seqs = (sequence(1, 10, 3), sequence(2, 13.2, .8, "right", "left"))
    plan = plan_sequences(seqs, scores_for(seqs, 20), 20, Config(pre_roll=0, post_roll=2),
                          SequenceFilterOptions("left"))
    assert plan.intervals == (Interval(10, 13.2),)
    seqs = (sequence(1, 2, 1, "right", "left"), sequence(2, 4, 3))
    plan = plan_sequences(seqs, scores_for(seqs, 15), 15, Config(pre_roll=3, post_roll=2),
                          SequenceFilterOptions("left"))
    assert plan.intervals == (Interval(3, 9),)


def test_trimmed_tail_is_a_merge_barrier():
    seqs = (sequence(1, 2, 12, "left", "left"), sequence(2, 16))
    plan = plan_sequences(seqs, scores_for(seqs, 30), 30,
                          Config(pre_roll=3, post_roll=1, merge_gap=50),
                          SequenceFilterOptions("left", attempt_duration=5))
    assert plan.decisions[0].action == "trim"
    assert len(plan.intervals) == 2
    assert plan.intervals[0].end <= 8.1 and plan.intervals[1].start >= 14


def test_untracked_activity_falls_back_to_legacy_motion_intervals():
    scores = tuple(Score(i / 10, .1 if 20 <= i < 40 else 0) for i in range(100))
    config = Config()
    plan = plan_sequences((), scores, 10, config, SequenceFilterOptions("left"))
    assert plan.intervals == tuple(select_intervals(scores, 10, config))
    assert plan.sequences[0].quality_flags == ("untracked_activity",)
    assert plan.decisions[0].action == "keep"


def test_tiny_track_boundary_residual_cannot_become_a_padded_empty_clip():
    seqs = (sequence(1, 2, 3, "right", "left"),)
    scores = tuple(Score(i / 10, .04 if 20 <= i <= 50 else 0) for i in range(100))
    plan = plan_sequences(seqs, scores, 10, Config(), SequenceFilterOptions("left"))
    assert plan.decisions[0].action == "omit"
    assert plan.intervals == ()


def test_multiple_attempts_without_a_clear_split_are_uncertain():
    positions = np.r_[np.linspace(.03, .8, 20), np.linspace(.8, .1, 20), np.linspace(.1, .97, 20)]
    seqs = (sequence(1, 2, entry="left", exit="right", positions=positions),)
    plan = plan_sequences(seqs, scores_for(seqs), 150, Config(), SequenceFilterOptions("left"))
    assert plan.decisions[0].reason == "multiple_approaches"
    assert not plan.duration_model.reference_ids


def observations(paths, seconds=10):
    items = []
    for i in range(round(seconds * 10)):
        time = i / 10
        boxes = tuple(region(float(values[i - start])) for start, values in paths if start <= i < start + len(values))
        items.append(Observation(time, .04 if boxes else 0, boxes))
    return tuple(items)


def test_tracker_detects_edges_and_separates_a_quick_return():
    obs = observations([(10, np.linspace(.02, .98, 20)), (32, np.linspace(.98, .02, 20))], 7)
    seqs = build_sequences(obs, 7, Config(), SequenceFilterOptions("left"))
    assert len(seqs) == 2
    assert (seqs[0].entry_side, seqs[0].exit_side) == ("left", "right")
    assert (seqs[1].entry_side, seqs[1].exit_side) == ("right", "left")
    assert seqs[0].entry_time == 1 and seqs[0].exit_time == 3
    assert not seqs[0].quality_flags


def test_middle_disappearance_is_not_an_exit():
    obs = observations([(10, np.linspace(.02, .4, 10)), (32, np.linspace(.4, .6, 10))], 6)
    seqs = build_sequences(obs, 6, Config(), SequenceFilterOptions("left"))
    assert seqs[0].exit_side is None
    assert "unobserved_departure" in seqs[0].quality_flags


def test_recording_boundaries_are_censored():
    obs = observations([(0, np.linspace(.3, .7, 30))], 3)
    seqs = build_sequences(obs, 3, Config(), SequenceFilterOptions("left"))
    assert seqs[0].entry_side is seqs[0].exit_side is None
    assert {"recording_start", "recording_end"} <= set(seqs[0].quality_flags)


def test_low_motion_and_detached_board_do_not_seed_false_sequences():
    small = tuple(Observation(i / 10, .004, (Region(.1, .2, .02, .2, .004),)) for i in range(30))
    assert build_sequences(small, 3, Config(), SequenceFilterOptions("left")) == ()
    board = tuple(Observation(i / 10, .02, (Region(.1 + i / 100, .8, .2, .03, .006),)) for i in range(30))
    assert build_sequences(board, 3, Config(), SequenceFilterOptions("left")) == ()


def test_visible_staging_pause_splits_next_approach():
    positions = np.r_[np.linspace(.03, .8, 25), np.linspace(.8, .1, 25),
                      np.repeat(.1, 10), np.linspace(.1, .97, 20)]
    seqs = (sequence(1, 2, entry="left", exit="right", positions=positions),)
    plan = plan_sequences(seqs, scores_for(seqs), 150, Config(), SequenceFilterOptions("left"))
    assert len(plan.sequences) == 2
    assert plan.sequences[1].entry_origin == "staging"
    assert plan.decisions[1].classification == "forward_pass"


def test_reports_include_observed_and_retained_boundaries():
    seqs = (sequence(1, 2, 12, "left", "left"),)
    plan = plan_sequences(seqs, scores_for(seqs), 150, Config(),
                          SequenceFilterOptions("left", attempt_duration=5))
    report = sequence_report(plan.sequences[0], plan.decisions[0])
    assert report["observed_end"] == 14 and report["retained_end"] < 14
    assert json.loads(json.dumps(report))["action"] == "trim"


@pytest.mark.parametrize("values", [dict(approach_from="top"), dict(approach_from="left", attempt_duration=-1),
    dict(approach_from="left", attempt_duration=float("nan")), dict(approach_from="left", same_side="guess"),
    dict(approach_from="left", trick_roi=(.9, 0, .2, 1)), dict(approach_from="left", min_references=1)])
def test_filter_options_validate(values):
    with pytest.raises(ValueError):
        SequenceFilterOptions(**values)
