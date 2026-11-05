from dataclasses import replace

import pytest

from skatecut.model import Config, Interval, Score, select_intervals


def scores(values, fps=10):
    return [Score(i / fps, value) for i, value in enumerate(values)]


@pytest.fixture
def config():
    return Config(start_threshold=0.1, min_motion=0.3, quiet_time=0.3,
                  pre_roll=0, post_roll=0, merge_gap=0)


def test_quiet_and_short_noise_are_removed(config):
    assert select_intervals(scores([0] * 10 + [0.3] * 2 + [0] * 10), 2.2, config) == []


def test_hysteresis_keeps_motion_below_start_threshold(config):
    result = select_intervals(scores([0] * 5 + [0.2] * 3 + [0.07] * 5 + [0] * 5), 1.8, config)
    assert result == [Interval(0.5, 1.3)]


def test_brief_quiet_does_not_split_activity(config):
    result = select_intervals(scores([0.2] * 4 + [0] * 2 + [0.2] * 4), 1, config)
    assert result == [Interval(0, 1)]


def test_padding_clamping_and_merge(config):
    config = replace(config, pre_roll=0.2, post_roll=0.2, merge_gap=0.2)
    result = select_intervals(scores([0.2] * 3 + [0] * 5 + [0.2] * 3), 1.1, config)
    assert len(result) == 1
    assert result[0].start == 0
    assert result[0].end == 1.1


def test_confirmed_activity_at_both_boundaries(config):
    result = select_intervals(scores([0.2] * 3 + [0] * 8 + [0.2] * 3), 1.4, config)
    assert result[0].start == 0
    assert result[0].end == pytest.approx(0.3)
    assert result[1].start == 1.1
    assert result[1].end == 1.4


def test_partial_last_bin_does_not_confirm_short_event(config):
    assert select_intervals(scores([0.2] * 3), 0.25, config) == []


@pytest.mark.parametrize("values", [
    {"sample_fps": 0}, {"sensitivity": -1}, {"pre_roll": -1},
    {"start_threshold": float("nan")}, {"sample_fps": float("inf")},
    {"roi": (0.9, 0, 0.2, 1)}, {"roi": (0, 0, 0, 1)},
    {"min_component": 2}, {"background_seconds": 0},
])
def test_reject_invalid_configuration(values):
    with pytest.raises(ValueError):
        Config(**values)


def test_reject_unordered_scores(config):
    with pytest.raises(ValueError, match="timestamps"):
        select_intervals([Score(0.2, 0), Score(0.1, 0)], 1, config)

