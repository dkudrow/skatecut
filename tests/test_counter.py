from dataclasses import replace
import json
import subprocess

import numpy as np
import pytest

from skatecut import Config, RenderOptions, analyze_video, render_video
from skatecut import media
from skatecut.cli import main
from skatecut.counter import AttemptSpan, plan_counter
from skatecut.media import MediaError, frames, probe
from skatecut.model import Interval, Score, select_intervals
from skatecut.sequences import SequenceFilterOptions, plan_sequences
from test_sequences import sequence, scores_for


def test_motion_events_are_counted_before_padding_and_merging():
    config = Config(pre_roll=3, post_roll=2, merge_gap=10)
    scores = tuple(Score(i / 10, .1 if 1 <= i / 10 < 2 or 4 <= i / 10 < 5 else 0)
                   for i in range(80))
    intervals = select_intervals(scores, 8, config)
    assert intervals == [Interval(0, 7)]
    # The next event's overlapping lead-in cannot change the previous core's label.
    assert plan_counter(intervals, scores, 8, config) == (
        AttemptSpan(1, 0, 2), AttemptSpan(2, 2, 7))


def test_removed_returns_do_not_count_and_numbering_covers_all_retained_frames():
    sequences = (sequence(1, 1, 2), sequence(2, 4, 2, "right", "left"), sequence(3, 8, 2))
    scores = scores_for(sequences, 12)
    config = Config(pre_roll=.2, post_roll=.5, merge_gap=20)
    plan = plan_sequences(sequences, scores, 12, config, SequenceFilterOptions("left"))
    spans = plan_counter(plan.intervals, scores, 12, config, plan)
    assert {span.number for span in spans} == {1, 2}
    assert sum(span.end - span.start for span in spans) == pytest.approx(
        sum(i.duration for i in plan.intervals))
    assert not any(s.start < 6 and s.end > 4 for s in spans)
    assert next(s.number for s in spans if s.start <= 9 < s.end) == 2


def test_split_sequence_keeps_its_number_and_empty_decisions_do_not_count():
    sequences = (sequence(1, 1, 6), sequence(2, 9, 2))
    config = Config(pre_roll=0, post_roll=0, merge_gap=0)
    scores = scores_for(sequences, 12)
    plan = plan_sequences(sequences, scores, 12, config, SequenceFilterOptions("left"))
    intervals = (Interval(1, 3), Interval(5, 7))
    decisions = (replace(plan.decisions[0], retained_intervals=intervals),
                 replace(plan.decisions[1], retained_intervals=()))
    plan = replace(plan, decisions=decisions, intervals=intervals)
    assert plan_counter(intervals, scores, 12, config, plan) == (
        AttemptSpan(1, 1, 3), AttemptSpan(1, 5, 7))


def test_empty_edit_and_explicit_intervals_without_scores():
    assert plan_counter((), (), 5, Config()) == ()
    assert plan_counter((Interval(1, 2), Interval(3, 4)), (), 5, Config()) == (
        AttemptSpan(1, 1, 2), AttemptSpan(2, 3, 4))


@pytest.mark.parametrize("value", [1, "true", None])
def test_counter_option_requires_a_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        RenderOptions(attempt_counter=value)


@pytest.mark.parametrize("values", [(0, 0, 1), (True, 0, 1), ("1", 0, 1),
                                     (1, -1, 2), (1, 1, 1), (1, 0, float("nan"))])
def test_invalid_label_values_are_rejected(values):
    with pytest.raises(ValueError):
        AttemptSpan(*values)


def test_counter_requires_drawtext_only_when_enabled(active_video, tmp_path, monkeypatch):
    analysis = analyze_video(active_video, config=Config(analysis_width=320))
    media.require_counter_filter.cache_clear()
    real_run = media._run

    def without_drawtext(command):
        if "-filters" in command:
            return subprocess.CompletedProcess(command, 0, stdout=" ... scale V->V Scale video\n")
        return real_run(command)

    monkeypatch.setattr(media, "_run", without_drawtext)
    output = tmp_path / "previous.mp4"
    output.write_bytes(b"previous output")
    try:
        with pytest.raises(MediaError, match="drawtext"):
            render_video(analysis, output, overwrite=True,
                         options=RenderOptions(attempt_counter=True))
        assert output.read_bytes() == b"previous output"
        # Ordinary exports do not depend on drawtext.
        render_video(analysis, output, overwrite=True, options=RenderOptions(preset="ultrafast"))
        assert probe(output).has_audio
    finally:
        media.require_counter_filter.cache_clear()


def test_cli_overlay_changes_numbers_within_a_merged_clip(active_video, tmp_path):
    try:
        media.require_counter_filter()
    except MediaError as exc:
        pytest.skip(str(exc))
    output = tmp_path / "numbered.mp4"
    assert main([str(active_video), "-o", str(output), "--analysis-width", "320",
                 "--merge-gap", "30", "--attempt-counter", "--preset", "ultrafast"]) == 0
    report = json.loads(output.with_suffix(".segments.json").read_text())
    assert report["render_options"]["attempt_counter"] is True
    assert len(report["intervals"]) == 1
    assert [s["number"] for s in report["attempt_counter"]] == [1, 2, 3, 4]
    info = probe(output)
    assert info.has_audio and (info.width, info.height) == (320, 180)
    assert info.duration == pytest.approx(report["kept_seconds"], abs=.15)
    images = list(frames(output, 320, 180, 1))
    # This part of the synthetic scene has no subject; white pixels are the label.
    labels = [images[t][:32, :120] > 150 for t in (1, 9, 17, 23)]
    assert all(np.count_nonzero(label) > 50 for label in labels)
    assert all(np.count_nonzero(a != b) > 8 for a, b in zip(labels, labels[1:]))
    # The final output must fully decode, including its audio.
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-xerror", "-i", str(output),
                    "-f", "null", "-"], check=True)


def test_analyze_only_reports_planned_counter_without_rendering(active_video, tmp_path, monkeypatch):
    def unexpected_render(*args, **kwargs):
        pytest.fail("Analysis-only must not render or require drawtext")

    monkeypatch.setattr(media, "require_counter_filter", unexpected_render)
    output = tmp_path / "planned.mp4"
    assert main([str(active_video), "-o", str(output), "--analyze-only",
                 "--analysis-width", "320", "--attempt-counter"]) == 0
    report = json.loads(output.with_suffix(".segments.json").read_text())
    assert report["rendered"] is False and not output.exists()
    assert {s["number"] for s in report["attempt_counter"]} == {1, 2, 3, 4}


def test_overlay_uses_rotated_display_and_nonzero_source_timestamps(active_video, tmp_path):
    try:
        media.require_counter_filter()
    except MediaError as exc:
        pytest.skip(str(exc))
    source = tmp_path / "phone.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(active_video),
                    "-t", "3", "-c", "copy", "-output_ts_offset", "5", str(source)], check=True)
    info = probe(source)
    assert info.video_start == pytest.approx(5, abs=.03)
    assert (info.width, info.height) == (180, 320)
    output = tmp_path / "upright.mp4"
    media.render(source, output, info, [Interval(.5, 1.5), Interval(2, 3)], preset="ultrafast",
                 attempt_spans=(AttemptSpan(1, .5, 1.5), AttemptSpan(2, 2, 3)))
    result = probe(output)
    assert result.has_audio and (result.width, result.height) == (180, 320)
    assert result.rotation % 360 == 0 and result.duration == pytest.approx(2, abs=.15)
    images = list(frames(output, 180, 320, 10))
    first, second = (images[i][:32, :120] > 150 for i in (3, 13))
    assert all(np.count_nonzero(label) > 50 for label in (first, second))
    assert np.count_nonzero(first != second) > 8
