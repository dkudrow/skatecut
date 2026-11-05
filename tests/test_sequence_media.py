from dataclasses import replace
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from skatecut import (Config, RenderOptions, SequenceFilterOptions, analyze_video,
                      render_video, replan_video)
from skatecut import pipeline
from skatecut.cli import main
from skatecut.media import probe
from skatecut.model import select_intervals


@pytest.fixture(scope="module")
def sequence_video(tmp_path_factory, ffmpeg):
    path = tmp_path_factory.mktemp("sequence-video") / "passes.mp4"
    width, height, fps, seconds = 320, 180, 10, 46
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-f", "lavfi", "-i",
               f"sine=frequency=880:sample_rate=48000:duration={seconds}", "-c:v", "libx264",
               "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", str(seconds), str(path)]
    y, x = np.indices((height, width))
    background = (40 + (x // 20 + y // 20) % 2 * 10).astype(np.uint8)
    with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        for i in range(seconds * fps):
            time = i / fps
            frame = background.copy()
            left = None
            for start in (1, 7, 13, 19, 25):
                if start <= time < start + 2:
                    left = -40 + (time - start) / 2 * 400
                if start + 3 <= time < start + 5:
                    left = 360 - (time - start - 3) / 2 * 400
            if 32 <= time < 34:
                left = -40 + (time - 32) / 2 * 260
            elif 34 <= time < 44:
                left = 220 - (time - 34) / 10 * 260
            if left is not None:
                x0, x1 = max(0, round(left)), min(width, round(left) + 40)
                if x1 > x0:
                    frame[55:140, x0:x1] = 220
            # Small wind-like activity never forms a subject-sized component.
            frame[5:9, 5 + i % 12:9 + i % 12] = 150
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        process.stdin = None
        errors = process.stderr.read().decode()
        process.stderr.close()
        process.stderr = None
        assert process.wait() == 0, errors
    return path


@pytest.fixture(scope="module")
def analysis(sequence_video):
    return analyze_video(sequence_video, config=Config(analysis_width=320, pre_roll=.2, post_roll=.5, merge_gap=.2),
                         sequence_filter=SequenceFilterOptions("left"))


def test_real_decoding_learns_duration_filters_returns_and_trims_turnback(analysis):
    plan = analysis.sequence_plan
    assert plan.duration_model.reliable and len(plan.duration_model.reference_ids) >= 5
    assert sum(d.classification == "return" and d.action == "omit" for d in plan.decisions) == 5
    assert sum(d.action == "trim" for d in plan.decisions) == 1
    assert analysis.kept_seconds < 22
    # Must-keep trick windows, including the beginning of the failed attempt.
    for start, end in [(s + .8, s + 1.4) for s in (1, 7, 13, 19, 25)] + [(33, 34.5)]:
        assert any(i.start <= start and i.end >= end for i in analysis.intervals)
    for time in (4.8, 10.8, 16.8, 22.8, 28.8, 42):
        assert not any(i.start <= time < i.end for i in analysis.intervals)
    payload = json.loads(json.dumps(analysis.to_dict()))
    assert payload["schema_version"] == 2
    assert payload["duration_model"]["reliable"]
    assert payload["sequences"][0]["trajectory"]
    assert not {"source", "output"} & payload.keys()


def test_filtering_works_with_motion_roi_without_inventing_frame_entries(sequence_video):
    analysis = analyze_video(sequence_video,
                             config=Config(analysis_width=320, pre_roll=.2, post_roll=.5,
                                           roi=(.4, .2, .2, .6)),
                             sequence_filter=SequenceFilterOptions("left"))
    first = analysis.sequence_plan.sequences[0]
    assert first.entry_side == "left" and first.exit_side == "right"
    assert first.entry_time < 1.5
    assert analysis.sequence_plan.duration_model.reliable


def test_replanning_does_not_decode_and_can_restore_legacy_edit(analysis, monkeypatch):
    def unexpected_decode(*args, **kwargs):
        pytest.fail("Replanning must not analyze the video again")

    monkeypatch.setattr(pipeline, "analyze", unexpected_decode)
    baseline = replan_video(analysis, sequence_filter=None)
    assert baseline.intervals == tuple(select_intervals(analysis.scores, analysis.video.duration, analysis.config))
    retained = replan_video(analysis, sequence_filter=SequenceFilterOptions("left", same_side="keep"))
    assert retained.kept_seconds > analysis.kept_seconds
    repadded = replan_video(analysis, sequence_filter=analysis.sequence_filter,
                            config=replace(analysis.config, pre_roll=2, post_roll=3, merge_gap=40))
    for sequence, decision in zip(repadded.sequence_plan.sequences, repadded.sequence_plan.decisions):
        if decision.action == "omit":
            assert not any(i.start < sequence.end and i.end > sequence.start for i in repadded.intervals)
    assert analysis.sequence_filter.same_side == "trim"
    with pytest.raises(ValueError, match="requires new video analysis"):
        replan_video(analysis, sequence_filter=analysis.sequence_filter,
                      config=replace(analysis.config, roi=(0, 0, .5, 1)))


def test_enabling_tracking_requires_a_new_analysis(sequence_video):
    baseline = analyze_video(sequence_video, config=Config(analysis_width=320))
    assert baseline.to_dict()["schema_version"] == 1
    with pytest.raises(ValueError, match="spatial tracking"):
        replan_video(baseline, sequence_filter=SequenceFilterOptions("left"))


def test_sequence_edit_exports_playable_audio_video(analysis, tmp_path):
    output = tmp_path / "filtered.mp4"
    render_video(analysis, output, options=RenderOptions(preset="ultrafast"))
    info = probe(output)
    assert info.has_audio
    assert info.duration == pytest.approx(analysis.kept_seconds, abs=.15)
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-xerror", "-i", str(output), "-f", "null", "-"], check=True)


def test_cli_sequence_report_and_parameter_validation(sequence_video, tmp_path):
    output = tmp_path / "cli.mp4"
    assert main([str(sequence_video), "-o", str(output), "--analyze-only", "--approach-from", "left",
                 "--analysis-width", "320"]) == 0
    report = json.loads(output.with_suffix(".segments.json").read_text())
    assert report["schema_version"] == 2 and not report["rendered"]
    assert report["sequences"] and report["duration_model"]["reliable"]
    assert output.with_suffix(".sequences.csv").read_text().startswith("sequence_id,observed_start")
    assert not output.exists()
    for extra in (["--attempt-duration", "5"], ["--same-side", "omit"], ["--trick-roi", "0", "0", "1", "1"]):
        assert main([str(sequence_video), "-o", str(tmp_path / "invalid.mp4"), *extra]) == 1
    assert main([str(sequence_video), "-o", str(tmp_path / "nan.mp4"),
                 "--approach-from", "left", "--attempt-duration", "nan"]) == 1
