from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json

import pytest

from skatecut import (Config, ProgressEvent, RenderOptions, analyze_video,
                      process_video, render_video)
from skatecut import media, pipeline
from skatecut.media import MediaError, probe
from conftest import make_video


@pytest.fixture(scope="module")
def analysis(active_video):
    return analyze_video(active_video, config=Config(analysis_width=320))


def test_python_api_returns_results_and_structured_progress(active_video, tmp_path, capsys):
    events = []
    output = tmp_path / "api-result.mp4"
    scratch = tmp_path / "job-work"
    scratch.mkdir()
    result = process_video(active_video, output, config=Config(analysis_width=320),
                           options=RenderOptions(preset="ultrafast"),
                           progress=events.append, scratch_dir=scratch)
    assert result.rendered and result.output == output
    assert len(result.analysis.intervals) == 4
    assert probe(output).duration == pytest.approx(result.analysis.kept_seconds, abs=0.15)
    assert capsys.readouterr().out == ""
    assert events and all(isinstance(event, ProgressEvent) for event in events)
    assert {"probe", "background", "analysis", "selection", "render", "joining", "complete"} <= {
        event.stage for event in events
    }
    assert events[-1].stage == "complete"
    counts = [event for event in events if event.stage == "render" and event.unit == "clips"]
    assert counts[-1].completed == counts[-1].total == 4
    summary = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert summary["schema_version"] == 1 and summary["rendered"] is True
    assert summary["render_options"]["preset"] == "ultrafast"
    assert not {"source", "output", "scores"} & summary.keys()
    assert str(active_video) not in json.dumps(summary)
    assert list(scratch.iterdir()) == []
    # Library callers opt into their own reporting/storage.
    assert not output.with_suffix(".segments.json").exists()
    assert not output.with_suffix(".scores.csv").exists()


def test_analysis_can_be_exported_concurrently_without_rerunning(analysis, tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    def unexpected_analysis(*args, **kwargs):
        pytest.fail("Export must reuse the existing analysis")

    monkeypatch.setattr(pipeline, "analyze", unexpected_analysis)

    def export(index):
        return render_video(analysis, tmp_path / f"result-{index}.mp4",
                            options=RenderOptions(preset="ultrafast", crf=18 + index),
                            scratch_dir=scratch)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(export, (0, 1)))
    assert first.output != second.output
    for result in (first, second):
        assert result.analysis is analysis
        assert probe(result.output).duration == pytest.approx(analysis.kept_seconds, abs=0.15)
    assert list(scratch.iterdir()) == []


def test_library_analysis_only_and_empty_edit(tmp_path, ffmpeg, capsys):
    source = make_video(tmp_path / "still.mp4", seconds=2)
    analysis_only = process_video(source)
    assert not analysis_only.rendered and analysis_only.output is None
    output = tmp_path / "empty.mp4"
    result = render_video(analysis_only.analysis, output)
    assert not result.rendered and not output.exists()
    assert result.to_dict()["intervals"] == []
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("alias", ["source", "symlink", "hardlink"])
def test_api_protects_input_before_analysis(active_video, tmp_path, monkeypatch, alias):
    destination = active_video
    if alias == "symlink":
        destination = tmp_path / "symlink.mp4"
        destination.symlink_to(active_video)
    elif alias == "hardlink":
        destination = tmp_path / "hardlink.mp4"
        destination.hardlink_to(active_video)

    def unexpected_analysis(*args, **kwargs):
        pytest.fail("Source aliases must be rejected before analysis")

    monkeypatch.setattr(pipeline, "analyze_video", unexpected_analysis)
    with pytest.raises(ValueError, match="source video"):
        process_video(active_video, destination, overwrite=True)


def test_api_rejects_existing_output_before_analysis(active_video, tmp_path, monkeypatch):
    output = tmp_path / "previous.mp4"
    output.write_bytes(b"previous result")

    def unexpected_analysis(*args, **kwargs):
        pytest.fail("Existing outputs must be rejected before analysis")

    monkeypatch.setattr(pipeline, "analyze_video", unexpected_analysis)
    with pytest.raises(FileExistsError):
        process_video(active_video, output)
    assert output.read_bytes() == b"previous result"


def test_failed_export_cleans_scratch_and_preserves_previous_output(analysis, tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    output = tmp_path / "previous.mp4"
    output.write_bytes(b"previous result")

    def fail_encoder(*args, **kwargs):
        assert list(scratch.iterdir()), "Intermediate directory must use the supplied scratch path"
        raise MediaError("simulated encoder failure")

    monkeypatch.setattr(media, "_run", fail_encoder)
    with pytest.raises(MediaError, match="simulated encoder failure"):
        render_video(analysis, output, overwrite=True, scratch_dir=scratch)
    assert output.read_bytes() == b"previous result"
    assert list(scratch.iterdir()) == []


def test_json_configuration_round_trip_and_immutability():
    values = {"pre_roll": 2, "roi": [0, 0, 0.5, 0.5]}
    config = Config(**values)
    values["roi"][0] = 0.25
    assert config.roi == (0, 0, 0.5, 0.5)
    assert Config(**json.loads(json.dumps(asdict(config)))) == config


@pytest.mark.parametrize("values", [
    {"analysis_width": 640.5}, {"sample_fps": "10"}, {"min_motion": True},
    {"roi": [0, 0, None, 1]},
])
def test_non_cli_configuration_errors_are_value_errors(values):
    with pytest.raises(ValueError):
        Config(**values)


@pytest.mark.parametrize("values", [{"crf": -1}, {"crf": 52}, {"crf": 1.5},
                                    {"crf": True}, {"preset": "unknown"}])
def test_non_cli_export_options_are_validated(values):
    with pytest.raises(ValueError):
        RenderOptions(**values)
