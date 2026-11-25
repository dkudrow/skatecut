import json
import subprocess

import numpy as np
import pytest

from skatecut.cli import main
from skatecut.detector import analyze, foreground_fraction
from skatecut.media import MediaError, frames, probe, render, require_hdr_filters, require_counter_filter
from skatecut.model import Config, Interval, select_intervals
from conftest import make_video


def test_small_disconnected_motion_is_ignored():
    mask = np.zeros((180, 320), dtype=np.uint8)
    for x in range(0, 300, 10):
        mask[5:9, x:x + 4] = 255
    assert foreground_fraction(mask, 0.001) == 0


def test_static_and_tiny_motion_produce_no_clips(tmp_path, ffmpeg):
    path = make_video(tmp_path / "quiet.mp4", seconds=5, tiny_motion=True)
    info = probe(path)
    config = Config(analysis_width=320)
    results = analyze(path, info, config, lambda _: None)
    assert select_intervals(results, info.duration, config) == []


def test_attempts_at_start_middle_and_end_are_found(active_video):
    info = probe(active_video)
    config = Config(analysis_width=320)
    results = analyze(active_video, info, config, lambda _: None)
    intervals = select_intervals(results, info.duration, config)
    assert len(intervals) == 4
    assert intervals[0].start == 0
    assert intervals[-1].end == 24
    for event_time in (0, 8, 16, 23):
        assert any(item.start <= event_time < item.end for item in intervals)
    assert sum(item.duration for item in intervals) < 18


def test_roi_excludes_activity(active_video):
    info = probe(active_video)
    config = Config(analysis_width=320, roi=(0, 0, 1, 0.25))
    results = analyze(active_video, info, config, lambda _: None)
    assert select_intervals(results, info.duration, config) == []


def test_render_and_cli_reports(active_video, tmp_path):
    output = tmp_path / "highlight.mp4"
    assert main([str(active_video), "-o", str(output), "--analysis-width", "320", "--preset", "ultrafast"]) == 0
    payload = json.loads(output.with_suffix(".segments.json").read_text())
    assert payload["rendered"] is True
    assert payload["schema_version"] == 1
    assert payload["render_options"]["preset"] == "ultrafast"
    assert len(payload["intervals"]) == 4
    info = probe(output)
    assert info.has_audio
    assert (info.width, info.height) == (320, 180)
    assert info.duration == pytest.approx(payload["kept_seconds"], abs=0.15)
    assert output.with_suffix(".scores.csv").read_text().splitlines()[0].startswith("time_seconds,motion_fraction")
    # Decode the whole output to catch muxing/concatenation errors.
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"], check=True)
    previous = output.read_bytes()
    assert main([str(active_video), "-o", str(output)]) == 1
    assert output.read_bytes() == previous


def test_analyze_only_and_empty_edit(tmp_path, ffmpeg):
    source = make_video(tmp_path / "still.mp4", seconds=2)
    output = tmp_path / "empty.mp4"
    assert main([str(source), "-o", str(output)]) == 0
    assert not output.exists()
    payload = json.loads(output.with_suffix(".segments.json").read_text())
    assert payload["intervals"] == []
    assert payload["rendered"] is False
    assert main([str(source), "-o", str(source), "--overwrite"]) == 1


def test_analyze_only_does_not_create_video(active_video, tmp_path):
    output = tmp_path / "analyzed.mp4"
    assert main([str(active_video), "-o", str(output), "--analyze-only"]) == 0
    assert output.with_suffix(".segments.json").exists()
    assert not output.exists()


def test_silent_video_can_be_rendered(tmp_path, ffmpeg):
    source = make_video(tmp_path / "silent.mp4", seconds=3, events=[(1, 2)])
    output = tmp_path / "edited.mp4"
    render(source, output, probe(source), [Interval(0.5, 2.5)], preset="ultrafast")
    info = probe(output)
    assert not info.has_audio
    assert info.duration == pytest.approx(2, abs=0.1)


def test_rotation_metadata_is_applied_once(active_video, tmp_path):
    rotated = tmp_path / "phone.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(active_video),
                    "-c", "copy", str(rotated)], check=True)
    info = probe(rotated)
    assert (info.width, info.height) == (180, 320)
    decoded = list(frames(rotated, 90, 160, 0.1))
    assert decoded and decoded[0].shape == (160, 90)
    output = tmp_path / "upright.mp4"
    render(rotated, output, info, [Interval(7, 10)], preset="ultrafast")
    result = probe(output)
    assert (result.width, result.height) == (180, 320)
    assert result.rotation % 360 == 0


def test_vfr_analysis_and_export(active_video, tmp_path):
    source = tmp_path / "vfr.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(active_video), "-vf",
                    "select='if(lt(t,12),1,not(mod(n,2)))'", "-fps_mode", "vfr",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "copy", str(source)], check=True)
    info = probe(source)
    config = Config(analysis_width=320)
    results = analyze(source, info, config, lambda _: None)
    assert len(results) == pytest.approx(info.duration * config.sample_fps, abs=2)
    intervals = select_intervals(results, info.duration, config)
    assert len(intervals) == 4
    output = tmp_path / "vfr-output.mp4"
    render(source, output, info, [Interval(7, 10), Interval(15, 18)], preset="ultrafast")
    assert probe(output).duration == pytest.approx(6, abs=0.15)


@pytest.mark.parametrize("timestamp_offset", [0, 5])
def test_audio_offsets_survive_multiple_cuts(tmp_path, ffmpeg, timestamp_offset):
    # The audio stream begins 0.25s after video. Its pulses still coincide with
    # visual events at 1–1.5s and 3.5–4s, even on a nonzero source timeline.
    silent = make_video(tmp_path / "visual.mp4", seconds=5, events=[(1, 1.5), (3.5, 4)])
    source = tmp_path / "offset.mov"
    expression = "if(between(t,0.75,1.25)+between(t,3.25,3.75),0.5*sin(2*PI*880*t),0)"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(silent), "-itsoffset", "0.25",
                    "-f", "lavfi", "-i", f"aevalsrc='{expression}':s=48000:d=5",
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                    "-output_ts_offset", str(timestamp_offset), str(source)], check=True)
    info = probe(source)
    assert info.video_start == pytest.approx(timestamp_offset, abs=0.03)
    output = tmp_path / "sync.mp4"
    render(source, output, info, [Interval(0.5, 2.5), Interval(3, 4.5)])
    data = subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-vn", "-ac", "1",
                           "-ar", "48000", "-f", "f32le", "pipe:1"], capture_output=True, check=True).stdout
    samples = np.frombuffer(data, dtype=np.float32)
    # Compare 10ms RMS bins; AAC edge ringing should stay within a few bins.
    usable = samples[:len(samples) // 480 * 480].reshape(-1, 480)
    rms = np.sqrt(np.mean(usable ** 2, axis=1))
    active = np.flatnonzero(rms > 0.1) / 100
    groups = np.split(active, np.flatnonzero(np.diff(active) > 0.05) + 1)
    assert len(groups) == 2
    assert groups[0][0] == pytest.approx(0.5, abs=0.06)
    assert groups[0][-1] == pytest.approx(1, abs=0.06)
    assert groups[1][0] == pytest.approx(2.5, abs=0.06)
    assert groups[1][-1] == pytest.approx(3, abs=0.06)
    visual = list(frames(output, 320, 180, 10))
    # The object is bright; the textured background stays below 60.
    visual_on = [i / 10 for i, frame in enumerate(visual) if np.count_nonzero(frame > 150) > 500]
    assert min(visual_on) == pytest.approx(0.5, abs=0.1)
    assert max(visual_on) == pytest.approx(2.9, abs=0.1)


@pytest.mark.parametrize("transfer", ["arib-std-b67", "smpte2084"])
@pytest.mark.parametrize("counter", [False, True])
def test_hdr_export_is_tone_mapped_and_tagged_sdr(active_video, tmp_path, transfer, counter):
    try:
        require_hdr_filters()
        if counter:
            require_counter_filter()
    except MediaError as exc:
        pytest.skip(str(exc))
    source = tmp_path / "hdr.mp4"
    conversion = ("setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709,"
                  "zscale=transfer=linear:npl=100,format=gbrpf32le,zscale=primaries=bt2020,"
                  f"zscale=transfer={transfer}:matrix=bt2020nc:range=limited,format=yuv420p10le")
    # FFV1 stores real 10-bit HDR pixels without requiring an HEVC encoder.
    source = source.with_suffix(".mkv")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(active_video), "-t", "2",
                    "-vf", conversion, "-c:v", "ffv1", "-c:a", "copy", str(source)],
                   check=True, capture_output=True)
    assert probe(source).hdr
    output = tmp_path / "sdr.mp4"
    assert main([str(source), "-o", str(output), "--preset", "ultrafast"]
                + (["--attempt-counter"] if counter else [])) == 0
    assert output.exists()
    assert not probe(output).hdr
    metadata = json.loads(subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(output)
    ], check=True, capture_output=True, text=True).stdout)["streams"][0]
    assert metadata["color_primaries"] == metadata["color_transfer"] == metadata["color_space"] == "bt709"
    assert metadata["pix_fmt"] == "yuv420p"
    report = json.loads(output.with_suffix(".segments.json").read_text())
    assert report["rendered"] is True
    assert report["color_conversion"] == "hdr_to_sdr_bt709"
    decoded = list(frames(output, 320, 180, 10))
    assert max(float(frame.std()) for frame in decoded) > 10
    if counter:
        assert np.count_nonzero(decoded[0][:32, :120] > 150) > 50


