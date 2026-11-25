"""FFmpeg decoding and export. All external commands use argument arrays."""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from collections.abc import Sequence

import numpy as np

from .model import Interval
from .counter import AttemptSpan
from .progress import ProgressCallback, emit


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration: float
    video_start: float
    has_audio: bool
    rotation: int
    hdr: bool


def require_tools():
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            raise MediaError(f"{name} is missing. Install FFmpeg and put it on your PATH.")


@cache
def require_hdr_filters():
    available = _run(["ffmpeg", "-hide_banner", "-filters"]).stdout
    names = {line.split()[1] for line in available.splitlines() if len(line.split()) >= 3}
    if not {"zscale", "tonemap", "sidedata"} <= names:
        raise MediaError("HDR export needs FFmpeg with zscale, tonemap, and sidedata filters. "
                         "Install a full FFmpeg build or use --analyze-only.")


@cache
def require_counter_filter():
    available = _run(["ffmpeg", "-hide_banner", "-filters"]).stdout
    names = {line.split()[1] for line in available.splitlines() if len(line.split()) >= 3}
    if "drawtext" not in names:
        raise MediaError("Attempt counter needs FFmpeg with the drawtext filter. "
                         "Install a full FFmpeg build or omit --attempt-counter.")


def counter_filters(info: VideoInfo, interval: Interval, spans: Sequence[AttemptSpan]) -> list[str]:
    """Draw fixed, generated text after rotation/tone mapping using clip-local time."""
    size = max(12, round(min(info.width, info.height) * 0.035))
    margin = max(8, round(min(info.width, info.height) * 0.025))
    border = max(3, round(size / 3))
    filters = []
    for span in spans:
        start, end = max(interval.start, span.start), min(interval.end, span.end)
        if end <= start:
            continue
        filters.append(
            f"drawtext=text='Attempt {span.number}':expansion=none:font=Sans:fontsize={size}:"
            f"fontcolor=white:box=1:boxcolor=black@0.55:boxborderw={border}:"
            f"x={margin}:y={margin}:fix_bounds=1:"
            f"enable='gte(t,{start - interval.start:.9f})*lt(t,{end - interval.start:.9f})'"
        )
    return filters


def hdr_to_sdr_filter() -> str:
    """Map a PQ/HLG base layer through linear RGB into SDR Rec.709.

    npl is the SDR reference luminance in cd/m². Remove HDR/Dolby frame side
    data after mapping; the output must describe its new SDR pixels.
    """
    return ("zscale=transfer=linear:npl=100,format=gbrpf32le,"
            "zscale=primaries=bt709,tonemap=tonemap=hable:desat=2,"
            "zscale=transfer=bt709:matrix=bt709:range=limited,"
            "format=yuv420p,sidedata=mode=delete")


def _run(command: list[str]):
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise MediaError(exc.stderr.strip()[-4000:] or "FFmpeg command failed") from exc


def probe(path: Path) -> VideoInfo:
    require_tools()
    payload = json.loads(_run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ]).stdout)
    streams = payload.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")]
    if not videos:
        raise MediaError("Input has no playable video stream")
    # The CLI selects the first video stream; reject unusual cover-art layouts.
    first_video = next(s for s in streams if s.get("codec_type") == "video")
    if first_video is not videos[0]:
        raise MediaError("Inputs with a cover-art video before the real video are unsupported")
    video = videos[0]
    try:
        start = float(video.get("start_time", 0))
        duration_value = video.get("duration")
        if duration_value in (None, "N/A"):
            duration_value = payload.get("format", {}).get("duration")
            format_start = float(payload.get("format", {}).get("start_time", 0))
            duration_value = float(duration_value) + format_start - start
        duration = float(duration_value)
        width, height = int(video["width"]), int(video["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaError("Input must have valid dimensions, start time, and duration") from exc
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(start) or min(width, height) <= 0:
        raise MediaError("Input has invalid dimensions or timestamps")
    rotation = int(float(video.get("tags", {}).get("rotate", 0)))
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            rotation = int(round(float(side["rotation"])))
    if rotation % 90:
        raise MediaError("Only right-angle phone rotation metadata is supported")
    if rotation % 180:
        width, height = height, width
    return VideoInfo(width, height, duration, start,
                     any(s.get("codec_type") == "audio" for s in streams), rotation,
                     video.get("color_transfer") in ("smpte2084", "arib-std-b67"))


def analysis_size(info: VideoInfo, max_width: int) -> tuple[int, int]:
    width = min(info.width, max_width)
    return width, max(1, round(info.height * width / info.width))


def frames(path: Path, width: int, height: int, fps: float,
           scratch_dir: Path | None = None):
    """Yield grayscale frames on a regular, video-relative timeline, even for VFR.

    FFmpeg applies phone rotation before scaling. Reading stderr into a file avoids
    pipe deadlocks; the generator always closes its child, including on errors.
    """
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-copyts", "-i", str(path),
        "-map", "0:v:0", "-an", "-sn", "-dn", "-vf",
        f"setpts=PTS-STARTPTS,fps={fps:.12g}:start_time=0:eof_action=pass,scale={width}:{height}:flags=area,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"
    ]
    with tempfile.TemporaryFile(dir=scratch_dir) as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            size = width * height
            while True:
                data = bytearray()
                while len(data) < size:
                    chunk = process.stdout.read(size - len(data))
                    if not chunk:
                        break
                    data.extend(chunk)
                if not data:
                    break
                if len(data) != size:
                    raise MediaError("Decoder returned an incomplete frame")
                yield np.frombuffer(data, dtype=np.uint8).reshape(height, width)
            if process.wait() != 0:
                errors.seek(0)
                raise MediaError(errors.read().decode(errors="replace")[-4000:])
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


@contextmanager
def staged_path(destination: Path):
    """Write beside the destination, then atomically publish without partial files."""
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=destination.suffix,
                                      dir=destination.parent)
    import os
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def publish(temporary: Path, destination: Path, overwrite: bool):
    if overwrite:
        temporary.replace(destination)
    else:
        # An atomic, exclusive hard link also guards against a late overwrite race.
        import os
        os.link(temporary, destination)


def render(source: Path, destination: Path, info: VideoInfo, intervals: Sequence[Interval],
           overwrite: bool = False, crf: int = 18, preset: str = "medium",
           progress: ProgressCallback | None = None, scratch_dir: Path | None = None,
           *, attempt_spans: Sequence[AttemptSpan] | None = None):
    if not intervals:
        raise ValueError("Cannot render an empty edit")
    if info.hdr:
        require_hdr_filters()
    if attempt_spans:
        require_counter_filter()
    # Encode intervals separately, then join without a second video encode. This
    # keeps memory/filter-graph size bounded even for hundreds of attempts.
    with tempfile.TemporaryDirectory(prefix="skatecut-", dir=scratch_dir) as directory:
        directory = Path(directory)
        for index, interval in enumerate(intervals):
            emit(progress, "render", f"  Rendering clip {index + 1}/{len(intervals)} "
                 f"({interval.start:.2f}s → {interval.end:.2f}s)", index, len(intervals), "clips")
            # -copyts keeps the source offsets until both tracks are put onto the
            # video timeline. Padding audio handles silent/short audio streams.
            origin = info.video_start + interval.start
            vfilter = (f"[0:v:0]setpts=PTS-({origin:.9f})/TB,"
                       f"trim=start=0:end={interval.duration:.9f},")
            if info.hdr:
                vfilter += hdr_to_sdr_filter() + ","
            vfilter += "pad=ceil(iw/2)*2:ceil(ih/2)*2,setsar=1"
            if attempt_spans:
                for overlay in counter_filters(info, interval, attempt_spans):
                    vfilter += "," + overlay
            vfilter += "[v]"
            filters = [vfilter]
            # Seek to a preceding keyframe and let the timestamp filters make
            # the exact cut. This also avoids FFmpeg's accurate-seek offset
            # interaction on inputs whose first timestamp is nonzero.
            command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-copyts",
                       "-seek_timestamp", "1", "-ss", f"{origin:.9f}", "-noaccurate_seek",
                       "-i", str(source)]
            if info.has_audio:
                filters.append(
                    f"[0:a:0]asetpts=PTS-({origin:.9f})/TB,"
                    "aresample=async=1:first_pts=0,apad,"
                    f"atrim=start=0:end={interval.duration:.9f}[a]"
                )
            command += ["-filter_complex", ";".join(filters), "-map", "[v]"]
            if info.has_audio:
                command += ["-map", "[a]", "-c:a", "pcm_s16le"]
            command += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset,
                        "-pix_fmt", "yuv420p", "-fps_mode", "vfr", "-map_metadata", "-1",
                        "-metadata:s:v:0", "rotate=0"]
            if info.hdr:
                command += ["-color_primaries", "bt709", "-color_trc", "bt709",
                            "-colorspace", "bt709", "-color_range", "tv"]
            command += [str(directory / f"part-{index:06d}.mkv")]
            _run(command)
        manifest = directory / "parts.ffconcat"
        manifest.write_text("ffconcat version 1.0\n" + "".join(
            f"file part-{index:06d}.mkv\nduration {interval.duration:.9f}\n"
            for index, interval in enumerate(intervals)
        ))
        with staged_path(destination) as temporary:
            emit(progress, "render", "All clips encoded.", len(intervals), len(intervals), "clips")
            emit(progress, "joining", "  Joining clips and encoding audio…")
            command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "1",
                       "-i", str(manifest), "-map", "0:v:0", "-c:v", "copy"]
            if info.has_audio:
                command += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k"]
            command += ["-t", f"{sum(i.duration for i in intervals):.9f}",
                        "-movflags", "+faststart", str(temporary)]
            _run(command)
            publish(temporary, destination, overwrite)

