"""Application-independent processing API.

Callers supply local paths and optional progress callbacks. This module has no
terminal, HTTP, cloud-storage, or automatic report-writing behavior.
"""

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from . import __version__
from .detector import analyze
from .media import VideoInfo, probe, render
from .model import Config, Interval, RenderOptions, Score, select_intervals
from .progress import ProgressCallback, emit
from .sequences import (MotionSequence, Observation, SequenceFilterOptions, SequencePlan,
                        build_sequences, plan_sequences, sequence_report)


@dataclass(frozen=True)
class AnalysisResult:
    source: Path
    video: VideoInfo
    config: Config
    scores: tuple[Score, ...]
    intervals: tuple[Interval, ...]
    sequence_filter: SequenceFilterOptions | None = None
    sequence_observations: tuple[MotionSequence, ...] = ()
    sequence_plan: SequencePlan | None = None
    spatial_analyzed: bool = False

    @property
    def kept_seconds(self) -> float:
        return sum(item.duration for item in self.intervals)

    @property
    def removed_seconds(self) -> float:
        return max(0, self.video.duration - self.kept_seconds)

    def to_dict(self) -> dict:
        """JSON-compatible summary without machine-local paths or frame data."""
        result = {
            "schema_version": 2 if self.spatial_analyzed else 1,
            "version": __version__,
            "video": asdict(self.video),
            "config": asdict(self.config),
            "effective_start_threshold": self.config.on_threshold,
            "effective_stop_threshold": self.config.off_threshold,
            "intervals": [asdict(item) for item in self.intervals],
            "kept_seconds": self.kept_seconds,
            "removed_seconds": self.removed_seconds,
            "color_conversion": "hdr_to_sdr_bt709" if self.video.hdr else "none",
        }
        if self.spatial_analyzed:
            result["sequence_filter"] = asdict(self.sequence_filter) if self.sequence_filter else None
            result["duration_model"] = asdict(self.sequence_plan.duration_model) if self.sequence_plan else None
            result["sequences"] = [sequence_report(s, d) for s, d in zip(
                self.sequence_plan.sequences, self.sequence_plan.decisions)] if self.sequence_plan else []
        return result


@dataclass(frozen=True)
class ProcessingResult:
    analysis: AnalysisResult
    output: Path | None
    render_options: RenderOptions

    @property
    def rendered(self) -> bool:
        return self.output is not None

    def to_dict(self) -> dict:
        return {**self.analysis.to_dict(), "rendered": self.rendered,
                "render_options": asdict(self.render_options)}


def _source_path(source: str | Path) -> Path:
    path = Path(source).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("Input must be a video file")
    return path


def _output_path(source: Path, output: str | Path, overwrite: bool) -> Path:
    destination = Path(output).expanduser().absolute()
    if destination.suffix.lower() != ".mp4":
        raise ValueError("Output must have an .mp4 extension")
    if destination.resolve() == source or (destination.exists() and destination.samefile(source)):
        raise ValueError("An output path refers to the source video")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    return destination


def _scratch_path(scratch_dir: str | Path | None) -> Path | None:
    if scratch_dir is None:
        return None
    path = Path(scratch_dir).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError("scratch_dir must be an existing directory")
    return path


def analyze_video(source: str | Path, *, config: Config | None = None,
                  progress: ProgressCallback | None = None,
                  scratch_dir: str | Path | None = None,
                  sequence_filter: SequenceFilterOptions | None = None) -> AnalysisResult:
    """Detect intervals, without exporting or writing any reports.

    Errors propagate as exceptions. Nothing is printed unless the caller's
    progress callback prints. Numeric scores and intervals are immutable snapshots.
    """
    source = _source_path(source)
    scratch = _scratch_path(scratch_dir)
    config = config if config is not None else Config()
    video = probe(source)
    emit(progress, "probe", f"Input: {video.width} × {video.height}, {video.duration:.2f}s, "
         f"{'audio' if video.has_audio else 'no audio'}")
    observations: list[Observation] = []
    scores = analyze(source, video, config, progress=progress, scratch_dir=scratch,
                     observation_callback=observations.append if sequence_filter else None)
    raw_sequences = ()
    plan = None
    if sequence_filter is not None:
        emit(progress, "tracking", "Building motion sequences from full-frame tracks…")
        raw_sequences = build_sequences(tuple(observations), video.duration, config, sequence_filter)
        emit(progress, "calibration", "Learning attempt duration and planning sequence edits…")
        plan = plan_sequences(raw_sequences, tuple(scores), video.duration, config, sequence_filter)
    emit(progress, "selection", "Selecting active intervals…")
    intervals = plan.intervals if plan else select_intervals(scores, video.duration, config)
    return AnalysisResult(source, video, config, tuple(scores), tuple(intervals), sequence_filter,
                          raw_sequences, plan, sequence_filter is not None)


def replan_video(analysis: AnalysisResult, *, sequence_filter: SequenceFilterOptions | None,
                 config: Config | None = None) -> AnalysisResult:
    """Replan an in-memory analysis without decoding. Pass None to disable filtering.

    Only thresholds, interval timing, and sequence policies may change. Settings
    that alter pixel analysis require a new analyze_video call.
    """
    config = config if config is not None else analysis.config
    editable = {"sensitivity", "start_threshold", "stop_ratio", "min_motion", "quiet_time",
                "pre_roll", "post_roll", "merge_gap"}
    for field in fields(config):
        if field.name not in editable and getattr(config, field.name) != getattr(analysis.config, field.name):
            raise ValueError(f"Changing {field.name} requires new video analysis")
    # Track segmentation depends on confirmation/quiet thresholds too.
    for name in ("min_motion", "quiet_time", "sensitivity", "start_threshold", "stop_ratio"):
        if analysis.spatial_analyzed and getattr(config, name) != getattr(analysis.config, name):
            raise ValueError(f"Changing {name} with sequence filtering requires new video analysis")
    if sequence_filter is not None:
        if not analysis.spatial_analyzed:
            raise ValueError("Sequence filtering requires an analysis with spatial tracking enabled")
        plan = plan_sequences(analysis.sequence_observations, analysis.scores,
                              analysis.video.duration, config, sequence_filter)
        return replace(analysis, config=config, intervals=plan.intervals,
                       sequence_filter=sequence_filter, sequence_plan=plan)
    return replace(analysis, config=config, intervals=tuple(select_intervals(
        analysis.scores, analysis.video.duration, config)), sequence_filter=None, sequence_plan=None)


def render_video(analysis: AnalysisResult, output: str | Path, *,
                 options: RenderOptions | None = None, overwrite: bool = False,
                 progress: ProgressCallback | None = None,
                 scratch_dir: str | Path | None = None) -> ProcessingResult:
    """Export an in-memory analysis; it can be reused for multiple exports.

    An empty edit returns output=None and does not touch an existing output.
    Intermediate files live in a unique subdirectory of scratch_dir (or the
    system temp directory). Atomic output staging is beside the destination.
    """
    options = options if options is not None else RenderOptions()
    source = _source_path(analysis.source)
    destination = _output_path(source, output, overwrite)
    scratch = _scratch_path(scratch_dir)
    if not analysis.intervals:
        message = ("No sequences retained after filtering; no video written." if analysis.sequence_filter
                   else "No sustained activity detected; no video written.")
        emit(progress, "complete", message)
        return ProcessingResult(analysis, None, options)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if analysis.video.hdr:
        emit(progress, "render", "HDR input: export will be tone-mapped to SDR (BT.709).")
    render(source, destination, analysis.video, analysis.intervals, overwrite,
           options.crf, options.preset, progress=progress, scratch_dir=scratch)
    emit(progress, "complete", "Video export complete.")
    return ProcessingResult(analysis, destination, options)


def process_video(source: str | Path, output: str | Path | None = None, *,
                  config: Config | None = None, options: RenderOptions | None = None,
                  overwrite: bool = False, progress: ProgressCallback | None = None,
                  scratch_dir: str | Path | None = None,
                  sequence_filter: SequenceFilterOptions | None = None) -> ProcessingResult:
    """Analyze and optionally export. output=None performs analysis only.

    This is synchronous: a future web application should run it in a background
    worker. Downloading/uploads, authentication, quotas, and retention belong to
    that caller; paths here always refer to files on the worker's filesystem.
    """
    options = options if options is not None else RenderOptions()
    source = _source_path(source)
    if output is not None:
        _output_path(source, output, overwrite)  # Fail before expensive analysis.
    analysis = analyze_video(source, config=config, progress=progress, scratch_dir=scratch_dir,
                             sequence_filter=sequence_filter)
    if output is None:
        emit(progress, "complete", "Analysis complete.")
        return ProcessingResult(analysis, None, options)
    return render_video(analysis, output, options=options, overwrite=overwrite,
                        progress=progress, scratch_dir=scratch_dir)
