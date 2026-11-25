"""Command-line entry point and human-readable reports."""

import argparse
import csv
import json
from pathlib import Path
import sys

from . import __version__
from .media import MediaError, publish, staged_path
from .model import Config, ENCODING_PRESETS, RenderOptions
from .pipeline import ProcessingResult, analyze_video, render_video
from .sequences import SequenceFilterOptions


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Remove quiet periods from fixed-camera skate videos.")
    p.add_argument("input", type=Path, help="Source video; never modified")
    p.add_argument("-o", "--output", type=Path, help="Output MP4 (default: INPUT.highlights.mp4)")
    p.add_argument("--analyze-only", action="store_true", help="Write reports without rendering video")
    p.add_argument("--overwrite", action="store_true", help="Replace existing outputs and reports")
    p.add_argument("--sensitivity", type=float, default=1.0,
                   help="Higher keeps more motion; lower keeps less (default: 1)")
    p.add_argument("--start-threshold", type=float, default=0.008,
                   help="Required foreground fraction before sensitivity adjustment (default: 0.008)")
    p.add_argument("--stop-ratio", type=float, default=0.5,
                   help="Stop threshold as a fraction of start threshold (default: 0.5)")
    p.add_argument("--min-motion", type=float, default=0.3, help="Seconds to confirm activity (default: 0.3)")
    p.add_argument("--quiet-time", type=float, default=0.6, help="Seconds of quiet to end activity (default: 0.6)")
    p.add_argument("--pre-roll", type=float, default=1.5, help="Seconds before activity to keep (default: 1.5)")
    p.add_argument("--post-roll", type=float, default=2.0, help="Seconds after activity to keep (default: 2)")
    p.add_argument("--merge-gap", type=float, default=1.0,
                   help="Merge padded clips separated by at most this many seconds (default: 1)")
    p.add_argument("--roi", type=float, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                   help="Obstacle area in normalized displayed-image coordinates, each 0–1")
    p.add_argument("--sample-fps", type=float, default=10, help="Analysis samples per second (default: 10)")
    p.add_argument("--analysis-width", type=int, default=640, help="Max analysis width in pixels (default: 640)")
    p.add_argument("--min-component", type=float, default=0.001,
                   help="Ignore moving regions below this fraction of ROI area (default: 0.001)")
    p.add_argument("--background-seconds", type=float, default=20,
                   help="Background adaptation timescale in seconds (default: 20)")
    p.add_argument("--crf", type=int, default=18, help="H.264 quality, 0–51; lower is better (default: 18)")
    p.add_argument("--preset", choices=ENCODING_PRESETS,
                   default="medium", help="Encoding speed/compression tradeoff (default: medium)")
    p.add_argument("--attempt-counter", action="store_true",
                   help="Overlay the current retained attempt number in the upper-left corner")
    p.add_argument("--scratch-dir", type=Path,
                   help="Existing directory for intermediate files (default: system temp directory)")
    p.add_argument("--approach-from", choices=("left", "right"),
                   help="Filter motion sequences for attempts entering from this displayed-frame edge")
    p.add_argument("--attempt-duration", type=float,
                   help="Override learned attempt-length target in seconds, excluding lead/trail padding")
    p.add_argument("--same-side", choices=("trim", "keep", "omit"), default=None,
                   help="Policy for approach-side turnbacks (default with filtering: trim)")
    p.add_argument("--trick-roi", type=float, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                   help="Optional expected trick area, separate from the motion ROI")
    p.add_argument("--version", action="version", version=f"skatecut {__version__}")
    return p


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    try:
        fields = Config.__dataclass_fields__
        values = {name: getattr(args, name) for name in fields}
        if values["roi"] is not None:
            values["roi"] = tuple(values["roi"])
        config = Config(**values)
        options = RenderOptions(crf=args.crf, preset=args.preset, attempt_counter=args.attempt_counter)
        sequence_filter = None
        if not args.approach_from and any(value is not None for value in (
                args.attempt_duration, args.same_side, args.trick_roi)):
            raise ValueError("--attempt-duration, --same-side, and --trick-roi require --approach-from")
        if args.approach_from:
            sequence_filter = SequenceFilterOptions(args.approach_from, args.attempt_duration,
                                                     args.same_side or "trim", args.trick_roi)
        source = args.input.expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("Input must be a video file")
        output = (args.output.expanduser() if args.output else source.with_name(
            source.stem + ".highlights.mp4")).absolute()
        if output.suffix.lower() != ".mp4":
            raise ValueError("Output must have an .mp4 extension")
        reports = [output.with_suffix(".segments.json"), output.with_suffix(".scores.csv")]
        if sequence_filter:
            reports.append(output.with_suffix(".sequences.csv"))
        targets = reports if args.analyze_only else [output, *reports]
        for target in targets:
            if target.resolve() == source or (target.exists() and target.samefile(source)):
                raise ValueError("An output path refers to the source video")
            if target.exists() and not args.overwrite:
                raise FileExistsError(f"Output already exists: {target}. Use --overwrite to replace it.")
        output.parent.mkdir(parents=True, exist_ok=True)
        progress = lambda event: print(event.message, flush=True)
        analysis = analyze_video(source, config=config, progress=progress, scratch_dir=args.scratch_dir,
                                 sequence_filter=sequence_filter)
        info, scores, intervals = analysis.video, analysis.scores, analysis.intervals
        kept = analysis.kept_seconds
        payload = {
            **ProcessingResult(analysis, None, options).to_dict(),
            "source": str(source), "output": str(output),
        }
        # Reports are useful even when a render fails or no activity is found.
        def write_json():
            with staged_path(reports[0]) as temporary:
                temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
                publish(temporary, reports[0], args.overwrite)

        with staged_path(reports[1]) as temporary:
            with temporary.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["time_seconds", "motion_fraction", "start_threshold", "stop_threshold", "kept"])
                interval_index = 0
                for score in scores:
                    while interval_index < len(intervals) and score.time >= intervals[interval_index].end:
                        interval_index += 1
                    selected = interval_index < len(intervals) and intervals[interval_index].start <= score.time
                    writer.writerow([f"{score.time:.6f}", f"{score.motion:.8f}",
                                     f"{config.on_threshold:.8f}", f"{config.off_threshold:.8f}", int(selected)])
            publish(temporary, reports[1], args.overwrite)
        write_json()
        if sequence_filter:
            columns = ("sequence_id", "observed_start", "observed_end", "duration", "active_duration",
                       "entry_side", "exit_side", "entry_origin", "classification", "action", "reason",
                       "estimated_end", "retained_start", "retained_end", "max_progress", "return_start",
                       "visited_trick_roi", "quality_flags")
            with staged_path(reports[2]) as temporary:
                with temporary.open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
                    writer.writeheader()
                    for sequence in payload["sequences"]:
                        writer.writerow({**sequence, "quality_flags": ";".join(sequence["quality_flags"])})
                publish(temporary, reports[2], args.overwrite)
            model = analysis.sequence_plan.duration_model
            print(f"Sequences: {len(analysis.sequence_plan.sequences)}; "
                  f"{sum(d.action == 'omit' for d in analysis.sequence_plan.decisions)} omitted, "
                  f"{sum(d.action == 'trim' for d in analysis.sequence_plan.decisions)} shortened.")
            if model.reliable:
                print(f"Attempt-length target: {model.target_seconds:.2f}s "
                      f"({model.source}; {len(model.reference_ids)} reference passes).")
            else:
                print(f"Automatic duration trimming unavailable: {model.reason}. "
                      "Clear returns can still be omitted; use --attempt-duration to supply a target.")
        print(f"Selected {len(intervals)} clip(s): {kept:.2f}s kept, "
              f"{info.duration - kept:.2f}s removed ({100 * (1 - kept / info.duration):.1f}%).", flush=True)
        for index, interval in enumerate(intervals, 1):
            print(f"  {index}: {interval.start:.2f}s → {interval.end:.2f}s")
        if not intervals:
            if sequence_filter:
                print("No clips retained after sequence filtering; no video written. "
                      "Review the sequence report or disable --approach-from.")
            else:
                print("No sustained activity detected; no video written. Try higher --sensitivity or an obstacle --roi.")
            if not args.analyze_only and output.exists():
                print(f"Existing video was left untouched: {output}. Reports describe this run.")
        elif not args.analyze_only:
            print("Rendering highlights…", flush=True)
            result = render_video(analysis, output, options=options, overwrite=args.overwrite,
                                  progress=progress, scratch_dir=args.scratch_dir)
            payload["rendered"] = result.rendered
            # This report was created by this invocation and can be updated.
            with staged_path(reports[0]) as temporary:
                temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
                publish(temporary, reports[0], overwrite=True)
            print(f"Video: {output}")
        print(f"Intervals: {reports[0]}\nMotion scores: {reports[1]}")
        if sequence_filter:
            print(f"Sequences: {reports[2]}")
        return 0
    except (MediaError, OSError, ValueError) as exc:
        print(f"skatecut: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nskatecut: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
