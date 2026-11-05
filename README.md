# Skatecut

A Python command-line prototype that removes quiet periods from a fixed-camera
skateboarding recording. It keeps sustained motion, with a little footage before
and after each active period, and joins the clips into an MP4 with audio.

This is a motion-based editor. It does not recognize tricks, judge success, or
distinguish a trick attempt from someone walking through the scene.

## Install

Requires Python 3.11 or newer, plus `ffmpeg` and `ffprobe` on your PATH. FFmpeg 6+
is recommended. The prototype was tested with Python 3.14 and FFmpeg 8 on macOS.

### Prerequisites

Install Python, Git, and FFmpeg using your system package manager. On macOS
with Homebrew:

```bash
brew install python git ffmpeg
```

On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip git ffmpeg
```

Check `python3 --version`; it must be 3.11 or newer. If your Linux distribution
provides an older version, install a newer Python before continuing.

FFmpeg supplies both `ffmpeg` and `ffprobe`. Pip installs the Python dependencies
automatically, but does not install these system executables.

### Install directly from GitHub with pip

Install from the [Skatecut GitHub repository](https://github.com/dkudrow/skatecut)
in an isolated Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/dkudrow/skatecut.git"
skatecut --help
```

Run the tool with the environment active:

```bash
skatecut session.mov -o highlights.mp4
```

In a new terminal, activate the environment again using
`source /path/to/.venv/bin/activate`. Run `deactivate` to leave it.

To install a particular release, append its tag to the repository URL, for
example `git+https://github.com/dkudrow/skatecut.git@v0.1.0` once that tag exists.
To update an installation from the repository's default branch:

```bash
python -m pip install --upgrade "git+https://github.com/dkudrow/skatecut.git"
```

### Install from a local checkout

For development, run these commands from the directory containing
`pyproject.toml`. The editable installation picks up changes to the Python source:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

OpenCV's headless package is used; no desktop GUI or model download is required.
Processing stays on your computer. No footage is uploaded.

## Use

```bash
python -m skatecut session.mov -o highlights.mp4
```

The installed `skatecut` command works too. Quote paths containing spaces.
If `-o` is omitted, the video is written next to the input as
`session.highlights.mp4`.

Each run also creates two reports beside the requested output:

- `highlights.segments.json`: retained source-time intervals, settings, video
   metadata, color conversion, retained/removed duration, and whether rendering finished.
- `highlights.scores.csv`: motion scores, thresholds, and a retained-sample flag
  on the analysis timeline. Open it in a spreadsheet to inspect the detector.

Reports include a `schema_version` and the export settings. Ordinary motion
reports use version 1; sequence-filtered analyses use version 2. The CLI includes
local source/output paths for convenience; the Python API's JSON summary omits
machine-local paths and frame-level scores.

The source video is never modified. Existing outputs and reports are rejected
unless you pass `--overwrite`. If no sustained activity is detected, reports are
written and no output video is created. Reports also remain available if export
fails. A no-activity result returns exit status 0; invalid inputs and media errors
return 1, and interruption returns 130.

If `--overwrite` is used and the new analysis finds no activity, any older video
at that output path is left untouched. The new report has `rendered: false`.

### Analyze before rendering

```bash
python -m skatecut session.mov -o highlights.mp4 --analyze-only
python -m skatecut session.mov -o highlights.mp4 --overwrite
```

The second command reruns analysis and renders. Saved JSON is an inspection
report, not currently an editable input to the renderer.

### Focus on the obstacle

Use `--roi X Y WIDTH HEIGHT` with normalized coordinates in the displayed image
after phone rotation. The origin `(0, 0)` is the top left; `(1, 1)` is the bottom
right. For example, analyze the middle half of the picture:

```bash
python -m skatecut session.mov -o highlights.mp4 --roi 0.25 0.25 0.5 0.5
```

The exported video keeps the full frame. ROI limits only motion analysis. Motion
and component area thresholds are measured relative to the ROI area, so a smaller
ROI also makes a distant skater occupy a larger fraction of the analysis image.

### Tune a recording

```bash
# Keep smaller or subtler movements.
python -m skatecut session.mov -o highlights.mp4 --sensitivity 2

# Require more motion when background activity triggers false clips.
python -m skatecut session.mov -o highlights.mp4 --sensitivity 0.5

# Keep longer approaches and landings.
python -m skatecut session.mov -o highlights.mp4 --pre-roll 2 --post-roll 3
```

When reusing an output name, add `--overwrite` to replace its previous reports.
Run `python -m skatecut --help` for every setting.

| Setting | Default | Meaning |
| --- | --- | --- |
| `--sample-fps` | 10 | Analysis samples per second; export retains original frame timing |
| `--analysis-width` | 640 | Maximum analysis width; original export dimensions are preserved |
| `--sensitivity` | 1 | Higher values lower the motion threshold |
| `--start-threshold` | 0.008 | Foreground fraction required to start, divided by sensitivity |
| `--stop-ratio` | 0.5 | Lower threshold used while an event is active |
| `--min-motion` | 0.3 s | Sustained motion needed to confirm a new event |
| `--quiet-time` | 0.6 s | Quiet period needed to close an event |
| `--pre-roll` / `--post-roll` | 1.5 / 2 s | Extra footage around an event |
| `--merge-gap` | 1 s | Maximum gap between padded clips that will be merged |
| `--min-component` | 0.001 | Minimum connected moving-region area relative to ROI |
| `--background-seconds` | 20 s | Approximate background adaptation timescale |
| `--crf` | 18 | H.264 quality; lower means larger, higher-quality files |
| `--preset` | medium | Encoding speed versus compression |
| `--scratch-dir` | System temp directory | Existing directory for decoder logs and intermediate clips |

## Optional approach and return filtering

For recordings where attempts generally travel across the shot, tell Skatecut
which edge the skater enters from:

```bash
python -m skatecut session.mov -o highlights.mp4 --approach-from left
```

`left` means attempts generally enter from the left and travel right; `right`
reverses that assumption. Coordinates refer to the displayed video after phone
rotation. Omit this flag for ordinary motion trimming, including transition
tricks where entering and leaving the same side is expected.

Filtering tracks substantial foreground regions and describes motion sequences.
It keeps clear forward passes, removes confidently observed reverse passes, and
can shorten a sequence containing an approach followed by a return. It does not
recognize falls, judge whether a trick was landed, or identify a person using a
learned model. Its default is to retain ambiguous activity.

Inspect the proposed edit before rendering:

```bash
python -m skatecut session.mov -o highlights.mp4 --approach-from left --analyze-only
python -m skatecut session.mov -o highlights.mp4 --approach-from left --overwrite
```

The second command reruns analysis. For in-memory reuse without decoding again,
see the Python API below.

| Sequence option | Default | Meaning |
| --- | --- | --- |
| `--approach-from left\|right` | Disabled | Expected entry side; enables sequence filtering |
| `--attempt-duration SECONDS` | Learned when reliable | Override the attempt-length target, excluding lead/trail padding |
| `--same-side trim\|keep\|omit` | trim | Policy for approach-side turnbacks; omission requires observed same-side entry/exit |
| `--trick-roi X Y WIDTH HEIGHT` | Unspecified | Optional expected trick area in normalized displayed-image coordinates |

The last three options require `--approach-from`. `--trick-roi` describes where
the trick should occur. It is separate from `--roi`, which controls motion
scoring. Spatial tracking always uses the full frame, so entering a motion ROI
cannot masquerade as entering the video. A confident turnback that never visits
an explicitly supplied trick area can be omitted; `--same-side keep` preserves it.

### Duration learning

The program uses clean observed traversals in the expected direction as duration
references. Here, "clean" describes tracking and geometry, not trick success.
References exclude recording-boundary sequences, ambiguous tracks, visible
multiple approaches, substantial backward travel, and long pauses. It removes
extreme duration outliers, requires at least five usable references, and computes
a median plus a robust variability allowance. If the durations vary too widely,
automatic duration trimming is unavailable; clear reverse passes can still be
removed. The console and report explain the reason.

To supply a target yourself:

```bash
python -m skatecut session.mov -o highlights.mp4 \
  --approach-from left --attempt-duration 5
```

This does not cap every clip at five seconds. The program needs evidence of a
return/turnback and a meaningfully longer sequence before shortening it. A late
observed reversal extends the estimate to preserve a longer attempt. The normal
trailing allowance is then applied once. A brief reversal during a trick alone
does not justify trimming.

Same-side sequences are retained when there is insufficient evidence or no
reliable duration estimate. A return to visible staging followed by another
inward approach can be split after an observed pause; that start is labeled
`staging` rather than a real frame entry. Longer tracking gaps or multiple
unresolved attempts cause conservative retention. Explicitly omitted returns and
trimmed tails are barriers: padding and `--merge-gap` cannot restore those sections.

### Sequence reports

When filtering is enabled, the CLI adds `highlights.sequences.csv`. It lists each
sequence's observed times/duration, entry/exit, quality flags, keep/trim/omit
decision, reason, estimated endpoint, and final retained bounds. Untracked motion
also gets an uncertain record and follows the original motion-trimming policy.

The version-2 JSON report includes the same decisions, final retained intervals,
duration-model statistics, reference/outlier IDs, and compact normalized
trajectories. `trajectory_fields` names the columns in each trajectory array.
It contains numeric observations, not video frames. The existing scores CSV
continues to describe the underlying motion signal.

For example, use `--same-side omit` when same-side passes are unwanted regardless
of duration, or `--same-side keep` to retain them while still filtering clear
reverse traversals. Uncertain tracking overrides omission policies and keeps
the relevant activity.

## Python API

The CLI uses the same processing functions available to other Python programs:

```python
from skatecut import Config, RenderOptions, process_video

def on_progress(event):
    print(event.stage, event.message)

result = process_video(
    "session.mov",
    "highlights.mp4",
    config=Config(pre_roll=2, post_roll=3, sensitivity=1),
    options=RenderOptions(crf=18, preset="fast"),
    progress=on_progress,
)

print(result.rendered)             # False when no activity was detected
print(result.output)               # Path, or None if no video was exported
print(result.analysis.intervals)   # Immutable source-time intervals
print(result.to_dict())            # JSON-compatible summary, schema version 1
```

Library calls are silent by default, raise exceptions instead of exiting Python,
and write only the requested video. JSON and CSV sidecars are CLI behavior; a
library caller chooses how to store its results. Omitting `output` from
`process_video` performs analysis only. Existing output files and source aliases
are protected unless output replacement is explicitly requested; the source is
always protected.

Analysis and export can also be called separately. An in-memory analysis can be
reused to render another output without decoding and analyzing the recording again:

```python
from skatecut import analyze_video, render_video

analysis = analyze_video("session.mov", config=Config(pre_roll=2, post_roll=3))
result = render_video(analysis, "highlights.mp4", options=RenderOptions(preset="fast"))
```

Sequence filtering uses a separate validated options object. Edits can be
replanned from the in-memory observations, including a new approach side,
duration target, turnback policy, trick area, or clip padding:

```python
from skatecut import SequenceFilterOptions, replan_video

analysis = analyze_video(
    "session.mov",
    config=Config(pre_roll=1.5, post_roll=2),
    sequence_filter=SequenceFilterOptions(approach_from="left"),
)

revised = replan_video(
    analysis,
    sequence_filter=SequenceFilterOptions("left", attempt_duration=5),
)
result = render_video(revised, "highlights.mp4")

# Recover the ordinary motion edit from the same scores.
ordinary = replan_video(analysis, sequence_filter=None)
```

`process_video` accepts the same `sequence_filter` keyword. Replanning does not
decode or modify the original analysis. Enabling filtering requires observations
collected with tracking enabled. Changing pixel-analysis settings (ROI, analysis
width/rate, component size, or background adaptation) requires new analysis.
For spatial analyses, motion thresholds and sequence segmentation settings also
require new analysis. Pre/post padding and merge gaps can be changed during
replanning. Saved JSON reports are inspection artifacts; loading them back into
an analysis object is not currently supported.

`ProgressEvent` contains `stage`, `message`, and optional `completed`, `total`, and
`unit` fields. Stages include `probe`, `background`, `analysis`, `selection`,
`render`, `joining`, and `complete`. Sequence mode also emits `tracking` and
`calibration`. Analysis reports seconds; rendering reports
clips. A UI can display those fields without parsing terminal messages. Callbacks
run synchronously on the caller's thread.

`Config` and `RenderOptions` validate settings for both entry points, including
callers constructing them from JSON. There is no global per-job state. Each
analysis owns its background model, and each export owns a unique temporary
directory. Optional `scratch_dir` arguments select an existing directory for
intermediate files, which are cleaned up after success or ordinary exceptions.
Atomic output staging remains beside the output so it works across filesystems.

### Future web integration

These functions are synchronous. A future web application should download the
input to a worker-local job directory, call `process_video` in a background
worker, and upload the returned result. It can forward progress events and
serialize `result.to_dict()` into its job database. Authentication, object
storage, quotas, retention, and worker timeouts belong to that application.

The current package remains a local CLI/library. Container images, hosted job
infrastructure, and deployment credentials can be added when hosting is needed;
they are not required for local development.

## How it works

1. FFprobe inspects dimensions, timestamps, rotation, and audio.
2. FFmpeg samples up to 31 frames distributed over the video. Their grayscale
   median estimates the background, avoiding a blind warm-up window at the start.
3. A second FFmpeg pass supplies small grayscale frames at a regular analysis
   rate. Rotation and variable-rate source timing are handled during decoding.
4. OpenCV blurs the image, compensates median exposure shifts, and applies an
   adaptive MOG2 background model. Morphological opening and connected-component
   filtering remove small isolated changes. The surviving foreground area is the
   motion score. Shadows count as motion to avoid suppressing a dark-clothed skater.
5. A time-based state machine confirms sustained motion, tolerates brief quiet
   gaps, adds padding, and merges nearby clips.
6. FFmpeg seeks to and re-encodes each selected interval, then concatenates the
   temporary clips without a second video encode. Audio offsets are preserved;
   audio is joined as PCM and encoded to AAC once. Output is H.264/AAC MP4 with
   browser-friendly streaming metadata.

HDR inputs tagged as HLG or PQ are exported as SDR Rec.709. The export linearizes
the HDR base layer with `zscale`, applies Hable tone mapping, converts its color
primaries and transfer function to Rec.709, and removes HDR frame metadata. This
requires FFmpeg's `zscale`, `tonemap`, and `sidedata` filters; analysis-only mode
does not require them. The console and JSON report identify the conversion.

Detection is approximate to the analysis sampling interval (0.1 s by default).
Export cuts fall on available source frames. The source frame cadence is retained;
10 fps analysis does not make the export a 10 fps video. Odd image dimensions are
padded by at most one pixel for H.264 compatibility. Display rotation is baked
into the exported image.

Analysis streams through the recording and retains only a small background
sample plus per-sample numeric scores. Export uses temporary disk space for the
retained clips; it does not load the full video into RAM. Temporary files are
cleaned up, and completed outputs are published atomically. Both analysis passes
decode the source recording, so very long or high-resolution inputs still take
time even at a low analysis rate.

## Current limitations

- Assumes a stationary camera. Camera bumps, large moving backgrounds, nearby
  traffic, and people resetting can create extra clips. ROI is the best first
  adjustment when the obstacle occupies a small part of the image.
- Very small, brief, low-contrast, or nearly stationary subjects can be missed.
  Start with generous padding and inspect the reports when tuning a new angle.
- The sampled median assumes the background is visible for much of the recording.
  A subject occupying the same spot for most of the recording may become part of
  the estimated background.
- HDR export produces an SDR rendition. It does not preserve HDR or reproduce
  Dolby Vision dynamic metadata. HLG/PQ-compatible base layers are used; other
  Dolby Vision profiles are not supported. Tone-mapped appearance may differ
  from the phone's native HDR playback.
- Uses the first playable video and first audio track. Does not retain subtitles,
  extra audio tracks, location metadata, or other auxiliary streams. Videos with
  cover-art streams ahead of the actual video and non-right-angle rotation are
  rejected.
- No visual timeline or manual clip review UI yet. Real skate footage is needed
  to evaluate recall and tune the defaults; synthetic checks establish pipeline
  correctness rather than real-world detection accuracy.
- Sequence filtering assumes one prominent skater. Its foreground tracker can
  lose a subject during occlusion, long stillness, a camera bump, or significant
  silhouette changes. Ambiguous tracks fall back to motion trimming. Small or
  distant subjects may never form reliable spatial tracks.
- Duration estimates are heuristics for a particular recording. They may retain
  some return/recovery footage, and need inspection when choosing new settings.
  Sampling or manual duration overrides do not establish an exact trick endpoint.

## Synthetic demo

The `demo/` directory includes a generated input, its edited highlights, and the
analysis reports. The input contains four large moving rectangles representing
passes at the obstacle plus a tiny continuously moving patch representing noise.
It includes activity at the very start and end. It is not skateboarding footage.

Regenerate the example with a new filename:

```bash
python examples/make_demo.py demo/new-input.mp4
python -m skatecut demo/new-input.mp4 -o demo/new-highlights.mp4
```

The synthetic demo has a constant test tone, rather than recorded skate audio.

The additional `demo/sequence-input.mp4` and `demo/sequence-highlights.mp4`
demonstrate five forward passes, five returns, and a long turnback. Regenerate
them under new filenames with:

```bash
python examples/make_sequence_demo.py demo/new-sequence-input.mp4
python -m skatecut demo/new-sequence-input.mp4 -o demo/new-sequence-highlights.mp4 \
  --approach-from left --pre-roll .2 --post-roll .5 --merge-gap .2
```

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest
```

The suite generates its own small videos and checks quiet scenes, tiny background
movement, attempts at the start and end, hysteresis, padding and merging, ROI,
silent video, variable frame rates, rotation, report generation, overwrite guards,
HDR-to-SDR export, and audio alignment across cuts with delayed audio and nonzero
source timestamps. API integration tests also cover structured progress, JSON
summaries, concurrent exports reusing one analysis, source-alias protection,
and cleanup/preservation of prior outputs after encoder failures.

Sequence tests add mirrored entry/exit tracking, quick returns, in-frame staging,
same-side policies, learned/manual duration targets, outliers and insufficient
references, motion/trick ROI separation, uncertain boundaries, untracked activity,
replanning, and removal barriers under large padding/merge settings. Generated
videos exercise the complete detector-to-export path and preserve labeled
must-keep trick windows.

## Layout

- `skatecut/model.py`: configuration and interval selection.
- `skatecut/pipeline.py`: reusable analysis and export API with typed results.
- `skatecut/progress.py`: structured progress events shared by callers.
- `skatecut/sequences.py`: sequence records, tracking, duration learning, and edit planning.
- `skatecut/detector.py`: background estimation and motion analysis.
- `skatecut/media.py`: FFprobe/FFmpeg decoding and export.
- `skatecut/cli.py`: command-line options and reports.
- `tests/`: interval tests and generated-video integration tests.

Algorithm and export references: [OpenCV background subtraction](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)
and [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html).
