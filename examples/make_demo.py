"""Generate a small fixed-camera test recording, without external assets.

Run from the project root:
    python examples/make_demo.py demo/input.mp4
    python -m skatecut demo/input.mp4 -o demo/highlights.mp4

The large passing rectangle stands in for a skater. The tiny moving patch
represents background motion. Four passes occur at 0, 8, 16, and 23 seconds.
"""

import argparse
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().absolute()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height, fps, duration = 320, 180, 10, 24
    y, x = np.indices((height, width))
    background = (40 + (x // 20 + y // 20) % 2 * 10).astype(np.uint8)
    events = [(0, 1), (8, 9), (16, 17), (23, 24)]
    command = ["ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "rawvideo",
               "-pix_fmt", "gray", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
               "-f", "lavfi", "-i", f"sine=frequency=880:sample_rate=48000:duration={duration}",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
               "-c:a", "aac", "-t", str(duration), str(output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        try:
            for i in range(duration * fps):
                frame = background.copy()
                time = i / fps
                left = 10 + i % 20
                frame[10:14, left:left + 4] = 150
                for start, end in events:
                    if start <= time < end:
                        left = int(20 + (time - start) / (end - start) * 220)
                        frame[65:145, left:left + 40] = 220
                process.stdin.write(frame.tobytes())
        finally:
            process.stdin.close()
            process.stdin = None
        if process.wait() != 0:
            raise SystemExit("FFmpeg failed to generate the demo")
    print(output)


if __name__ == "__main__":
    main()
