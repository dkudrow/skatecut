"""Generate five forward passes/returns and one long in-frame turnback.

The moving rectangle represents a skater. It is synthetic test footage, not an
action-recognition benchmark. Requires NumPy and FFmpeg, like Skatecut itself.
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
    width, height, fps, seconds = 320, 180, 10, 46
    command = ["ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "gray",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-f", "lavfi", "-i",
               f"sine=frequency=880:sample_rate=48000:duration={seconds}", "-c:v", "libx264",
               "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", str(seconds), str(output)]
    y, x = np.indices((height, width))
    background = (40 + (x // 20 + y // 20) % 2 * 10).astype(np.uint8)
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        try:
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
                frame[5:9, 5 + i % 12:9 + i % 12] = 150
                process.stdin.write(frame.tobytes())
        finally:
            process.stdin.close()
            process.stdin = None
        if process.wait() != 0:
            raise SystemExit("FFmpeg failed to generate the demo")
    print(output)


if __name__ == "__main__":
    main()
