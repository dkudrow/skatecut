from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


@pytest.fixture(scope="session")
def ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe required for media integration tests")


def make_video(path: Path, seconds=8, fps=10, events=(), audio=False, tiny_motion=False):
    """A textured fixed scene with a large passing object and optional tiny noise."""
    width, height = 320, 180
    y, x = np.indices((height, width))
    background = (40 + (x // 20 + y // 20) % 2 * 10).astype(np.uint8)
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo",
               "-pix_fmt", "gray", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0"]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=880:sample_rate=48000:duration={seconds}",
                    "-c:a", "aac"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
                "-t", str(seconds), str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(round(seconds * fps)):
            frame = background.copy()
            t = i / fps
            if tiny_motion:
                left = 10 + i % 20
                frame[10:14, left:left + 4] = 150
            for start, end in events:
                if start <= t < end:
                    left = int(20 + (t - start) / (end - start) * 220)
                    frame[65:145, left:left + 40] = 220
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode()
        assert process.wait() == 0, error
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stderr.close()
    return path


@pytest.fixture(scope="session")
def active_video(tmp_path_factory, ffmpeg):
    return make_video(tmp_path_factory.mktemp("media") / "session.mp4", seconds=24,
                      events=[(0, 1), (8, 9), (16, 17), (23, 24)], audio=True, tiny_motion=True)

