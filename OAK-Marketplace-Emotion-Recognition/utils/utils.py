import skvideo.io
from typing import List
import numpy as np

FFMPEG_OUTPUT_DICT = {
    "-vcodec": "libx265",
    "-vf": "format=yuv420p",
    "-movflags": "+faststart",
}


def save_video(visualization_results: List[np.ndarray], save_path: str = "inference_results.mp4") -> None :
    writer = skvideo.io.FFmpegWriter(save_path, outputdict=FFMPEG_OUTPUT_DICT)
    for frame in visualization_results:
        writer.writeFrame(frame[:, :, ::-1])
    writer.close()
