import skvideo.io
import os

FFMPEG_OUTPUT_DICT = {
    "-vcodec": "libx265",
    "-vf": "format=yuv420p",
    "-movflags": "+faststart",
}


def save_video(visualization_results):
    writer = skvideo.io.FFmpegWriter("inference_results.mp4", outputdict=FFMPEG_OUTPUT_DICT,
    )
    for frame in visualization_results:
        writer.writeFrame(frame[:, :, ::-1])
    writer.close()