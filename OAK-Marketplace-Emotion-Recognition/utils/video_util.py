import skvideo.io

FFMPEG_OUTPUT_DICT = {
    "-vcodec": "libx265",
    "-vf": "format=yuv420p",
    "-movflags": "+faststart",
}


def save_video(visualization_results, save_path: str = "inference_results.mp4"):
    writer = skvideo.io.FFmpegWriter(save_path, outputdict=FFMPEG_OUTPUT_DICT,
    )
    for frame in visualization_results:
        writer.writeFrame(frame[:, :, ::-1])
    writer.close()