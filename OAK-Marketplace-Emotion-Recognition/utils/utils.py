from typing import List
from loguru import logger

import cv2
import numpy as np
import skvideo.io
from emotion_analyzer import EmotionAnalyzer
from modelplace_api import EmotionLabel
from tqdm import tqdm
from .visualization import resize_emotion_bboxes


def save_results_into_video(
    images: List[np.ndarray],
    results: List[EmotionLabel],
    fps: float,
    size: int = 1080,
    save_path: str = "inference_results.mp4",
) -> None:
    # resize image and model inference results
    emotion_analyzer = EmotionAnalyzer(visualization_size=size)
    output_dict = {
        "-vcodec": "libx265",
        "-vf": "format=yuv420p",
        "-movflags": "+faststart",
        "-r": f"{fps}",
    }

    logger.info("Saving the video ...")
    writer = skvideo.io.FFmpegWriter(
        save_path, outputdict=output_dict, inputdict={"-r": f"{fps}"},
    )
    for frame, ret in tqdm(zip(images, results), total=len(images)):
        ret = resize_emotion_bboxes(frame, ret, size)
        frame = cv2.resize(frame, (size, size))
        emotion_analyzer.update_bars(ret)
        frame = emotion_analyzer.draw(frame, ret)
        writer.writeFrame(frame[:, :, ::-1])
    writer.close()

    logger.info(f"Saved as {save_path}")
