from time import time
from typing import List, Tuple
from loguru import logger

import cv2
import numpy as np
from emotion_analyzer import EmotionAnalyzer
from emotion_recognition_retail import InferenceModel
from modelplace_api.objects import EmotionLabel
from oak_inference_utils.inference import process_frame
from utils.visualization import resize_emotion_bboxes


def process_cam(
    model: InferenceModel,
    emotion_analyzer: EmotionAnalyzer,
    show: bool = True,
) -> Tuple[List[np.ndarray], List[EmotionLabel], float]:
    original_images = []
    results = []

    # build camera object and add it to OAK pipeline
    model.add_cam_to_pipeline()
    proceed = True
    start = time()
    try:
        logger.info("Processing ...")
        while proceed:
            # get input shapes of emotion recognition model
            input_width, input_height = model.get_input_shapes()

            # grab frame from camera and reshape it into model input shapes
            image = np.ascontiguousarray(
                model.get_frame_from_camera()
                .reshape((3, input_height, input_width))
                .transpose(1, 2, 0),
            )

            # model inference
            ret, proceed = process_frame(image, model, visualization_func=None)

            # save original images and inference results
            original_images.append(image)
            results.append(ret)

            # update emotion statistics
            emotion_analyzer.update_bars(ret)

            # show processed image
            if show:
                # visualize bars
                vis_result = emotion_analyzer.draw(image, ret)
                cv2.imshow("result", vis_result)
                if cv2.waitKey(1) == ord("q"):
                    cv2.destroyAllWindows()
                    proceed = False

    except KeyboardInterrupt:
        logger.info("Interrupted!")

    elapsed_time = time() - start
    fps = round(len(original_images) / elapsed_time, 4)
    return original_images, results, fps
