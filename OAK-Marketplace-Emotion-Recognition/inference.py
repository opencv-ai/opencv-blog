import cv2
import numpy as np
from typing import List
from emotion_analyzer import EmotionAnalyzer
from modelplace_api.objects import EmotionLabel
from emotion_recognition_retail import InferenceModel
from oak_inference_utils.inference import process_frame


def resize_emotion_bboxes(img: np.ndarray, results: List[EmotionLabel], target_size: int) -> List[EmotionLabel]:
    img_h, img_w, _ = img.shape
    if img_h / img_w < target_size / target_size:
        scale = img_h / target_size
    else:
        scale = img_w / target_size
    for ret in results:
        ret.bbox.x1 = int(ret.bbox.x1 / scale)
        ret.bbox.x2 = int(ret.bbox.x2 / scale)
        ret.bbox.y1 = int(ret.bbox.y1 / scale)
        ret.bbox.y2 = int(ret.bbox.y2 / scale)
    return results


def process_cam(
    model: InferenceModel,
    emotion_analyzer: EmotionAnalyzer,
    show: bool = True,
    visualization_size: int = 300,
)-> List[np.ndarray]:
    visualization_results = []

    # build camera object and add it to OAK pipeline
    model.add_cam_to_pipeline()
    proceed = True
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

        # resize image and model inference results
        if visualization_size != input_width:
            if ret:
                ret = resize_emotion_bboxes(image, ret, visualization_size)
            image = cv2.resize(image, (visualization_size, visualization_size))

        # update emotion statistic and visualize it using emotion bar
        vis_result = emotion_analyzer.draw(image, ret)
        visualization_results.append(vis_result)

        # show processed image
        if show:
            cv2.imshow("result", vis_result)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                proceed = False

    return visualization_results
