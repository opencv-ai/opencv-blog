from emotion_recognition_retail import process_frame
import cv2
import numpy as np
from visualization import statistics_update


def process_cam(model, emotion_analyzer, show: bool = True):
    visualization_results = []
    model.add_cam_to_pipeline()
    proceed = True
    while proceed:
        input_width, input_height = model.get_input_shapes()
        image = np.ascontiguousarray(
            model.get_frame_from_camera()
                .reshape((3, input_height, input_width))
                .transpose(1, 2, 0),
        )
        ret, proceed, _ = process_frame(image, model, visualization_func=None)
        vis_result = statistics_update(image, ret, emotion_analyzer)
        visualization_results.append(vis_result)
        if show:
            cv2.imshow('result', vis_result)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                proceed = False
    return visualization_results
