from emotion_recognition_retail import InferenceModel, process_frame
import cv2
import numpy as np
from .visualization import statistics_update


def build_model(model_path):
    model = InferenceModel(model_path=model_path)
    model.model_load()
    return model


def process_cam(model, emotion_collector, show: bool = True):
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
        vis_result = statistics_update(image, ret, emotion_collector)
        visualization_results.append(vis_result)
        if show:
            cv2.imshow('result', vis_result)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                proceed = False
    return visualization_results
