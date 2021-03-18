from emotion_recognition_retail import process_frame
import cv2
import numpy as np


def resize_emotion_label(img, ret, target_shape):
    img_h, img_w, _ = img.shape
    if img_h / img_w < target_shape[1] /target_shape[0]:
        scale = img_h / target_shape[1]
    else:
        scale = img_w / target_shape[0]
    ret[0].bbox.x1 = int(ret[0].bbox.x1 / scale)
    ret[0].bbox.x2 = int(ret[0].bbox.x2 / scale)
    ret[0].bbox.y1 = int(ret[0].bbox.y1 / scale)
    ret[0].bbox.y2 = int(ret[0].bbox.y2 / scale)
    return ret


def process_cam(model, emotion_analyzer, show: bool = True, visualization_shape: tuple = (300, 300)):
    visualization_results = []
    # build camera object and add it to OAK pipeline
    model.add_cam_to_pipeline()
    proceed = True
    while proceed:
        # get input shapes of emotion recognition model
        input_width, input_height = model.get_input_shapes()

        # grab frame from camera and reshape it into model input shapes
        image = np.ascontiguousarray(
            model.get_frame_from_camera().reshape((3, input_height, input_width)).transpose(1, 2, 0),
        )

        # model inference
        ret, proceed = process_frame(image, model, visualization_func=None)

        # resized image and model inference results
        if visualization_shape != (300, 300):
            if ret:
                ret = resize_emotion_label(image, ret, visualization_shape)
            image = cv2.resize(image, tuple(visualization_shape))

        # update emotion statistic and visualize it using emotion bar
        vis_result = emotion_analyzer.statistics_update(image, ret)
        visualization_results.append(vis_result)

        # show processed image
        if show:
            cv2.imshow('result', vis_result)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                proceed = False

    return visualization_results
