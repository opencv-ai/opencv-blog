from typing import List

import numpy as np
from modelplace_api.objects import EmotionLabel
from modelplace_api.visualization import MONTSERATT_BOLD_TTF_PATH
from PIL import Image, ImageDraw, ImageFont

INFO_TEXT_SIZE = 12
NORM_HEIGHT = 591
TEXT_OFFSET_X = 16
TEXT_OFFSET_Y = 8
WHITE_TEXT_COLOR = (255, 255, 255, 1)


def resize_emotion_bboxes(
    img: np.ndarray, results: List[EmotionLabel], target_size: int,
) -> List[EmotionLabel]:
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


def add_class_names_and_percents(image, coords: list, text: str) -> np.ndarray:
    img_h, img_w, _ = image.shape
    scale = min([img_w, img_h]) / NORM_HEIGHT
    text_size = int(scale * INFO_TEXT_SIZE)
    text_offset_y = int(scale * TEXT_OFFSET_Y)
    text_offset_x = int(scale * TEXT_OFFSET_X)
    coords = (
        coords[0] - int(2 * scale) + text_offset_x,
        coords[1] + int(2 * scale) - text_offset_y,
    )
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    montserrat = ImageFont.truetype(MONTSERATT_BOLD_TTF_PATH, text_size)
    draw.text(coords, text, font=montserrat, fill=WHITE_TEXT_COLOR)
    return np.array(pil_img)


def overlay_image(
    img, img_overlay: np.ndarray, x: int, y: int, alpha_mask: np.ndarray = None,
) -> np.ndarray:
    source_image = img.copy()
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return source_image

    img_crop = source_image[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha
    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
    return source_image
