import numpy as np
from modelplace_api.visualization import (
    INFO_TEXT_SIZE,
    MONTSERATT_BOLD_TTF_PATH,
    NORM_HEIGHT,
    TEXT_OFFSET_X,
    TEXT_OFFSET_Y,
    WHITE_TEXT_COLOR,
)
from PIL import Image, ImageDraw, ImageFont


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


def overlay_image(img, img_overlay: np.ndarray, x: int, y: int, alpha_mask: np.ndarray = None) -> np.ndarray:
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
