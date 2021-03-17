import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import operator
import os
from modelplace_api.visualization import (WHITE_TEXT_COLOR,MONTSERATT_BOLD_TTF_PATH,
                                          NORM_HEIGHT,INFO_TEXT_SIZE,
                                          TEXT_OFFSET_Y, TEXT_OFFSET_X, add_info, BACKGROUND_COLOR,
                                          draw_emotion_recognition_one_frame)

X_OFFSET = 5
Y_OFFSET = 15
OVERLAY_PATH = "images/overlay.png"


def statistics_update(image, ret, emotions_collector):
    overlay = cv2.imread(OVERLAY_PATH, cv2.IMREAD_UNCHANGED)
    if ret:
        class_name = ret[0].emotions[0].class_name
        emotions_collector.update_bars(class_name)
    image = cv2.resize(image, (450, 450))
    vis_result = draw_emotion_recognition_one_frame(image, ret)

    start_x, start_y = np.clip(vis_result.shape[0] - emotions_collector.bars_size, 0, vis_result.shape[0]), \
                       np.clip(vis_result.shape[1] - emotions_collector.bars_size - Y_OFFSET, 0,
                               vis_result.shape[0])
    alpha_mask_overlay = overlay[:, :, 3] / 255.0
    overlay_background = overlay[..., :3]
    overlay_image_alpha(vis_result, overlay_background, 0, 0, alpha_mask_overlay)
    for emotion, emotion_bar in emotions_collector.bars.items():
        percent = emotions_collector.get_current_emotion_percent(emotion)
        text = f'{emotion.upper()} - {percent}%'
        vis_result = add_class_names_and_percents(vis_result, [start_x, int(vis_result.shape[1] - Y_OFFSET / 2)],
                                                  text)
        alpha_mask = emotion_bar.alpha_mask
        overlay_image_alpha(vis_result, emotion_bar.bar, start_x, start_y, alpha_mask)
        start_x = np.clip(start_x - emotions_collector.bars_size - X_OFFSET, 0, vis_result.shape[0])
    return vis_result


def add_class_names_and_percents(image, coords, text):
    img_h, img_w, _ = image.shape
    scale = min([img_w, img_h]) / NORM_HEIGHT
    text_size = int(scale * INFO_TEXT_SIZE)
    text_offset_y = int(scale * TEXT_OFFSET_Y)
    text_offset_x = int(scale * TEXT_OFFSET_X)
    coords = coords[0] - int(2 * scale) + text_offset_x, coords[1] + int(2 * scale) - text_offset_y
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    montserrat = ImageFont.truetype(MONTSERATT_BOLD_TTF_PATH, text_size)
    draw.text(coords, text, font=montserrat, fill=WHITE_TEXT_COLOR)
    return np.array(pil_img)


def overlay_image_alpha(img, img_overlay, x, y, alpha_mask=None):
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha
    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop


class EmotionBarAnalyzer:
    def __init__(self, statistic_path: str = "result_statistic.json", emotion_size: tuple = (25, 25), size: int = 85):
        self.bars = {
            "happy": EmotionBar(emotion='happy', size=size, emotion_size=emotion_size),
            "sad": EmotionBar(emotion='sad',size=size, emotion_size=emotion_size),
            "anger": EmotionBar(emotion='anger',size=size, emotion_size=emotion_size),
            "surprise": EmotionBar(emotion='surprise',size=size, emotion_size=emotion_size),
            "neutral": EmotionBar(emotion='neutral',size=size, emotion_size=emotion_size),
        }
        self.output_statistic_path = statistic_path
        self.bars_size = size
        self.fill_bars()

    def fill_bars(self):
        for x in self.bars.values():
            x.update_bar()

    def get_emotions_total_amount(self):
        return sum([self.bars[x].emotion_amount for x in self.bars.keys()])

    def update_bars(self, emotion):
        self.bars[emotion].emotion_amount = self.bars[emotion].emotion_amount + 1
        # respectively update all others bars
        for emotion in self.bars.keys():
            progress = self.bars[emotion].emotion_amount / self.get_emotions_total_amount()
            self.bars[emotion].update_bar(progress)

    def dump_statistic_to_json(self):
        with open(self.output_statistic_path, 'w') as outfile:
            statistic = {x: y.emotion_amount for x, y in self.bars.items()}
            json.dump(statistic, outfile, indent=4, sort_keys=True)

    def get_current_emotion_percent(self, emotion):
        if self.get_emotions_total_amount() != 0:
            return int((self.bars[emotion].emotion_amount / self.get_emotions_total_amount()) * 100)
        else:
            return 0

    def create_statistic_pie_chart(self, save: bool = False):
        statistic = {x: y.emotion_amount for x, y in self.bars.items()}
        emotions = list(statistic.keys())
        emotions_amount = list(statistic.values())
        total_emotion_amount = sum(emotions_amount)
        if total_emotion_amount:
            explode = [float(x) / total_emotion_amount for x in emotions_amount]
        else:
            explode = [0.0] * len(emotions)
        plt.pie(
            emotions_amount, labels=emotions,
            startangle=90, autopct='%1.1f%%', shadow=True, explode=explode,
        )
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
        if save:
            plt.savefig('result_statistic_pie_chart.png')

    def get_current_top_bar_parameters(self):
        top_emotion, _ = sorted({x: y.emotion_amount for x, y in self.bars.items()}.items(), key=operator.itemgetter(1))[-1]
        return self.bars[top_emotion].progress, top_emotion

    def create_result_emotion_bar(self, size, save: bool = False):
        result_statistic = np.zeros((size, size, 3), dtype=np.uint8)
        result_statistic = add_info(result_statistic, [0, int(0.1 * size)], BACKGROUND_COLOR, 'STATISTICS', WHITE_TEXT_COLOR)
        top_bar_progress, top_emotion = self.get_current_top_bar_parameters()
        top_bar_size = int(size / 4)
        top_bar_emotion_size = (int(top_bar_size / 3), int(top_bar_size / 3))
        top_bar = EmotionBar(size=top_bar_size, emotion=top_emotion, emotion_size=top_bar_emotion_size)
        top_bar.update_bar(top_bar_progress)
        top_bar_x = int(size / 2 - top_bar.width / 2)
        top_bar_y = int(size / 2 - top_bar.height / 2)
        text = f'MOSTLY {top_bar.emotion.upper()} - {self.get_current_emotion_percent(top_bar.emotion)}%'
        result_statistic = add_class_names_and_percents(result_statistic,
                                                          [top_bar_x,
                                                           int(top_bar_y - 2 * Y_OFFSET / 2)],
                                                           text)
        overlay_image_alpha(result_statistic, top_bar.bar, top_bar_x, top_bar_y, top_bar.alpha_mask)
        start_x, start_y = np.clip(result_statistic.shape[0] - 2 * self.bars_size, 0, result_statistic.shape[0]) , \
                               np.clip(result_statistic.shape[1] - self.bars_size - Y_OFFSET, 0, result_statistic.shape[0])
        for emotion, emotion_bar in self.bars.items():
            if emotion == top_emotion:
                continue
            percent = self.get_current_emotion_percent(emotion)
            text = f'{emotion.upper()} - {percent}%'
            result_statistic = add_class_names_and_percents(result_statistic, [start_x, int(result_statistic.shape[1] - Y_OFFSET / 2)],
                                                        text)
            alpha_mask = emotion_bar.alpha_mask
            overlay_image_alpha(result_statistic, emotion_bar.bar, start_x, start_y, alpha_mask)
            start_x = np.clip(start_x - self.bars_size - 15 * X_OFFSET, 0, result_statistic.shape[0])
        if save:
            cv2.imwrite('result_statistic_emotion_bar.png', result_statistic)
        return result_statistic


class EmotionBar:
    def __init__(
        self, background_color: tuple = (28, 28, 30),
        progress_color: tuple = (145, 82, 225),
        size: int = 80,
        thickness: int = 10,
        start_angle: int = -220,
        end_angle: int = 40,
        emotion: str = 'happy',
        emotion_amount: int = 0,
        progress: float = 0.0,
        emotion_size: tuple = (25, 25),
        emotions_path: str = 'images/emoji',
    ):
        self.emotion_images = {
            'neutral': np.array(
                Image.open(os.path.join(emotions_path, "neutral.png")).resize(emotion_size)),
            'happy': np.array(
                Image.open(os.path.join(emotions_path, "happy.png")).resize(emotion_size),
            ),
            'sad': np.array(
                Image.open(os.path.join(emotions_path, "sad.png")).resize(emotion_size)),
            'anger': np.array(
                Image.open(os.path.join(emotions_path, "anger.png")).resize(emotion_size),
            ),
            'surprise': np.array(
                Image.open(os.path.join(emotions_path, "surprise.png")).resize(emotion_size)),
        }
        if emotion not in self.emotion_images.keys():
            raise Exception(f"emotion argument should be one of {self.emotion_images.keys()}!")
        self.emotion = emotion
        self._emotion_amount = emotion_amount
        self._progress = progress
        self.background_color = background_color
        self.progress_color = progress_color
        self.size = size
        self.thickness = thickness
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.bar = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        self.height, self.width = self.bar.shape[0:2]
        self.radius = self.size // 2
        self.center = (self.width // 2, self.width // 2)
        self.axes = (self.radius, self.radius)
        self.alpha_mask = np.ones((self.size, self.size, 3), dtype=np.uint8)

    def update_bar(self, progress: float = 0.0):
        self.progress = progress
        diff = (self.end_angle - self.start_angle) * progress
        self.bar = cv2.ellipse(
            self.bar, center=self.center, axes=self.axes, angle=0, startAngle=self.start_angle,
            endAngle=self.end_angle, color=self.background_color, thickness=-1,
        )
        self.bar = cv2.ellipse(
            self.bar, center=self.center, axes=self.axes, angle=0, startAngle=self.start_angle,
            endAngle=self.start_angle + diff, color=self.progress_color, thickness=-1,
        )
        self.bar = cv2.ellipse(
            self.bar, center=self.center, axes=(self.radius - self.thickness, self.radius - self.thickness),
            angle=0, startAngle=self.start_angle - 1, endAngle=self.end_angle + 1,
            color=(0, 0, 0), thickness=-1,
        )
        # get emotion image
        emotion_image = self.emotion_images[self.emotion]
        alpha_mask = emotion_image[:, :, 3] / 255.0
        overlay_emotion = emotion_image[:, :, :3][:,:,::-1]
        overlay_image_alpha(
            self.bar, overlay_emotion, int(self.center[0] - emotion_image.shape[0] / 2),
            int(self.center[1] - emotion_image.shape[1] / 2), alpha_mask,
        )
        self.alpha_mask = (~(self.bar == np.zeros((self.size, self.size, 3), dtype=np.uint8))[:, :, 0]).astype(int)

    @property
    def emotion_amount(self):
        return self._emotion_amount

    @property
    def progress(self):
        return self._progress

    @emotion_amount.setter
    def emotion_amount(self, value):
        self._emotion_amount = value

    @progress.setter
    def progress(self, value):
        self._progress = value