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

Y_OFFSET_PERCENT = 0.05
OVERLAY_PATH = "images/overlay.png"


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


class EmotionAnalyzer:
    def __init__(self, out_statistic_path: str = "result_statistic.json",
                 visualization_shape: int = 300,
                 report_statistics_shape: int = 720,
                 emotion_bar_offset_percent: float = 0.1,
                 emotion_images_path: str = "images/emoji"):
        self.emotion_bar_offset_percent = emotion_bar_offset_percent
        self.output_statistic_path = out_statistic_path
        self.bar_size, self.padding, emotion_size = self.relative_size_estimation(visualization_shape)
        self.vis_shape = visualization_shape
        self.report_statistics_shape = report_statistics_shape
        self.bars = {
            "happy": EmotionBar(emotion='happy', size=self.bar_size,
                                emotion_size=emotion_size,
                                emotion_path= os.path.join(emotion_images_path, 'happy.png')),
            "sad": EmotionBar(emotion='sad', size=self.bar_size, emotion_size=emotion_size,
                              emotion_path= os.path.join(emotion_images_path, 'sad.png')),
            "anger": EmotionBar(emotion='anger', size=self.bar_size, emotion_size=emotion_size,
                                emotion_path= os.path.join(emotion_images_path, 'anger.png')),
            "surprise": EmotionBar(emotion='surprise', size=self.bar_size, emotion_size=emotion_size,
                                   emotion_path= os.path.join(emotion_images_path, 'surprise.png')),
            "neutral": EmotionBar(emotion='neutral', size=self.bar_size, emotion_size=emotion_size,
                                  emotion_path= os.path.join(emotion_images_path, 'neutral.png'))
        }
        self.fill_bars()

    def relative_size_estimation(self, visualization_size, bar_amount: int = 5):
        bar_section_size = visualization_size / bar_amount
        bar_size = int(bar_section_size - 2 * bar_section_size * self.emotion_bar_offset_percent)
        padding = int(bar_section_size - bar_size)
        emotion_size = (int(bar_size / 3), int(bar_size / 3))
        return bar_size, padding,  emotion_size

    def fill_bars(self):
        for x in self.bars.values():
            x.update_bar()

    def get_emotions_total_amount(self):
        return sum([self.bars[x].update_count for x in self.bars.keys()])

    def update_bars(self, emotion):
        self.bars[emotion].update_count = self.bars[emotion].update_count + 1
        # respectively update all others bars
        for emotion in self.bars.keys():
            progress = self.bars[emotion].update_count / self.get_emotions_total_amount()
            self.bars[emotion].update_bar(progress)

    def dump_statistic_to_json(self):
        with open(self.output_statistic_path, 'w') as outfile:
            statistic = {x: y.update_count for x, y in self.bars.items()}
            json.dump(statistic, outfile, indent=4, sort_keys=True)

    def get_current_emotion_percent(self, emotion):
        if self.get_emotions_total_amount() != 0:
            return int((self.bars[emotion].update_count / self.get_emotions_total_amount()) * 100)
        else:
            return 0

    def create_statistic_pie_chart(self, save: bool = False, save_path: str = 'result_statistic_pie_chart.png'):
        statistic = {x: y.update_count for x, y in self.bars.items()}
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
            plt.savefig(save_path)
        return plt

    def get_current_top_bar_parameters(self):
        top_emotion, _ = sorted({x: y.update_count for x, y in self.bars.items()}.items(), key=operator.itemgetter(1))[-1]
        return self.bars[top_emotion].progress, top_emotion

    def visualization_update(self, image, ret):
        overlay = cv2.imread(OVERLAY_PATH, cv2.IMREAD_UNCHANGED)
        if self.vis_shape != 300:
            overlay = cv2.resize(overlay, (self.vis_shape, self.vis_shape))
        if ret:
            class_name = ret[0].emotions[0].class_name
            self.update_bars(class_name)
        vis_result = draw_emotion_recognition_one_frame(image, ret)
        y_offset = int(vis_result.shape[1] * Y_OFFSET_PERCENT)
        start_x, start_y = np.clip(vis_result.shape[0] - self.bar_size - int(self.padding / 2), 0, vis_result.shape[0]), \
                           np.clip(vis_result.shape[1] - self.bar_size - y_offset, 0,
                                   vis_result.shape[0])
        alpha_mask_overlay = overlay[:, :, 3] / 255.0
        overlay_background = overlay[..., :3]
        overlay_image_alpha(vis_result, overlay_background, 0, 0, alpha_mask_overlay)
        for emotion, emotion_bar in self.bars.items():
            percent = self.get_current_emotion_percent(emotion)
            text = f'{emotion.upper()} - {percent}%'
            vis_result = add_class_names_and_percents(vis_result, [start_x, int(vis_result.shape[1] - y_offset / 2)],
                                                      text)
            alpha_mask = emotion_bar.alpha_mask
            overlay_image_alpha(vis_result, emotion_bar.bar, start_x, start_y, alpha_mask)
            start_x = np.clip(start_x - self.bar_size - self.padding, 0, vis_result.shape[0])
        return vis_result

    def create_result_emotion_bar(self, save: bool = False, save_path: str = 'result_statistic_emotion_bar.png'):
        result_statistic = np.zeros((self.report_statistics_shape, self.report_statistics_shape, 3), dtype=np.uint8)
        result_statistic = add_info(result_statistic, [0, int(0.1 * self.report_statistics_shape)], BACKGROUND_COLOR,
                                    'STATISTICS', WHITE_TEXT_COLOR)
        # create top bar
        top_bar_progress, top_emotion = self.get_current_top_bar_parameters()
        top_bar_size = int(self.report_statistics_shape / 4)
        top_bar_emotion_size = (int(top_bar_size / 3), int(top_bar_size / 3))
        top_bar = EmotionBar(size=top_bar_size,
                             emotion_size=top_bar_emotion_size, emotion=top_emotion)
        top_bar.update_bar(progress=top_bar_progress)
        top_bar_x = int(self.report_statistics_shape / 2 - top_bar.width / 2)
        top_bar_y = int(self.report_statistics_shape / 2 - top_bar.height / 2)
        text = f'MOSTLY {top_bar.emotion.upper()} - {self.get_current_emotion_percent(top_bar.emotion)}%'

        y_offset = int(self.report_statistics_shape * Y_OFFSET_PERCENT)

        # place text with top emotion and percent
        result_statistic = add_class_names_and_percents(result_statistic,
                                                          [top_bar_x,
                                                           int(top_bar_y - y_offset / 2)],
                                                           text)
        # overlay bar on result statistic image
        overlay_image_alpha(result_statistic, top_bar.bar, top_bar_x, top_bar_y, top_bar.alpha_mask)

        # calculate new relative bar parameters
        result_bar_size, result_bar_padding, result_emotion_size = self.relative_size_estimation(
                                                                        self.report_statistics_shape, bar_amount=4)

        start_x, start_y = np.clip(self.report_statistics_shape - result_bar_size - int(result_bar_padding / 2), 0,
                                   self.report_statistics_shape), \
                           np.clip(self.report_statistics_shape - result_bar_size - y_offset, 0,
                                   self.report_statistics_shape)
        
        for emotion, emotion_bar in self.bars.items():
            if emotion == top_emotion:
                continue
            percent = self.get_current_emotion_percent(emotion)
            text = f'{emotion.upper()} - {percent}%'
            cur_progress = emotion_bar.progress
            cur_bar = EmotionBar(size=result_bar_size, emotion_size=result_emotion_size)
            cur_bar.update_bar(progress=cur_progress)

            result_statistic = add_class_names_and_percents(result_statistic,
                                                            [start_x, int(result_statistic.shape[1] - y_offset / 2)],
                                                            text)
            alpha_mask = cur_bar.alpha_mask
            overlay_image_alpha(result_statistic, cur_bar.bar, start_x, start_y, alpha_mask)
            start_x = np.clip(start_x - result_bar_size - result_bar_padding, 0, result_statistic.shape[0])
        if save:
            cv2.imwrite(save_path, result_statistic)
        return Image.fromarray(result_statistic[:, :, ::-1])


class EmotionBar:
    def __init__(
        self, background_color: tuple = (28, 28, 30),
        progress_color: tuple = (145, 82, 225),
        size: int = 80,
        thickness: int = 10,
        start_angle: int = -220,
        end_angle: int = 40,
        emotion: str = 'happy',
        update_count: int = 0,
        progress: float = 0.0,
        emotion_size: tuple = (25, 25),
        emotion_path: str = 'images/emoji/happy.png',
    ):
        self.__update_count = update_count
        self.__progress = progress
        self.emotion_image = np.array(Image.open(emotion_path).resize(emotion_size))
        self.size = size
        self.emotion_size = emotion_size
        self.emotion = emotion
        self.background_color = background_color
        self.progress_color = progress_color
        self.thickness = thickness
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.bar = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        self.alpha_mask = np.ones((self.size, self.size, 3), dtype=np.uint8)
        self.width, self.height, self.radius, self.center, self.axes = self.calculate_bar_parameters()

    @property
    def update_count(self):
        return self.__update_count

    @property
    def progress(self):
        return self.__progress

    @update_count.setter
    def update_count(self, value):
        self.__update_count = value

    @progress.setter
    def progress(self, value):
        self.__progress = value

    def calculate_bar_parameters(self):
        height, width = self.bar.shape[0:2]
        radius = self.size // 2
        center = (width // 2, width // 2)
        axes = (radius, radius)
        return width, height, radius, center, axes

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
        alpha_mask = self.emotion_image[:, :, 3] / 255.0
        overlay_emotion = self.emotion_image[:, :, :3][:,:,::-1]
        overlay_image_alpha(
            self.bar, overlay_emotion, int(self.center[0] - self.emotion_image.shape[0] / 2),
            int(self.center[1] - self.emotion_image.shape[1] / 2), alpha_mask,
        )
        self.alpha_mask = (~(self.bar == np.zeros((self.size, self.size, 3), dtype=np.uint8))[:, :, 0]).astype(int)



