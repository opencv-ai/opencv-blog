import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import operator
import os
from modelplace_api.visualization import (WHITE_TEXT_COLOR, add_info, BACKGROUND_COLOR,
                                          draw_emotion_recognition_one_frame)
from utils.visualization import overlay_image, add_class_names_and_percents


class EmotionAnalyzer:
    def __init__(self, out_statistics_path: str = "result_statistic.json",
                 visualization_size: int = 300,
                 report_statistics_size: int = 720,
                 emotion_images_path: str = "images/emoji",
                 overlay_image_path: str = "images/overlay.png"):
        self.y_offset_percent = 0.05
        self.emotion_bar_offset_percent = 0.1

        self.output_statistics_path = out_statistics_path
        self.bar_size, self.padding, emotion_size = self._relative_size_estimation(visualization_size)
        self.report_statistics_shape = report_statistics_size
        self.overlay = cv2.resize(cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED), (visualization_size, visualization_size))
        self._init_bars(emotion_size=emotion_size, emotion_images_path=emotion_images_path)

    def _init_bars(self, emotion_size, emotion_images_path):
        self.bars = {
            "happy": {"bar": EmotionBar(emotion='happy', bar_size=self.bar_size,
                                        emotion_size=emotion_size,
                                        emotion_path=os.path.join(emotion_images_path, 'happy.png')), "count": 0},
            "sad": {"bar": EmotionBar(emotion='sad', bar_size=self.bar_size, emotion_size=emotion_size,
                              emotion_path=os.path.join(emotion_images_path, 'sad.png')), "count": 0},
            "anger": {"bar": EmotionBar(emotion='anger', bar_size=self.bar_size, emotion_size=emotion_size,
                                emotion_path=os.path.join(emotion_images_path, 'anger.png')), "count": 0},
            "surprise": {"bar": EmotionBar(emotion='surprise', bar_size=self.bar_size, emotion_size=emotion_size,
                                   emotion_path=os.path.join(emotion_images_path, 'surprise.png')), "count": 0},
            "neutral": {"bar":EmotionBar(emotion='neutral', bar_size=self.bar_size, emotion_size=emotion_size,
                                  emotion_path=os.path.join(emotion_images_path, 'neutral.png')), "count": 0}
        }
        self._fill_bars()

    def _fill_bars(self):
        for x in self.bars.values():
            x["bar"].draw()

    def _increment_count(self, emotion):
        self.bars[emotion]["count"] += 1

    def _relative_size_estimation(self, visualization_size, bar_amount: int = 5):
        bar_section_size = visualization_size / bar_amount
        bar_size = int(bar_section_size - 2 * bar_section_size * self.emotion_bar_offset_percent)
        padding = int(bar_section_size - bar_size)
        emotion_size = (int(bar_size / 3), int(bar_size / 3))
        return bar_size, padding,  emotion_size

    def draw_bars(self, vis_result):
        y_offset = int(vis_result.shape[1] * self.y_offset_percent)
        alpha_mask_overlay = self.overlay[:, :, 3] / 255.0
        overlay_background = self.overlay[..., :3]
        vis_result = overlay_image(vis_result, overlay_background, 0, 0, alpha_mask_overlay)
        start_x, start_y = np.clip(vis_result.shape[1] - self.bar_size - int(self.padding / 2), 0, vis_result.shape[1]), \
                           np.clip(vis_result.shape[0] - self.bar_size - y_offset, 0,
                                   vis_result.shape[0])
        for emotion, emotion_bar in self.bars.items():
            percent = self.get_current_emotion_percent(emotion)
            text = f'{emotion.upper()} - {percent}%'
            vis_result = add_class_names_and_percents(vis_result, [start_x, int(vis_result.shape[0] - y_offset / 2)],
                                                      text)
            alpha_mask = emotion_bar["bar"].alpha_mask
            vis_result = overlay_image(vis_result, emotion_bar["bar"].bar, start_x, start_y, alpha_mask)
            start_x = np.clip(start_x - self.bar_size - self.padding, 0, vis_result.shape[1])
        return vis_result

    def get_total_update_counts(self):
        return sum([self.bars[x]["count"] for x in self.bars.keys()])

    def update_bars(self, top_emotion):
        self._increment_count(top_emotion)
        total_amount = self.get_total_update_counts()
        # update all others bars
        for emotion in self.bars.keys():
            progress = self.bars[emotion]["count"] / total_amount
            self.bars[emotion]["bar"].draw(progress)

    def save_statistic_to_json(self):
        with open(self.output_statistics_path, 'w') as outfile:
            statistics = {x: y["count"] for x, y in self.bars.items()}
            json.dump(statistics, outfile, indent=1, sort_keys=True)

    def get_current_emotion_percent(self, emotion):
        if self.get_total_update_counts() != 0:
            return int((self.bars[emotion]["count"] / self.get_total_update_counts()) * 100)
        else:
            return 0

    def create_statistics_pie_chart(self, save: bool = False, save_path: str = 'result_statistics_pie_chart.png'):
        statistics = {x: y["count"] for x, y in self.bars.items()}
        emotions = list(statistics.keys())
        emotions_amount = list(statistics.values())
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
        top_emotion, _ = sorted({x: y["count"] for x, y in self.bars.items()}.items(), key=operator.itemgetter(1))[-1]
        return self.bars[top_emotion]["bar"].progress, top_emotion

    def draw(self, image, ret):
        if ret:
            class_name = ret[0].emotions[0].class_name
            self.update_bars(class_name)
        vis_result = draw_emotion_recognition_one_frame(image, ret)
        vis_result = self.draw_bars(vis_result)
        return vis_result

    def create_result_emotion_bar(self, save: bool = False, save_path: str = 'result_statistic_emotion_bar.png'):
        result_statistic = np.zeros((self.report_statistics_shape, self.report_statistics_shape, 3), dtype=np.uint8)
        result_statistic = add_info(result_statistic, [0, int(0.1 * self.report_statistics_shape)], BACKGROUND_COLOR,
                                    'STATISTICS', WHITE_TEXT_COLOR)
        # create top bar
        top_bar_progress, top_emotion = self.get_current_top_bar_parameters()
        top_bar_size = int(self.report_statistics_shape / 4)
        top_bar_emotion_size = (int(top_bar_size / 3), int(top_bar_size / 3))
        top_bar = EmotionBar(bar_size=top_bar_size,
                             emotion_size=top_bar_emotion_size, emotion=top_emotion)
        top_bar.draw(progress=top_bar_progress)
        top_bar_x = int(self.report_statistics_shape / 2 - top_bar_size / 2)
        top_bar_y = int(self.report_statistics_shape / 2 - top_bar_size / 2)
        text = f'MOSTLY {top_bar.emotion.upper()} - {self.get_current_emotion_percent(top_bar.emotion)}%'

        y_offset = int(self.report_statistics_shape * self.y_offset_percent)

        # place text with top emotion and percent
        result_statistic = add_class_names_and_percents(result_statistic,
                                                          [top_bar_x,
                                                           int(top_bar_y - y_offset / 2)],
                                                           text)
        # overlay bar on result statistic image
        result_statistic = overlay_image(result_statistic, top_bar.bar, top_bar_x, top_bar_y, top_bar.alpha_mask)

        # calculate new relative bar parameters
        result_bar_size, result_bar_padding, result_emotion_size = self._relative_size_estimation(
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
            cur_progress = emotion_bar["bar"].progress
            cur_bar = EmotionBar(bar_size=result_bar_size, emotion_size=result_emotion_size)
            cur_bar.draw(progress=cur_progress)

            result_statistic = add_class_names_and_percents(result_statistic,
                                                            [start_x, int(result_statistic.shape[0] - y_offset / 2)],
                                                            text)
            alpha_mask = cur_bar.alpha_mask
            result_statistic = overlay_image(result_statistic, cur_bar.bar, start_x, start_y, alpha_mask)
            start_x = np.clip(start_x - result_bar_size - result_bar_padding, 0, result_statistic.shape[1])
        if save:
            cv2.imwrite(save_path, result_statistic)
        return Image.fromarray(result_statistic[:, :, ::-1])


class EmotionBar:
    def __init__(
        self,
        bar_size: int = 80,
        emotion: str = 'happy',
        progress: float = 0.0,
        emotion_size: tuple = (25, 25),
        emotion_path: str = 'images/emoji/happy.png',
        **kwargs
    ):
        self._progress = progress
        self.bar = np.zeros((bar_size, bar_size, 3), dtype=np.uint8)
        self.alpha_mask = np.ones((bar_size, bar_size, 3), dtype=np.uint8)
        self.emotion_image = np.array(Image.open(emotion_path).resize(emotion_size))
        self.bar_size = bar_size
        self.emotion_size = emotion_size
        self.emotion = emotion
        self._init_params(**kwargs)

    def _init_params(self, **kwargs):
        self.background_color = kwargs.get("background_color", (28, 28, 30))
        self.start_angle = kwargs.get("start_angle", -220)
        self.progress_color = kwargs.get("progress_color", (145, 82, 225))
        self.thickness = kwargs.get("thickness", 10)
        self.end_angle = kwargs.get("end_angle", 40)
        self.radius = self.bar_size // 2
        self.center = (self.bar_size // 2, self.bar_size // 2)
        self.axes = (self.radius, self.radius)

    @property
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, value):
        self._progress = value

    def draw(self, progress: float = 0.0):
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
        self.bar = overlay_image(
            self.bar, overlay_emotion, int(self.center[0] - self.emotion_image.shape[1] / 2),
            int(self.center[1] - self.emotion_image.shape[0] / 2), alpha_mask,
        )

        # should be simplified
        self.alpha_mask = (~(self.bar == np.zeros((self.bar_size, self.bar_size, 3), dtype=np.uint8))[:, :, 0]).astype(
            int)

