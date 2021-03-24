import json
import operator
import os
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from modelplace_api.objects import EmotionLabel
from modelplace_api.visualization import add_info, draw_emotion_recognition_one_frame
from PIL import Image
from utils.visualization import add_class_names_and_percents, overlay_image
from operator import getitem

BACKGROUND_COLOR = (79, 79, 79, 1)
WHITE_TEXT_COLOR = (255, 255, 255, 1)


class EmotionAnalyzer:
    def __init__(
        self,
        visualization_size: int = 300,
        report_statistics_size: int = 720,
        emotion_images_path: str = "images/emoji",
        overlay_image_path: str = "images/overlay.png",
    ) -> None:
        self.y_offset_percent = 0.05
        self.x_offset_percent = 0.05
        self.emotion_bar_offset_percent = 0.1

        self.emotion_images_path = emotion_images_path
        self.report_statistics_size = report_statistics_size

        self.overlay = cv2.resize(
            cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED),
            (visualization_size, visualization_size),
        )

        self.bar_size, self.padding, emotion_size = self._relative_size_estimation(
            visualization_size,
        )

        self._init_bars(emotion_size=emotion_size)

    def _init_bars(self, emotion_size: int) -> None:
        self.bars = {
            "happy": {
                "bar": EmotionBar(
                    emotion="happy",
                    bar_size=self.bar_size,
                    emotion_size=emotion_size,
                    emotion_path=os.path.join(self.emotion_images_path, "happy.png"),
                ),
                "count": 0,
            },
            "sad": {
                "bar": EmotionBar(
                    emotion="sad",
                    bar_size=self.bar_size,
                    emotion_size=emotion_size,
                    emotion_path=os.path.join(self.emotion_images_path, "sad.png"),
                ),
                "count": 0,
            },
            "anger": {
                "bar": EmotionBar(
                    emotion="anger",
                    bar_size=self.bar_size,
                    emotion_size=emotion_size,
                    emotion_path=os.path.join(self.emotion_images_path, "anger.png"),
                ),
                "count": 0,
            },
            "surprise": {
                "bar": EmotionBar(
                    emotion="surprise",
                    bar_size=self.bar_size,
                    emotion_size=emotion_size,
                    emotion_path=os.path.join(self.emotion_images_path, "surprise.png"),
                ),
                "count": 0,
            },
            "neutral": {
                "bar": EmotionBar(
                    emotion="neutral",
                    bar_size=self.bar_size,
                    emotion_size=emotion_size,
                    emotion_path=os.path.join(self.emotion_images_path, "neutral.png"),
                ),
                "count": 0,
            },
        }
        self._fill_bars()

    def _fill_bars(self) -> None:
        for x in self.bars.values():
            x["bar"].update()

    def _increment_count(self, emotion: str) -> None:
        self.bars[emotion]["count"] += 1

    def _relative_size_estimation(
        self, visualization_size: int, bar_amount: int = 5,
    ) -> Tuple[int, int, int]:
        bar_section_size = visualization_size / bar_amount
        bar_size = int(
            bar_section_size - 2 * bar_section_size * self.emotion_bar_offset_percent,
        )
        padding = int(bar_section_size - bar_size)
        emotion_size = int(bar_size / 3)
        return bar_size, padding, emotion_size

    def get_total_update_counts(self) -> int:
        return sum([self.bars[x]["count"] for x in self.bars.keys()])

    def get_top_bar_parameters(self) -> Tuple[float, str]:
        top_emotion, _ = sorted(
            {x: y["count"] for x, y in self.bars.items()}.items(),
            key=operator.itemgetter(1),
        )[-1]
        return self.bars[top_emotion]["bar"].progress, top_emotion

    def get_emotion_percent(self, emotion: str) -> int:
        if self.get_total_update_counts() != 0:
            return int(
                (self.bars[emotion]["count"] / self.get_total_update_counts()) * 100,
            )
        else:
            return 0

    def update_bars(self, ret: List[EmotionLabel]) -> None:
        if ret:
            class_name = ret[0].emotions[0].class_name
            self._increment_count(class_name)
            total_amount = self.get_total_update_counts()
            # update all others bars
            for emotion in self.bars.keys():
                progress = self.bars[emotion]["count"] / total_amount
                self.bars[emotion]["bar"].update(progress)

    def draw_bars(self, vis_result: np.ndarray) -> np.ndarray:
        overlay_mask = self.overlay[:, :, 3] / 255.0
        overlay_colors = self.overlay[..., :3]

        y_offset = int(vis_result.shape[1] * self.y_offset_percent)
        left_top_x, left_top_y = 0, 0

        vis_result = overlay_image(
            vis_result, overlay_colors, left_top_x, left_top_y, overlay_mask,
        )
        start_x, start_y = (
            np.clip(
                vis_result.shape[1] - self.bar_size - self.padding // 2,
                0,
                vis_result.shape[1],
            ),
            np.clip(
                vis_result.shape[0] - self.bar_size - y_offset, 0, vis_result.shape[0],
            ),
        )

        for emotion, emotion_bar in self.bars.items():
            percent = self.get_emotion_percent(emotion)
            text = f"{emotion.upper()} - {percent}%"
            bar_mask = emotion_bar["bar"].mask

            vis_result = add_class_names_and_percents(
                vis_result, [start_x, int(vis_result.shape[0] - y_offset / 2)], text,
            )

            vis_result = overlay_image(
                vis_result, emotion_bar["bar"].bar, start_x, start_y, bar_mask,
            )

            start_x = np.clip(
                start_x - self.bar_size - self.padding, 0, vis_result.shape[1],
            )
        return vis_result

    def draw(self, image: np.ndarray, ret: List[EmotionLabel]) -> np.ndarray:
        vis_result = draw_emotion_recognition_one_frame(image, ret)
        vis_result = self.draw_bars(vis_result)
        return vis_result

    def create_statistics_pie_chart(
        self, save: bool = False, save_path: str = "result_statistics_pie_chart.png",
    ) -> plt:
        statistics = {x: y["count"] for x, y in self.bars.items()}
        emotions = list(statistics.keys())

        emotions_amount = list(statistics.values())
        total_emotion_amount = sum(emotions_amount)

        if total_emotion_amount != 0:
            explode = [float(x) / total_emotion_amount for x in emotions_amount]
        else:
            explode = [0.0] * len(emotions)

        plt.pie(
            emotions_amount,
            labels=emotions,
            startangle=90,
            autopct="%1.1f%%",
            shadow=True,
            explode=explode,
        )
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            shadow=True,
            ncol=5,
        )
        if save:
            plt.savefig(save_path)
            logger.info(f"Saved Emotion Pie Chart as {save_path}")
        return plt

    def create_result_emotion_bar(
        self, save: bool = False, save_path: str = "result_statistics_emotion_bar.png",
    ) -> Image.Image:
        result_statistics = np.zeros(
            (self.report_statistics_size, self.report_statistics_size, 3),
            dtype=np.uint8,
        )
        result_statistics = add_class_names_and_percents(
            result_statistics,
            [
                int(
                    self.report_statistics_size // 2
                    - 1.5 * self.x_offset_percent * self.report_statistics_size,
                ),
                int(1.5 * self.y_offset_percent * self.report_statistics_size),
            ],
            "SUMMARY",
        )
        # create top bar
        top_bar_progress, top_emotion = self.get_top_bar_parameters()
        top_bar_size = int(self.report_statistics_size / 4)
        top_bar_emotion_size = int(top_bar_size / 3)
        top_bar = EmotionBar(
            bar_size=top_bar_size,
            emotion_size=top_bar_emotion_size,
            emotion_path=os.path.join(self.emotion_images_path, f"{top_emotion}.png"),
        )
        top_bar.update(progress=top_bar_progress)
        top_bar_x = int(self.report_statistics_size / 2 - top_bar_size / 2)
        top_bar_y = int(self.report_statistics_size / 2 - top_bar_size / 2)
        text = f"YOU WERE MOSTLY  {top_emotion.upper()} - {self.get_emotion_percent(top_emotion)}%"

        y_offset = int(self.report_statistics_size * self.y_offset_percent)
        # place text with top emotion and percent
        result_statistics = add_info(
            result_statistics,
            [top_bar_x - top_bar_size // 4, int(top_bar_y - y_offset)],
            BACKGROUND_COLOR,
            text,
            WHITE_TEXT_COLOR,
        )

        # overlay top bar on result statistics image
        top_bar_mask = top_bar.mask
        result_statistics = overlay_image(
            result_statistics, top_bar.bar, top_bar_x, top_bar_y, top_bar_mask,
        )

        # calculate new relative bar parameters
        (
            result_bar_size,
            result_bar_padding,
            result_emotion_size,
        ) = self._relative_size_estimation(self.report_statistics_size, bar_amount=4)

        # calculate initial coordinates
        start_x, start_y = (
            np.clip(
                self.report_statistics_size - result_bar_size - result_bar_padding // 2,
                0,
                self.report_statistics_size,
            ),
            np.clip(
                self.report_statistics_size - result_bar_size - y_offset,
                0,
                self.report_statistics_size,
            ),
        )

        for emotion, emotion_bar in sorted(self.bars.items(), key=lambda x: getitem(x[1], 'count')):
            if emotion == top_emotion:
                continue
            percent = self.get_emotion_percent(emotion)
            text = f"{emotion.upper()} - {percent}%"
            cur_progress = emotion_bar["bar"].progress
            cur_bar = EmotionBar(
                bar_size=result_bar_size,
                emotion_size=result_emotion_size,
                emotion_path=os.path.join(self.emotion_images_path, f"{emotion}.png"),
            )
            cur_bar.update(progress=cur_progress)

            # place text with emotion and percent
            result_statistics = add_class_names_and_percents(
                result_statistics,
                [start_x, int(result_statistics.shape[0] - y_offset)],
                text,
            )

            # overlay bar on result statistics image
            bar_mask = cur_bar.mask
            result_statistics = overlay_image(
                result_statistics, cur_bar.bar, start_x, start_y, bar_mask,
            )
            start_x = np.clip(
                start_x - result_bar_size - result_bar_padding,
                0,
                result_statistics.shape[1],
            )
        if save:
            cv2.imwrite(save_path, result_statistics)
            logger.info(f"Saved Emotion Bar Statistics as {save_path}")
        return Image.fromarray(result_statistics[:, :, ::-1])

    def save_statistics_to_json(self, output_statistics_path: str) -> None:
        with open(output_statistics_path, "w") as outfile:
            statistics = {x: y["count"] for x, y in self.bars.items()}
            json.dump(statistics, outfile, indent=1, sort_keys=True)


class EmotionBar:
    def __init__(
        self,
        bar_size: int = 80,
        progress: float = 0.0,
        emotion_size: int = 25,
        emotion_path: str = "images/emoji/happy.png",
        **kwargs,
    ) -> None:
        self._progress = progress

        self.emotion_shape = (emotion_size, emotion_size)
        self.bar_size = bar_size

        self._bar = np.zeros((bar_size, bar_size, 3), dtype=np.uint8)
        self.emotion_image = np.array(
            Image.open(emotion_path).resize(self.emotion_shape),
        )

        self._init_params(**kwargs)

    def _init_params(self, **kwargs) -> None:
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

    @property
    def bar(self):
        return self._bar

    @property
    def mask(self) -> np.ndarray:
        return np.all(self._bar, axis=2).astype(int)

    def update(self, progress: float = 0.0) -> None:
        self._progress = progress
        diff = (self.end_angle - self.start_angle) * progress
        self._bar = cv2.ellipse(
            self._bar,
            center=self.center,
            axes=self.axes,
            angle=0,
            startAngle=self.start_angle,
            endAngle=self.end_angle,
            color=self.background_color,
            thickness=-1,
        )
        self._bar = cv2.ellipse(
            self._bar,
            center=self.center,
            axes=self.axes,
            angle=0,
            startAngle=self.start_angle,
            endAngle=self.start_angle + diff,
            color=self.progress_color,
            thickness=-1,
        )
        self._bar = cv2.ellipse(
            self._bar,
            center=self.center,
            axes=(self.radius - self.thickness, self.radius - self.thickness),
            angle=0,
            startAngle=self.start_angle - 1,
            endAngle=self.end_angle + 1,
            color=(0, 0, 0),
            thickness=-1,
        )
        # place emotion
        emotion_mask = self.emotion_image[:, :, 3] / 255.0
        emotion_color = self.emotion_image[:, :, :3][:, :, ::-1]
        self._bar = overlay_image(
            self._bar,
            emotion_color,
            int(self.center[0] - self.emotion_image.shape[1] / 2),
            int(self.center[1] - self.emotion_image.shape[0] / 2),
            emotion_mask,
        )
