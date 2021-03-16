import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import operator
from modelplace_api.visualization import (WHITE_TEXT_COLOR,MONTSERATT_BOLD_TTF_PATH,
                                          NORM_HEIGHT,INFO_TEXT_SIZE,
                                          TEXT_OFFSET_Y, TEXT_OFFSET_X, add_info, BACKGROUND_COLOR)

X_OFFSET = 5
Y_OFFSET = 15


def place_class_names_and_percents(image, coords, text):
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


class EmotionGraphsCollector:
    def __init__(self, statistic_path:str = "result_statistic.json", emotion_size: tuple = (25,25), size: int = 85):
        self.graphs = {
            "happy": EmotionGraph(emotion='happy', size=size, emotion_size=emotion_size),
            "sad": EmotionGraph(emotion='sad',size=size, emotion_size=emotion_size),
            "anger": EmotionGraph(emotion='anger',size=size, emotion_size=emotion_size),
            "surprise": EmotionGraph(emotion='surprise',size=size, emotion_size=emotion_size),
            "neutral": EmotionGraph(emotion='neutral',size=size, emotion_size=emotion_size),
        }
        self.output_statistic_path = statistic_path
        self.graphs_size = size
        self.fill_graphs()

    def fill_graphs(self):
        return [x.update_graph() for x in self.graphs.values()]

    def get_emotions_amount(self):
        return sum([self.graphs[x].emotion_amount for x in self.graphs.keys()])

    def update_graph(self, emotion):
        self.graphs[emotion].update_emotion_amount()
        # respectively update all others graphs
        for emotion in self.graphs.keys():
            progress = self.graphs[emotion].emotion_amount / self.get_emotions_amount()
            self.graphs[emotion].update_graph(progress)

    def dump_statistic_to_json(self):
        with open(self.output_statistic_path, 'w') as outfile:
            statistic = {x: y.emotion_amount for x, y in self.graphs.items()}
            json.dump(statistic, outfile, indent=4, sort_keys=True)

    def get_current_emotion_percent(self, emotion):
        if not self.get_emotions_amount():
            return 0
        return int((self.graphs[emotion].emotion_amount / self.get_emotions_amount()) * 100)

    def plot_statistic_result(self,is_save: bool = False):
        statistic = {x: y.emotion_amount for x, y in self.graphs.items()}
        emotions = list(statistic.keys())
        emotions_amount = list(statistic.values())
        total_emotion_amount = sum(emotions_amount)
        explode = [float(x) / total_emotion_amount for x in emotions_amount]
        plt.pie(
            emotions_amount, labels=emotions,
            startangle=90, autopct='%1.1f%%', shadow=True, explode=explode,
        )
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
        if is_save:
            plt.savefig('emotion_statistic.png')
        plt.show()

    def get_current_top_graph_parameters(self):
        top_emotion, _ = sorted({x: y.emotion_amount for x, y in self.graphs.items()}.items(), key=operator.itemgetter(1))[-1]
        return self.graphs[top_emotion].progress, top_emotion

    def create_result_emotion_graphs(self, size, is_save: bool = False):
        result_statistic = np.zeros((size, size, 3), dtype=np.uint8)
        result_statistic = add_info(result_statistic, [0, int(0.1 * size)], BACKGROUND_COLOR, 'STATISTICS', WHITE_TEXT_COLOR)
        top_graph_progress, top_emotion = self.get_current_top_graph_parameters()
        top_graph_size = int(size / 4)
        top_graph_emotion_size = (int(top_graph_size / 3), int(top_graph_size / 3))
        top_graph = EmotionGraph(size=top_graph_size, emotion=top_emotion, emotion_size=top_graph_emotion_size)
        top_graph.update_graph(top_graph_progress)
        top_graph_x = int(size / 2 - top_graph.width / 2)
        top_graph_y = int(size / 2 - top_graph.height / 2)
        text = f'MOSTLY {top_graph.emotion.upper()} - {self.get_current_emotion_percent(top_graph.emotion)}%'
        result_statistic = place_class_names_and_percents(result_statistic,
                                                          [top_graph_x,
                                                           int(top_graph_y - 2 * Y_OFFSET / 2)],
                                                           text)
        overlay_image_alpha(result_statistic, top_graph.graph, top_graph_x, top_graph_y, top_graph.alpha_mask)
        start_x, start_y = np.clip(result_statistic.shape[0] - 2 * self.graphs_size, 0, result_statistic.shape[0]) , \
                               np.clip(result_statistic.shape[1] - self.graphs_size - Y_OFFSET, 0, result_statistic.shape[0])
        for emotion, ellipse_graph in self.graphs.items():
            if emotion == top_emotion:
                continue
            percent = self.get_current_emotion_percent(emotion)
            text = f'{emotion.upper()} - {percent}%'
            result_statistic = place_class_names_and_percents(result_statistic, [start_x, int(result_statistic.shape[1] - Y_OFFSET / 2)],
                                                        text)
            alpha_mask = ellipse_graph.alpha_mask
            overlay_image_alpha(result_statistic, ellipse_graph.graph, start_x, start_y, alpha_mask)
            start_x = np.clip(start_x - self.graphs_size - 15 * X_OFFSET, 0, result_statistic.shape[0])
        if is_save:
            cv2.imwrite('result_statistic.png', result_statistic)
        return result_statistic


class EmotionGraph:
    def __init__(
        self, background_color: tuple = (28, 28, 30),
        progress_color: tuple = (145, 82, 225),
        size: int = 80,
        thickness: int = 10,
        start_angle: int = -220,
        end_angle: int = 40,
        emotion: str = 'happy',
        emotion_size: tuple = (25, 25)
    ):
        self.background_color = background_color
        self.progress_color = progress_color
        self.size = size
        self.thickness = thickness
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.emotion_size = emotion_size
        self.emotion_images = {
            'neutral': np.array(
                Image.open(
                    os.path.join(
                        os.path.dirname(__file__),
                        "emotions-pack", "neutral.png",
                    ),
                ).resize(self.emotion_size),
            ),
            'happy': np.array(
                Image.open(
                    os.path.join(
                        os.path.dirname(__file__),
                        "emotions-pack", "happy.png",
                    ),
                ).resize(self.emotion_size),
            ),
            'sad': np.array(
                Image.open(
                    os.path.join(
                        os.path.dirname(__file__),
                        "emotions-pack", "sad.png",
                    ),
                ).resize(self.emotion_size),
            ),
            'anger': np.array(
                Image.open(
                    os.path.join(
                        os.path.dirname(__file__),
                        "emotions-pack", "anger.png",
                    ),
                ).resize(self.emotion_size),
            ),
            'surprise': np.array(
                Image.open(
                    os.path.join(
                        os.path.dirname(__file__),
                        "emotions-pack", "surprise.png",
                    ),
                ).resize(self.emotion_size),
            ),
        }
        if emotion not in self.emotion_images.keys():
            raise Exception(f"emotion argument should be one of {self.emotion_images.keys()}!")
        self.emotion = emotion
        self.graph = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        self.height, self.width = self.graph.shape[0:2]
        self.radius = self.size // 2
        self.center = (self.width // 2, self.width // 2)
        self.axes = (self.radius, self.radius)
        self.emotion_amount = 0
        self.alpha_mask = np.ones((self.size, self.size, 3), dtype=np.uint8)

    def update_graph(self, progress: float = 0.0):
        self.progress = progress
        diff = (self.end_angle - self.start_angle) * progress
        self.graph = cv2.ellipse(
            self.graph, center=self.center, axes=self.axes, angle=0, startAngle=self.start_angle,
            endAngle=self.end_angle, color=self.background_color, thickness=-1,
        )
        self.graph = cv2.ellipse(
            self.graph, center=self.center, axes=self.axes, angle=0, startAngle=self.start_angle,
            endAngle=self.start_angle + diff, color=self.progress_color, thickness=-1,
        )
        self.graph = cv2.ellipse(
            self.graph, center=self.center, axes=(self.radius - self.thickness, self.radius - self.thickness),
            angle=0, startAngle=self.start_angle - 1, endAngle=self.end_angle + 1,
            color=(0, 0, 0), thickness=-1,
        )
        # get emotion image
        emotion_image = self.emotion_images[self.emotion]
        alpha_mask = emotion_image[:, :, 3] / 255.0
        overlay_emotion = emotion_image[:, :, :3][:,:,::-1]
        overlay_image_alpha(
            self.graph, overlay_emotion, int(self.center[0] - emotion_image.shape[0] / 2),
            int(self.center[1] - emotion_image.shape[1] / 2), alpha_mask,
        )
        self.alpha_mask = (~(self.graph == np.zeros((self.size, self.size, 3), dtype=np.uint8))[:, :, 0]).astype(int)

        return True

    def update_emotion_amount(self):
        self.emotion_amount += 1