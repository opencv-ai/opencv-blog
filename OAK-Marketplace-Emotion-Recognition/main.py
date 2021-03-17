import os.path as osp
from emotion_recognition_retail import InferenceModel
from argsparser import parse_args
from utils.inference import process_cam
from utils.helpers import save_video
from visualization import EmotionAnalyzer
import cv2


def main():
    args = parse_args()
    model_path = osp.join(args.model_path, "checkpoint")  # Path to emotion recognition checkpoint
    # build model
    model = InferenceModel(model_path=model_path)  # Initialize the emotion recognition model
    model.model_load()  # Load weights
    emotion_analyzer = EmotionAnalyzer()
    visualization_results = process_cam(model, emotion_analyzer, show=args.visualize)
    if args.save_video:
        save_video(visualization_results)
    result_emotion_bar = emotion_analyzer.create_result_emotion_bar(save=args.save_statistics)
    result_emotion_pie_chart = emotion_analyzer.create_statistic_pie_chart(save=args.save_statistics)
    cv2.imshow('Result Statistic', result_emotion_bar)
    cv2.waitKey(0)
    result_emotion_pie_chart.show()


if __name__ == "__main__":
    main()
