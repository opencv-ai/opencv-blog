import os.path as osp
from argsparser import parse_args
from utils.inference import build_model, process_cam
from utils.helpers import save_video
from utils.visualization import EmotionBarAnalyzer
import cv2


def main():
    args = parse_args()
    model_path = osp.join(args.model_path, "checkpoint")
    model = build_model(model_path)
    emotion_analyzer = EmotionBarAnalyzer()
    visualization_results = process_cam(model, emotion_analyzer, show=args.vis)
    if args.save_video:
        save_video(visualization_results)
    statistic_result = emotion_analyzer.create_result_emotion_bar(720, save=args.save_statistics)
    emotion_analyzer.create_statistic_pie_chart(save=args.save_statistics)
    cv2.imshow('Result Statistic', statistic_result)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
