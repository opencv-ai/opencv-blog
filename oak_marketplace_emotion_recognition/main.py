import os.path as osp
from args_parser import parse_args
from utils.inference import build_model, process_cam
from utils.helpers import save_video
from utils.visualization import EmotionGraphsCollector
import cv2

def main():
    args = parse_args()
    model_path = osp.join(args.model_path, "checkpoint")
    model = build_model(model_path)
    emotions_collector = EmotionGraphsCollector()
    visualization_results = process_cam(model, emotions_collector, is_show=bool(args.vis))
    if args.save_video:
        save_video(visualization_results)
    is_save = bool(args.save_statistics)
    statistic_result = emotions_collector.create_result_emotion_graphs(720, is_save=is_save)
    emotions_collector.create_statistiс_pie_chart(is_save=is_save)
    cv2.imshow('Result Statistic', statistic_result)
    cv2.waitKey(0)



if __name__ == "__main__":
    main()
