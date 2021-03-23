import os.path as osp

from argsparser import parse_args
from emotion_analyzer import EmotionAnalyzer
from emotion_recognition_retail import InferenceModel
from inference import process_cam
from utils.utils import save_video


def main():
    args = parse_args()
    model_path = osp.join(
        args.model_path, "checkpoint",
    )  # Path to emotion recognition checkpoint

    # build model
    model = InferenceModel(
        model_path=model_path,
    )  # Initialize the emotion recognition model
    model.model_load()  # Load weights

    # build emotion analyzer object, which will aggregate model results and display up-to-date emotion bars
    emotion_analyzer = EmotionAnalyzer(visualization_size=model.get_input_shapes()[0])
    # inference model and get visualization results with emotion bars
    original_images, results, fps = process_cam(
        model, emotion_analyzer, show=args.visualize, return_fps=True,
    )

    # create report with emotion bar statistic
    result_emotion_bar = emotion_analyzer.create_result_emotion_bar(
        save=args.save_statistics,
    )

    # create report emotion pie chart
    result_emotion_pie_chart = emotion_analyzer.create_statistics_pie_chart(
        save=args.save_statistics,
    )

    # show statistic
    result_emotion_bar.show()
    result_emotion_pie_chart.show()

    # save result video
    if args.save_video:
        save_video(original_images, results, fps=fps, size=args.visualization_size)


if __name__ == "__main__":
    main()
