from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model-path",
        help="Path to Emotion Recognition model",
        type=str,
    )
    parser.add_argument(
        "-visualize",
        type=int,
        choices=[0, 1],
        help="If set to 1, visualize the results",
        default=1,
    )
    parser.add_argument(
        "--save-video",
        type=int,
        choices=[0, 1],
        help="If set to 1, save the visualization results onto a MP4 video file",
        default = 1,
    )
    parser.add_argument(
        "--save-statistics",
        type=int,
        choices=[0, 1],
        help="If set to 1, save the statistics onto a PNG image file",
        default=1,
    )
    parser.add_argument(
        "--visualization-size",
        type=int,
        help="Visualization results size. You should specify only one number.",
        default=300,
    )
    return parser.parse_args()
