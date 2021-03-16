from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-model",
        "--root_model_path",
        help="Path to root model directory",
        type=str,
    )
    parser.add_argument(
        "-vis",
        "--visualization",
        action="store_true",
        help="Visualize the results from the network",
    )
    parser.add_argument(
        "--output-video",
        "-out_vid",
        help="Save by-frame visualization results of the inference into video",
        action="store_true",
    )
    parser.add_argument(
        "--output-statistic",
        "-out_stat",
        help="Save emotion statistic during video watching",
        action="store_true",
    )
    return parser.parse_args()
