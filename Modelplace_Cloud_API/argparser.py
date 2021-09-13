import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Example of a Python application using the Modelplace.AI Cloud API'
    )
    parser.add_argument(
        "-e",
        "--email",
        type=str,
        help="Your Modelplace.AI account email",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--password",
        type=str,
        help="Your Modelplace.AI account password",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model-id",
        type=int,
        help="Model ID - the model you want to run on your data. "
             "Choose a model from the list on the model page - https://modelplace.ai/models "
             "and get Model ID from the URI: https://modelplace.ai/models/<MODEL ID> "
             "e.g. for Tiny YOLO v4 (https://modelplace.ai/models/32), specify 32 as Model ID.",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=str, 
        help="Path to the file you want to run a model on",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--save-folder",
        type=str,
        default=None,
        help="Folder for saving the results",
    )
    return parser.parse_args()