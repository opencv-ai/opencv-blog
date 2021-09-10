import requests
import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Example of a Python application using the Modelplace Cloud API'
    )
    parser.add_argument(
        "-e",
        "--email",
        type=str,
        help="Set your email from your Modelplace.AI account.",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--password",
        type=str,
        help="Set your password from your Modelplace.AI account.",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model-id",
        type=int,
        help="Model id. \n"
            "To determine the model_id open model in Modelplace.AI "
            "and copy the last number in the address bar: https://modelplace.ai/models/<model_id>",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--data-path",
        type=str, 
        help="Path to the source test file",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--save-folder",
        type=str, 
        help="Folder for saving the results",
        required=True,
    )
    return parser.parse_args()


def save_prediction_result(data: json, save_folder: str, file_name: str) -> None:
    save_path = os.path.join(save_folder, file_name + '.json')
    with open(save_path, 'w') as f:
        json.dump(data, f)


def save_visualization_result(url: str, save_folder: str, file_name: str, file_type: str) -> None:
    save_path = os.path.join(save_folder, file_name + '.' + file_type.split('/')[-1])
    data = requests.get(url).content
    with open(save_path, 'wb') as f:
        f.write(data)
