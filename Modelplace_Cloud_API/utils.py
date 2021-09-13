import requests
import json
import os


def save_prediction_results(data: json, save_folder: str, file_name: str) -> None:
    save_path = os.path.join(save_folder, file_name + '.json')
    with open(save_path, 'w') as f:
        json.dump(data, f)


def save_visualization_results(url: str, save_folder: str, file_name: str) -> None:
    file_type = url.split("?")[0].split('.')[-1]
    save_path = os.path.join(save_folder, file_name + '.' + file_type.split('/')[-1])
    data = requests.get(url).content
    with open(save_path, 'wb') as f:
        f.write(data)
        