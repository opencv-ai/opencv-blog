from loguru import logger
import requests
import json
import os


def log_error_status(response: requests.models.Response) -> None:
    logger.error(f'Failed request: {response.url}. '
                f'Status code: {response.status_code}. '
                f'Detail: {response.json()["detail"]}')

def save_results(results: json, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)
    save_prediction_results(results['result'], save_path, file_name='prediction')
    save_visualization_results(results['visualization'], save_path, file_name='visualization')
    logger.info(f'The prediction is saved in the \"{save_path}\" folder')

def save_prediction_results(data: json, save_folder: str, file_name: str) -> None:
    save_path = os.path.join(save_folder, file_name + '.json')
    with open(save_path, 'w') as f:
        json.dump(data, f)

def save_visualization_results(url: str, save_path: str, file_name: str) -> None:
    file_type = url.split("?")[0].split('.')[-1]
    save_path = os.path.join(save_path, file_name + '.' + file_type.split('/')[-1])
    data = requests.get(url).content
    with open(save_path, 'wb') as f:
        f.write(data)
