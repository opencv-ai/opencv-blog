from loguru import logger
import time
import json
import os

from argparser import parse_args
from api_requests import login, process, task
from utils import save_prediction_results, save_visualization_results


def get_access_token(email:str, password:str) -> str:
    login_data = login(email, password)
    access_token = login_data['access_token']
    logger.info('Successful login')
    return access_token

def get_task_id(model_id: int, input_file: str, access_token: str) -> json:
    process_data = process(model_id, input_file, access_token)
    task_id = process_data['task_id']
    logger.info('Prediction run')
    return task_id

def run_model(model_id: int, input_file: str, access_token: str) -> json:
    task_id = get_task_id(model_id, input_file, access_token)
    status = False
    i = 0
    while status is False:
        prediction_data = task(task_id, access_token)
        status = prediction_data['status'] == prediction_data['visualization_status'] == 'finished'
        logger.info(f"Prediction computation. Iteration #{i}")
        i += 1
        time.sleep(2)
    logger.info('Prediction received')
    return prediction_data

def save_results(results: json, save_folder: str) -> None:
    os.makedirs(save_folder, exist_ok=True)
    save_prediction_results(results['result'], save_folder, file_name='prediction')
    save_visualization_results(results['visualization'], save_folder, file_name='visualization')
    logger.info(f'The prediction is saved in the \"{save_folder}\" folder')


if __name__ == "__main__":
    args = parse_args()
    
    access_token = get_access_token(args.email, args.password)
    results = run_model(args.model_id, args.input_file, access_token)
    
    if args.save_folder is not None:
        save_results(results, args.save_folder)
