from loguru import logger
import requests
import json
import time
import os


API = 'https://api.modelplace.ai/v3'

def log_error_status(response: requests.models.Response) -> None:
    logger.error(f'Failed request: {response.url}. '
                f'Status code: {response.status_code}. '
                f'Detail: {response.json()["detail"]}')

def get_access_token(email: str, password:str) -> str:
    response = requests.post(
        url=os.path.join(API, 'login'), 
        data=json.dumps({ 'email': email, 'password': password })
    )
    if response.ok:
        logger.info('Successful login')
    else:
        log_error_status(response)
        raise RuntimeError('Failed request')
    return response.json()['access_token']

def run_model(model_id: int, input_file: str, access_token: str) -> str:
    with open(input_file, 'rb') as file:
        file_name = input_file.split('/')[-1]
        response = requests.post(
                url=os.path.join(API, 'process'), 
                headers={'Authorization': 'Bearer ' + access_token},    
                params=(('model_id', str(model_id)), ),
                files={'upload_data': (file_name, file)},
        )
        if response.ok:
            logger.info('Prediction run')
        else:
            log_error_status(response)
            raise RuntimeError('Failed request')
        return response.json()['task_id']

def get_results(task_id: int, access_token: str, interval: int = 1, times: int = 60) -> json:
    num_iteration = int(times / interval)
    for iter in range(num_iteration):
        prediction_data = get_task(task_id, access_token)
        logger.info('Prediction computation. Iterating #{}', iter)
        if prediction_data['status'] == prediction_data['visualization_status'] == 'finished':
            break
        time.sleep(interval)
    return prediction_data

def get_task(task_id: str, access_token: str) -> json:
    response = requests.get(
        url=os.path.join(API, 'task'), 
        headers={'Authorization': 'Bearer ' + access_token}, 
        params=(
            ('task_id', task_id),
            ('visualize', 'true'),
        )
    )
    if not response.ok:
        log_error_status(response)
        raise RuntimeError('Failed request')
    return response.json()