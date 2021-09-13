import requests
import json
import os


API = 'https://api.modelplace.ai/v3'


def login(email: str, password:str) -> json:
    response = requests.post(
        url=os.path.join(API, 'login'), 
        data=json.dumps({ 'email': email, 'password': password })
    )
    if response.status_code != 200:
        raise RuntimeError(response.status_code, response.text)
    return response.json()

def process(model_id: int, input_file: str, access_token: str) -> json:
    with open(input_file, 'rb') as file:
        file_name = input_file.split('/')[-1]
        response = requests.post(
                url=os.path.join(API, 'process'), 
                headers={'Authorization': 'Bearer ' + access_token},    
                params=(('model_id', str(model_id)), ),
                files={'upload_data': (file_name, file)},
        )
        if response.status_code != 201:
            raise RuntimeError(response.status_code, response.text)
        return response.json()

def task(task_id: int, access_token: str) -> json:
    response = requests.get(
        url=os.path.join(API, 'task'), 
        headers={'Authorization': 'Bearer ' + access_token}, 
        params=(
            ('task_id', task_id),
            ('visualize', 'true'),
        )
    )
    if response.status_code != 200:
        raise RuntimeError(response.status_code, response.text)
    return response.json()
