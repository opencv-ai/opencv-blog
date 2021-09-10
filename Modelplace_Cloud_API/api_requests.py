import requests
import json


def login_request(email: str, password:str) -> json:
    response = requests.post(
        url="https://api.modelplace.ai/v3/login", 
        data=json.dumps({ 'email': email, 'password': password })
    )
    if response.status_code != 200:
        raise RuntimeError(response.status_code, response.text)
    return response.json()


def process_request(access_token: str, model_id: int, data_path: str) -> json:
    with open(data_path, "rb") as file:
        file_name = data_path.split('/')[-1]
        response = requests.post(
                url='https://api.modelplace.ai/v3/process', 
                headers={'Authorization': 'Bearer ' + access_token},    
                params=(('model_id', str(model_id)), ),
                files={'upload_data': (file_name, file)},
        )
        if response.status_code != 201:
            raise RuntimeError(response.status_code, response.text)
        return response.json()


def task_request(access_token: str, task_id: int) -> json:
    response = requests.get(
        url='https://api.modelplace.ai/v3/task', 
        headers={'Authorization': 'Bearer ' + access_token}, 
        params=(
            ('task_id', task_id),
            ('visualize', 'true'),
        )
    )
    if response.status_code != 200:
        raise RuntimeError(response.status_code, response.text)
    return response.json()
    