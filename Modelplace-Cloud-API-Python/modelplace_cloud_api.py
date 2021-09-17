import json
import os

import requests
from loguru import logger

API = "https://api.modelplace.ai/v3"


def request_logger(function):
    def wrapper(*args, **kwargs):
        response = function(*args, **kwargs)
        if response.ok:
            logger.info(f"Successful {response.url} request")
        else:
            logger.error(
                f"Failed request: {response.url}. "
                f"Status code: {response.status_code}. "
                f"Details: {response.json()['detail']}"
            )
        return response

    return wrapper


@request_logger
def login(email: str, password: str) -> requests.Response:
    return requests.post(
        url=os.path.join(API, "login"),
        data=json.dumps({"email": email, "password": password}),
    )


@request_logger
def process(model_id: int, input_file: str, access_token: str) -> requests.Response:
    with open(input_file, "rb") as file:
        return requests.post(
            url=os.path.join(API, "process"),
            headers={"Authorization": "Bearer " + access_token},
            params=(("model_id", str(model_id)),),
            files={"upload_data": (input_file, file)},
        )


@request_logger
def task(task_id: str, access_token: str) -> requests.Response:
    return requests.get(
        url=os.path.join(API, "task"),
        headers={"Authorization": "Bearer " + access_token},
        params=(
            ("task_id", task_id),
            ("visualize", "true"),
        ),
    )
