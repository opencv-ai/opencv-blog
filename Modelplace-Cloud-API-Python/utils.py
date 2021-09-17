import json
import os
from urllib.parse import urlparse

import requests
from loguru import logger


def save_results(results: json, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)
    save_json(results["result"], os.path.join(save_path, "results.json"), indent=1)
    download_file(results["visualization"], os.path.join(save_path, "visualization"))
    logger.info(f'The results and visualization are saved into "{save_path}" directory')


def save_json(data: dict, save_path: str, indent=0) -> None:
    with open(save_path, "w") as outfile:
        json.dump(data, outfile, indent=indent)
    logger.info(f"{save_path} saved")


def download_file(url: str, save_path: str, extension: str = "") -> None:
    data = requests.get(url).content
    logger.info(f"Downloading {url} content")

    if not extension:
        save_path += os.path.splitext(urlparse(url).path)[1]

    with open(save_path, "wb") as f:
        f.write(data)
    logger.info(f"{save_path} saved")
