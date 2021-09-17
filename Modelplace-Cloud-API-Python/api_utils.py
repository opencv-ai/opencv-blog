from time import sleep

from loguru import logger

from modelplace_cloud_api import login, process, task


def get_access_token(email: str, password: str) -> str:
    return login(email, password).json().get("access_token", "")


def get_task(task_id: str, access_token: str) -> dict:
    return task(task_id, access_token).json()


def run_model(model_id: int, input_file: str, access_token: str) -> str:
    return process(model_id, input_file, access_token).json().get("task_id", "")


def get_results(
    task_id: str, access_token: str, interval: int = 5, times: int = 10
) -> dict:
    for i in range(times):
        logger.info(
            f"Getting the results for Task ID ({task_id}) Step: {i + 1}/{times}"
        )

        results = get_task(task_id, access_token)

        if results.get("status") == "finished":
            logger.info(f"Results are ready. Checking visualization ...")
            if results.get("visualization_status") == "finished":
                return results
            else:
                logger.info(f"Visualization is not ready yet.")
        else:
            logger.info(f"Results and visualization are not ready yet.")

        logger.info(f"Sleeping {interval}s")
        sleep(interval)
