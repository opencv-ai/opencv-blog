from loguru import logger
from tqdm import tqdm
import time

from api_requests import login_request, process_request, task_request 
from utils import parse_args, save_prediction_result, save_visualization_result


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Login 
    login_data = login_request(
        args.email, 
        args.password
    )
    logger.info('Successful login')

    # Prediction run
    process_data = process_request(
        login_data['access_token'], 
        args.model_id, 
        args.data_path
    )
    logger.info('Prediction run')

    # Prediction computation
    # Wait until the model's execution state is "finished"
    status = False
    i = 0
    while status is False:
        prediction_data = task_request(
            login_data['access_token'], 
            process_data['task_id']
        )
        status = prediction_data['status'] == prediction_data['visualization_status'] == 'finished'
        logger.info(f"Prediction computation. Iteration #{i}")
        i += 1
        time.sleep(2)
    logger.info('Prediction received')


    # Save prediction and visualization results
    save_prediction_result(
        prediction_data['result'], 
        args.save_folder, 
        file_name='prediction'
    )
    save_visualization_result(
        prediction_data['visualization'], 
        args.save_folder, 
        file_name='visualization', 
        file_type=prediction_data['visualization_type']
    )
    logger.info(f'The prediction is saved in the \"{args.save_folder}\" folder')
    