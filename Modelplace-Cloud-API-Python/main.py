from api_utils import get_access_token, get_results, run_model
from argparser import parse_args
from utils import save_results

if __name__ == "__main__":
    args = parse_args()

    access_token = get_access_token(args.email, args.password)
    task_id = run_model(args.model_id, args.file, access_token)
    results = get_results(task_id, access_token)

    if args.save_path is not None:
        save_results(results, args.save_path)
