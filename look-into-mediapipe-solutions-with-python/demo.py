import cv2
import mediapipe as mp
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_image",
        "-i",
        required=True,
        help="Path to image for processing",
    )
    parser.add_argument(
        "--model_complexity",
        "-m",
        type=int,
        default=1,
        choices=[1, 2],
        help="Landmark detection model: 1 for middle ckpt, 2 for heavy",
    )
    parser.add_argument(
        "--vis_landmarks",
        "-v",
        help="""Path to output image with visualized results;
            if not specified, results won't be visualized""",
    )
    parser.add_argument(
        "--outputs",
        "-o",
        nargs="*",
        default=["pose_landmarks", "pose_world_landmarks"],
        help="Names of outputs to print",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    for arg in vars(args):
        print(f"{arg} -- {getattr(args, arg)}")

    image = cv2.imread(args.input_image)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp.solutions.pose.Pose(
                model_complexity=args.model_complexity,
                static_image_mode=True
                ) as pose:
        results = pose.process(image_rgb)

    if args.vis_landmarks is not None:
        mp.solutions.drawing_utils.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                    )
        cv2.imwrite(args.vis_landmarks, image)

    for output_name in args.outputs:
        print(f"{output_name}:", getattr(results, output_name))


if __name__ == "__main__":
    main()
