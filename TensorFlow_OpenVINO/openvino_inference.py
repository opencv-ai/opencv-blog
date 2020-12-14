import os
import cv2
import argparse
import time
import numpy as np
from openvino.inference_engine import IECore
from tqdm import tqdm

IMG_EXT = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='OpenVINO inference script')
    parser.add_argument('-i', '--input', type=str, default='',
                        help='Directory to load input images, path to a video or '
                             'skip to get stream from the camera (default).')
    parser.add_argument('-m', '--model', type=str, default='./models/inference_graph.xml',
                        help='Path to IR model')
    return parser.parse_args()


def load_to_IE(model):
    # Getting the *.bin file location
    model_bin = model[:-3] + "bin"
    # Loading the Inference Engine API
    ie = IECore()

    # Loading IR files
    net = ie.read_network(model=model, weights=model_bin)
    input_shape = net.inputs["img_placeholder"].shape

    # Loading the network to the inference engine
    exec_net = ie.load_network(network=net, device_name="CPU")
    print("IR successfully loaded into Inference Engine.")

    return exec_net, input_shape


def sync_inference(exec_net, image):
    input_blob = next(iter(exec_net.inputs))
    return exec_net.infer({input_blob: image})


def main(args):

    if os.path.isdir(args.input):
        # Create a list of test images
        image_filenames = [os.path.join(args.input, f) for f in os.listdir(args.input) if
                           os.path.isfile(os.path.join(args.input, f)) and f.endswith(IMG_EXT)]
        image_filenames.sort()
    else:
        image_filenames = [args.input]

    exec_net, net_input_shape = load_to_IE(args.model)
    # We need dynamically generated key for fetching output tensor
    output_key = list(exec_net.outputs.keys())[0]

    times = []
    for i in range(25):
        for image_num in tqdm(range(len(image_filenames))):
            image = cv2.imread(image_filenames[image_num])
            image = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))

            X = cv2.dnn.blobFromImage(image, swapRB=True)
            start = time.time()
            out = sync_inference(exec_net, image=X)
            times.append(time.time() - start)
            result_image = np.squeeze(np.clip(out[output_key], 0, 255).astype(np.uint8), axis=0).transpose((1, 2, 0))

            cv2.imshow("Out", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyWindow("Out")

    if len(image_filenames) > 1:
        stat = np.asarray(times)

        print(f'Inference time on {len(times)} images:')
        print('Mean: {}'.format(np.mean(stat)))
        print('Min: {}'.format(np.min(stat)))
        print('Max: {}'.format(np.max(stat)))
        times.clear()


if __name__ == "__main__":
    main(parse_args())
