import cv2
import numpy as np
from openvino.inference_engine import IECore
from openvino.inference_engine import IENetwork
from blazeface_numpy import BlazeFace


def get_crop_face(detections, image):
    w, h = image.shape[0], image.shape[1]

    ymin = int(detections[0] * w)
    xmin = int(detections[1] * h)
    ymax = int(detections[2] * w)
    xmax = int(detections[3] * h)

    margin_x = int(0.25 * (xmax - xmin))
    margin_y = int(0.25 * (ymax - ymin))

    ymin -= margin_y
    ymax += margin_y
    xmin -= margin_x
    xmax += margin_x

    face_img = image[ymin:ymax, xmin:xmax]
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    return xmin, ymin, face_img


# postprocessing for mesh
def get_mesh_face(detections, face_image, image, xmin, ymin):
    xscale, yscale = 192 / face_img.shape[1], 192 / face_img.shape[0]
    for i in range(detections.shape[0]):
        x, y = int(detections[i, 0] / xscale), int(detections[i, 1] / yscale)
        image = cv2.circle(image, (xmin + x, ymin + y), 1, (255, 0, 0), 1)


def load_to_IE(model):
    # Loading the Inference Engine API
    ie = IECore()
    # Loading IR files
    net = IENetwork(model=model + ".xml", weights=model + ".bin")
    # Loading the network to the inference engine
    exec_net = ie.load_network(network=net, device_name="CPU")

    return exec_net


def do_inference(exec_net, image):
    input_blob = next(iter(exec_net.inputs))
    return exec_net.infer({input_blob: image})


# load BlazeFace model
blaze_net = load_to_IE("models/blazeface")
# load FaceMesh model
mesh_net = load_to_IE("models/facemesh")

# We need dynamically generated key for fetching output tensor
blaze_outputs = list(blaze_net.outputs.keys())
mesh_outputs = list(mesh_net.outputs.keys())

# to reuse postprocessing from BlazeFace
blazenet = BlazeFace()
blazenet.load_anchors("anchors.npy")


videoCapture = cv2.VideoCapture(0)
while True:
    ret, image = videoCapture.read()
    if not ret:
        break

    # get face detection boxes------------------------------------------------------------------
    # preprocessing
    face_img = cv2.dnn.blobFromImage(image, 1.0 / 127.5, (128, 128), (1, 1, 1), True)
    # inference
    output = do_inference(blaze_net, image=face_img)
    # postprocessing
    boxes = output[blaze_outputs[0]]
    confidences = output[blaze_outputs[1]]
    detections = blazenet._tensors_to_detections(boxes, confidences, blazenet.anchors)
    filtered_detections = []
    for i in range(len(detections)):
        faces = blazenet._weighted_non_max_suppression(detections[i])
        faces = np.stack(faces) if len(faces) > 0 else np.zeros((0, 17))
        filtered_detections.append(faces)
    # for demo we expect only one face
    detections = np.squeeze(filtered_detections, axis=0)
    # take boxes
    xmin, ymin, face_img = get_crop_face(detections, image)

    # get face mesh ----------------------------------------------------------------------------
    # preprocessing
    mesh_img = cv2.dnn.blobFromImage(face_img, 1.0 / 127.5, (192, 192), (1, 1, 1), True)
    # inference
    output = do_inference(mesh_net, image=mesh_img)
    # postprocessing
    detections = output[mesh_outputs[1]].reshape(-1, 3)
    # take mesh
    get_mesh_face(detections, face_img, image, xmin, ymin)

    # show processed image
    cv2.imshow("capture", image)

    if cv2.waitKey(3) & 0xFF == 27:
        break
