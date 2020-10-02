import cv2
from facemesh import FaceMesh
from blazeface import BlazeFace


# load FaceMesh model
mesh_net = FaceMesh()
mesh_net.load_weights("facemesh.pth")

# load BlazeFace model
blaze_net = BlazeFace()
blaze_net.load_weights("blazeface.pth")
blaze_net.load_anchors("anchors.npy")

# postprocessing for face detector
def get_crop_face(detections, image):
    w, h = image.shape[0], image.shape[1]

    ymin = int(detections[0, 0] * w)
    xmin = int(detections[0, 1] * h)
    ymax = int(detections[0, 2] * w)
    xmax = int(detections[0, 3] * h)

    margin_x = int(0.25 * (xmax - xmin))
    margin_y = int(0.25 * (ymax - ymin))

    ymin -= margin_y
    ymax += margin_y
    xmin -= margin_x
    xmax += margin_x

    face_img = image[ymin: ymax, xmin: xmax]
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    return xmin, ymin, face_img

# postprocessing for mesh
def get_mesh_face(detections, face_img, image, xmin, ymin):
    xscale, yscale = 192 / face_img.shape[1], 192 / face_img.shape[0]
    for i in range(detections.shape[0]):
        x, y = int(detections[i, 0] / xscale), int(detections[i, 1] / yscale)
        image = cv2.circle(image, (xmin + x, ymin + y), 1, (255, 0, 0), 1)


videoCapture = cv2.VideoCapture(0)
while True:
    ret, image = videoCapture.read()
    if not ret:
        break

    # preprocess image
    face_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (128, 128))

    # get face detection boxes
    detections = blaze_net.predict_on_image(face_img).numpy()
    xmin, ymin, face_img = get_crop_face(detections, image)


    # get face mesh
    mesh_img = cv2.resize(face_img, (192, 192))
    detections = mesh_net.predict_on_image(mesh_img).numpy()
    get_mesh_face(detections, face_img, image, xmin, ymin)

    # show processed image
    cv2.imshow("capture", image)

    if cv2.waitKey(3) & 0xFF == 27:
        break
