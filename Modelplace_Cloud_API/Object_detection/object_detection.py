import requests
import json
import os


# Set default params
api = 'https://api.modelplace.ai/v3'
current_dir = os.path.abspath(os.path.dirname(__file__))
image_path = os.path.join(current_dir, 'example.png')
prediction_dir = os.path.join(current_dir, 'prediction')
if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)


# Login 
user_info = json.dumps({
    'email': 'irina.maksimova@xperience.ai',  #'YOUR_EMAIL',
    'password': '65kikpop'  #'YOUR_PASSWORD'
})
response = requests.post(api+"/login", data=user_info)
if response.status_code == 200:
    login_data = response.json()
    print('Successful login')
else:
     raise RuntimeError(response.status_code, response.text)


# Set access_token
headers = {'Authorization': 'Bearer ' + login_data['access_token']}


# Prediction run
params = (('model_id', '2'),)
with open(image_path, "rb") as image:
    files = {"upload_data": ("example.png", image)}
    response = requests.post(api + '/process', headers=headers, params=params, files=files)
if response.status_code == 201:
    process_data = response.json()
    print('Prediction run')
else:
   raise RuntimeError(response.status_code, response.text)


# Get prediction
params = ( 
    ('task_id', process_data['task_id']),
    ('visualize', 'true'),
)

from tqdm import tqdm
import time

status = False
with tqdm(desc='Prediction is made') as progress_bar:
    while status is False:
        progress_bar.update()
        response = requests.get(api + '/task', headers=headers, params=params)
        if response.status_code == 200:
            result_data = response.json()
            status = result_data['status'] == result_data['visualization_status'] == 'finished'
            time.sleep(2)
        else:
            raise RuntimeError(response.status_code, response.text)
print('Prediction received')


# Save prediction result
prediction_path = os.path.join(prediction_dir, 'prediction.json')
visualization_path = os.path.join(prediction_dir, 'visualization.gif')
with open(prediction_path, 'w') as f:
    json.dump(result_data['result'], f)
    print(f'Save {prediction_path}')
with open(visualization_path, 'wb') as f:
   f.write(requests.get(result_data['visualization']).content)
   print(f'Save {visualization_path}')


# Plot model results
# Run in the terminal: 
#   pip install modelplace-api@https://github.com/opencv-ai/modelplace-api/archive/v0.4.15.zip
from modelplace_api.visualization import draw_detections_one_frame
from modelplace_api.objects import BBox
import numpy as np
import cv2 

objects_data = result_data['result']
ret = []
for object in objects_data:
    ret.append(
        BBox(
            x1=object['x1'],
            y1=object['y1'],
            x2=object['x2'],
            y2=object['y2'],
            score=object['score'],
            class_name=object['class_name'], 
        )
    )

image = cv2.imread(image_path)
img = draw_detections_one_frame(np.array(image), ret)
my_visualization_path = os.path.join(prediction_dir, 'my_visualization.png')
cv2.imwrite(my_visualization_path, img)
print(f'Save {my_visualization_path}')