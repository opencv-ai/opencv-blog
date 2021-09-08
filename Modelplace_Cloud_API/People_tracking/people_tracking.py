import requests
import json
import os


# Set default params
api = 'https://api.modelplace.ai/v3'
current_dir = os.path.abspath(os.path.dirname(__file__))
video_path = os.path.join(current_dir, 'example.webm')
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
params = (('model_id', '10'),)
with open(video_path, "rb") as video:
   files = {"upload_data": ("example.webm", video)}
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
            time.sleep(5)
        else:
            raise RuntimeError(response.status_code, response.text)
print('Prediction received')


# Save prediction result
prediction_path = os.path.join(prediction_dir, 'prediction.json')
visualization_path = os.path.join(prediction_dir, 'visualization.mp4')
with open(prediction_path, 'w') as f:
    json.dump(result_data['result'], f)
    print(f'Save {prediction_path}')
with open(visualization_path, 'wb') as f:
   f.write(requests.get(result_data['visualization']).content)
   print(f'Save {visualization_path}')


# Plot model results
## Run in the terminal: 
#       pip install modelplace-api@https://github.com/opencv-ai/modelplace-api/archive/v0.4.15.zip
#       pip install sk-video
from modelplace_api.visualization import draw_tracks
from modelplace_api.objects import VideoFrame, TrackBBox
from skvideo.io import vread

video_data = result_data['result']
ret = []
for frame in video_data:
    ret.append(
        VideoFrame(
            number=frame['number'],
            boxes=[
                TrackBBox(
                    x1=box['x1'],
                    y1=box['y1'],
                    x2=box['x2'],
                    y2=box['y2'],
                    score=box['score'],
                    class_name=box['class_name'],
                    track_number=box['track_number']
                )
                for box in frame['boxes']
            ]
        )
    )

video = vread(video_path) 
my_visualization_path = os.path.join(prediction_dir, 'my_visualization.mp4')
video = draw_tracks(video, ret, save_path=my_visualization_path, fps=30)
print(f'Save {my_visualization_path}')
