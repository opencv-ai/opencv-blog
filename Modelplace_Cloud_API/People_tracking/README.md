# People tracking

In this block on the example of [CenterNet and OSNet based tracker], we will describe in detail the steps of model integration into the Python application using [Modelplace Cloud API].

To easily follow along with this tutorial, please download the [full code] and a [test video] and put these files in the same folder.  
To run the script, run the command: 
``` bash
python people_tracking.py 
```


## Step 1: Import the required libraries and set default parameters
``` python
import requests
import json
import os


api = 'https://api.modelplace.ai/v3'
current_dir = os.path.abspath(os.path.dirname(__file__))
video_path = os.path.join(current_dir, 'example.webm')
prediction_dir = os.path.join(current_dir, 'prediction')
if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)
```

## Step 2: Get an access token  using **/login** method
### 2.1 Set your email and password from your [Modelplace.AI] account
``` python 
user_info = json.dumps({
   'email': 'YOUR_EMAIL',
   'password': 'YOUR_PASSWORD'
})
```

### 2.2 Make a request and check the return status
``` python
response = requests.post(api+"/login", data=user_info)
if response.status_code == 200:
    login_data = response.json()
    print('Successful login')
else:
    raise RuntimeError(response.status_code, response.text)
```

If successful, **/login** returns an *access_token*, which is necessary for further work with the protected API methods 

### 2.3 Set your *access_token* for accessing protected API methods
``` python
headers = {'Authorization': 'Bearer ' + login_data['access_token']}
```

## Step 3: Run the model inference using **/process** method
### 3.1 Set the model_id  
``` python
params = (('model_id', '10'),)
```
To determine the *model_id* open model in [Modelplace.AI] and copy the last number in the address bar: https://modelplace.ai/models/<model_id>

<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1GoSVDnTs4arBNsOaAXhQ-qRshj6lUWwk" height="300"/>
</p>

### 3.2 Load an image, run the model inference and check the return status 
``` python
with open(video_path, "rb") as video:
   files = {"upload_data": ("example.webm", video)}
   response = requests.post(api + '/process', headers=headers, params=params, files=files)
if response.status_code == 201:
   process_data = response.json()
   print('Prediction run')
else:
   raise RuntimeError(response.status_code, response.text)
```
If successful, **/process** returns *task_id*, which identifies the task created in the cloud.

## Step 4: Get the result of the model using /task method
### 4.1 Set the task_id in the query parameters. 
Set *true* value for 'visualize' to get a built-in  visualization of how the model performs.

``` python
params = ( 
    ('task_id', process_data['task_id']),
    ('visualize', 'true'),
)
```

### 4.2 Wait until the model's execution state is "finished"
``` python
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
```
See [Modelplace Cloud API] docs for a full description of the model's execution states

## Step 5: Save the results
``` python
prediction_path = os.path.join(prediction_dir, 'prediction.json')
visualization_path = os.path.join(prediction_dir, 'visualization.mp4')
with open(prediction_path, 'w') as f:
    json.dump(result_data['result'], f)
    print(f'Save {prediction_path}')
with open(visualization_path, 'wb') as f:
   f.write(requests.get(result_data['visualization']).content)
   print(f'Save {visualization_path}')
```
Check your root folder.  
If the program runs successfully, it should have a *prediction* folder that contains: 
* prediction.json
* visualization.mp4  

You can see an example of the content of these files on the [Faster R-CNN] by running "test model" using the [test image]. 

<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1oRsv1BII2SRgN6Rl8pq7sqdwAqb_FIz8" height="300"/>
</p>

## Step 6: Drawing the results on your machine
For quick rendering you can use the out-of-the-box methods of the [modelplace-api] and skvideo libraries.  
To install them, run the following commands in the terminal: 
``` bash
рip install modelplace-api@https://github.com/opencv-ai/modelplace-api/archive/v0.4.15.zip
pip install sk-video
```
Now you can visualize the results on your machine. To do this:
### 6.1 Import the required libraries
``` python
from modelplace_api.visualization import draw_tracks
from modelplace_api.objects import VideoFrame, TrackBBox
from skvideo.io import vread
```
### 6.2 Pre-process detection results
``` python
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
```
### 6.3 Open the source video 
``` python
video = vread(video_path)
```
### 6.4 Draw the tracking boxes and save the result
``` python
my_visualization_path = os.path.join(prediction_dir, 'my_visualization.mp4')
video = draw_tracks(video, ret, save_path=my_visualization_path, fps=30)
print(f'Save {my_visualization_path}')
```

Check your root folder.  
If the program runs successfully, it should have a *prediction* folder that contains: 
* prediction.json 
* visualization.mp4
* my_visualization.mp4   


## Troubleshooting
If you get any troubles working with Modelplace Cloud API, please, contact us at modelplace@opencv.ai


[Modelplace.AI]:https://modelplace.ai/models
[modelplace-api]:https://github.com/opencv-ai/modelplace-api
[CenterNet and OSNet based tracker]:https://modelplace.ai/models/10
[Modelplace Cloud API]: https://modelplace.ai/blog/
[full code]: people_tracking.py
[test video]: example.webm