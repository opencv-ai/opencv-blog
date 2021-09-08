# Object detection

In this block, using the example of the [Faster R-CNN detector], we will describe in detail the steps of model integration into the Python application using [Modelplace Cloud API]. 

To easily follow along with this tutorial, please download the [full code] and a [test image].

## Step 1: Import the required libraries and set default parameters
``` python
import requests
import json
import os

api = 'https://api.modelplace.ai/v3'
current_dir = os.path.abspath(os.path.dirname(__file__))
image_path = os.path.join(current_dir, 'example.png')
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
params = (('model_id', '2'),)
```
To determine the *model_id* open model in [Modelplace.AI] and copy the last number in the address bar: https://modelplace.ai/models/<model_id>


<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1GoSVDnTs4arBNsOaAXhQ-qRshj6lUWwk" height="300"/>
</p>

### 3.2 Load an image, run the model inference and check the return status 
``` python
with open(image_path, "rb") as image:
    files = {"upload_data": ("example.png", image)}
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
See [Modelplace Cloud API] docs for a full description of the model's execution states
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

## Step 5: Save the results
``` python
prediction_path = os.path.join(prediction_dir, 'prediction.json')
visualization_path = os.path.join(prediction_dir, 'visualization.gif')
with open(prediction_path, 'w') as f:
    json.dump(result_data['result'], f)
    print(f'Save {prediction_path}')
with open(visualization_path, 'wb') as f:
   f.write(requests.get(result_data['visualization']).content)
   print(f'Save {visualization_path}')
```
Check your root folder.  
If the program runs successfully, it should have a prediction folder that contains: 
* prediction.json
* visualization.png  

You can see an example of the content of these files on the [Faster R-CNN] by running "test model" using the [test image]. 

## Step 6: Drawing the results on your machine
For quick rendering you can use the out-of-the-box methods of the modelplace-api library. To install it, run the following command in the terminal: 
``` bash
рip install modelplace-api@https://github.com/opencv-ai/modelplace-api/archive/v0.4.15.zip
```
Now you can visualize the results on your machine. To do this:
### 6.1 Import the required libraries
``` python
from modelplace_api.visualization import draw_detections_one_frame
from modelplace_api.objects import BBox
import numpy as np
import cv2 
```
### 6.2 Pre-process detection results
``` python
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
```
### 6.3 Open the source image 
``` python
image = cv2.imread(image_path)
```
### 6.4 Draw the detection boxes and save the result
``` python
img = draw_detections_one_frame(np.array(image), ret)
my_visualization_path = os.path.join(prediction_dir, 'my_visualization.png')
cv2.imwrite(my_visualization_path, img)
print(f'Save {my_visualization_path}') 
```

Check your root folder.  
If the program runs successfully, it should have a prediction folder that contains: 
* prediction.json 
* visualization.png 
* my_visualization.png  


[Faster R-CNN]: https://modelplace.ai/models/2
[Faster R-CNN detector]: https://modelplace.ai/models/2
[Modelplace Cloud API]: https://modelplace.ai/blog/
[full code]: object_detection.py
[test image]: example.png
[Modelplace.AI]:https://modelplace.ai/models
