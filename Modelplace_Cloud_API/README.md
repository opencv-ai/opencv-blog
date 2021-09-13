# Modelplace cloud API

This page contains a sample Python application using the Modelplace.AI cloud API. 

Our API is a simple HTTP interface with an API key and an image or video as input and JSON with results and visualization as output. Explore our [documentation] and this example to integrate artificial intelligence models into your application or workflow.

[documentation]:https://modelplace.ai/blog/
[Modelplace.AI]:https://modelplace.ai
[Tiny YOLO v4]:https://modelplace.ai/models/32
[model page]:https://modelplace.ai/models

## Installation
### Pre-requirements

Make sure you have installed:

- Python3.7+

Download a sample Python application:
``` bash
git clone https://github.com/opencv-ai/opencv-blog.git
```

## Usage
1. To access the API, register with [Modelplace.AI]
1. Choose a model from the list on the [model page] 
1. Get **Model_ID** from the URI: ```https://modelplace.ai/models/<MODEL_ID>```.  
For example,  for [Tiny YOLO v4], specify 32 as Model ID.
1. Go to the root folder and create a directory for saving the predictions.  
``` bash 
cd opencv-blog/Modelplace_Cloud_API && mkdir prediction
```
4. Running the model and getting a prediction on our test data.   
Read more about these optional arguments [here](#helper).
``` bash 
python3 main.py -e <YOUR_EMAIL> -p <YOUR_PASSWORD> -m <MODEL_ID> -i data/<FILE> -s prediction 
```  
5. Check your root folder.   
``` bash 
ls prediction
```
If the program runs successfully, the **prediction.json** and **visualization file** should appear in prediction folder. 

## Helper
``` 
usage: main.py [-h] -e EMAIL -p PASSWORD -m MODEL_ID -i INPUT_FILE [-s SAVE_FOLDER]

Example of a Python application using the Modelplace.AI Cloud API

optional arguments:
  -h, --help            show this help message and exit
  -e EMAIL, --email EMAIL
                        Your Modelplace.AI account email
  -p PASSWORD, --password PASSWORD
                        Your Modelplace.AI account password
  -m MODEL_ID, --model-id MODEL_ID
                        Model ID - the model you want to run on your data. 
                        Choose a model from the list on the model page - https://modelplace.ai/models 
                        and get Model ID from the URI: https://modelplace.ai/models/<MODEL ID> 
                        e.g. for Tiny YOLO v4 (https://modelplace.ai/models/32), specify 32 as Model ID.
  -i INPUT_FILE, --input-file INPUT_FILE
                        Path to the file you want to run a model on
  -s SAVE_FOLDER, --save-folder SAVE_FOLDER
                        Folder for saving the results
```

## Troubleshooting

If you get any troubles working with Modelplace Cloud API, please, contact us at modelplace@opencv.ai