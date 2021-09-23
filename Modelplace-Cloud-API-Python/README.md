# Modelplace Cloud API

This page contains a Python Application that uses Modelplace.AI Cloud API.  

Our API is a simple HTTP interface with an API key. It uses images or videos as an input, and outputs JSON with results and visualization. Explore our [documentation] and this Python example to integrate AI models into your application or workflow.   

![](data/demo.gif)

## Requirements

1. Install Python 3.7+
2. Clone this repository
    ```bash
    git clone git@github.com:opencv-ai/opencv-blog.git
    ```
3. Change directory
    ```bash
    cd ./opencv-blog/Modelplace-Cloud-API-Python
    ```
4. Install requirements
   ```bash
   python3 -m pip install -r requirements.txt
   ```

## Usage

### Quick Start

To quickly start using Modelplace.AI Cloud API run the following command in the terminal:
```bash
python3 main.py -e <EMAIL> -p <PASSWORD> -i <MODEL_ID> -f <FILE>
```
where:
- `EMAIL` - Your [Modelplace.AI] account email
- `PASSWORD` - Your [Modelplace.AI] account password
- `MODEL_ID` - Model ID - the model you want to run on your data. Choose a model from the list on the [model page] and get Model ID from the URI: `https://modelplace.ai/models/<MODEL ID>`  
E.g. for [Tiny YOLOv4], specify 32 as Model ID
- `FILE` - Path to the file you want to run a model on
``` 
usage: main.py [-h] -e EMAIL -p PASSWORD -id MODEL_ID -f FILE [-s SAVE_PATH]
​
Modelplace.AI Cloud API Python Application
​
optional arguments:
  -h, --help            show this help message and exit
  -e EMAIL, --email EMAIL
                        Your Modelplace.AI account email
  -p PASSWORD, --password PASSWORD
                        Your Modelplace.AI account password
  -id MODEL_ID, --model-id MODEL_ID
                        Model ID - the model you want to run on your data. Choose a model from the list on the model page - https://modelplace.ai/models and get Model ID from the URI: https://modelplace.ai/models/<MODEL ID>
                        e.g. for Tiny YOLO v4 (https://modelplace.ai/models/32), specify 32 as Model ID.
  -f FILE, --file FILE  Path to the file you want to run a model on
  -s SAVE_PATH, --save-path SAVE_PATH
                        The directory where the results will be saved
```

## Troubleshooting

If you get any troubles working with Modelplace Cloud API, please, contact us at modelplace@opencv.ai

[documentation]:https://modelplace.ai/blog/cloud-api
[Modelplace.AI]:https://modelplace.ai
[model page]:https://modelplace.ai/models
[Tiny YOLOv4]:https://modelplace.ai/models/32
