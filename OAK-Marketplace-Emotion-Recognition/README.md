This contains the code for **Emotion Recognition using OpenCV AI KIT and Modelplace**. For more information - visit [OpenCV.AI](https://www.opencv.ai/).

# OAK Emotion Recognition 

![](images/demo.gif)

## Requirements
- Install Python 3.7 or higher and run: ```python3 -m pip install -r requirements.txt```

- Install [FFmpeg](https://ffmpeg.org/download.html)

- Install [Emotion Recognition OAK model](https://github.com/opencv-ai/oak-model-samples/tree/main/emotion_recognition_retail)


## Installation of Emotion Recognition OAK package:


1. Clone [repository](https://github.com/opencv-ai/oak-model-samples) with OAK model samples by the following command:

    ``` git clone https://github.com/opencv-ai/oak-model-samples.git && cd oak-model-samples &&  git checkout 54fa4fe8bd2bb42d6bae49213dbbe6b2f1ce5c5e ```
    

2. Set up Emotion Recognition package by following commands:

   2.1 `cd emotion_recognition_retail`
   
   2.2 `export PATH_TO_EMOTION_RECOGNITION_MODEL=$(pwd)`
   
   2.3 `python3 setup.py bdist_wheel && rm -R build/ *.egg-info`

   2.4 `pip3 install dist/*.whl -f https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/depthai/ && rm -R dist/`

## Usage

To make the code working as described in a blog post, run it with the following command:

`python main.py --model-path $PATH_TO_EMOTION_RECOGNITION_MODEL`


```
usage: main.py [-h] [--model-path MODEL_PATH] [-visualize {0,1}]
               [--save-video {0,1}] [--save-statistics {0,1}]
               [--visualization-size VISUALIZATION_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        Path to Emotion Recognition model
  -visualize {0,1}      If set to 1, visualize the results
  --save-video {0,1}    If set to 1, save the visualization results onto a MP4
                        video file
  --save-statistics {0,1}
                        If set to 1, save the statistics onto a PNG image file
  --visualization-size  Visualization results size. You should specify only
                        one number.
```

