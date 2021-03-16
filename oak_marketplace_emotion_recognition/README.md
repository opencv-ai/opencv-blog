This contains the code for \***\*. For more information - visit [\*\***]()

## Installation:

1. Clone [repository](https://github.com/opencv-ai/oak-model-samples) with oak samples
2. Set up model package:

   2.1 `cd emotion_recognition_retail`

   2.2 `python3 setup.py bdist_wheel && rm -R build/ *.egg-info`

   2.3 `pip3 install dist/*.whl -f https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/depthai/ && rm -R dist/`

## Usage

```
usage: main.py [-h] [-model ROOT_MODEL_PATH] [-vis] [--output-video]
               [--output-statistic]

optional arguments:
  -h, --help            show this help message and exit
  -model ROOT_MODEL_PATH, --root_model_path ROOT_MODEL_PATH
                        Path to root model directory
  -vis, --visualization
                        Visualize the results from the network
  --output-video, -out_vid
                        Save by-frame visualization results of the inference
                        into video
  --output-statistic, -out_stat
                        Save emotion statistic during video watching
```

To see results visualization run the script with the following arguments:

```
python3 main.py -model <path_to_emotion_recognition_retail> -vis
```

You can also use `python3 main.py -model <path_to_emotion_recognition_retail> -vis -out_vid` for create video with visualization results.
It will be stored in `inference_results.mp4`.

`python3 main.py -model <path_to_emotion_recognition_retail> -vis -out_stat` will create `emotion_statistic.png` with emotion pie chart.
