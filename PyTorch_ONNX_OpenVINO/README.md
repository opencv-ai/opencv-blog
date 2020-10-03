# How to Speed Up Deep Learning Inference Using OpenVINO Toolkit
The blog post is here: https://opencv.org/how-to-speed-up-deep-learning-inference-using-openvino-toolkit-2/

## Requirements
- Install Python 3.6 or 3.7 and run: ```python3 -m pip install -r requirements.txt```
- Install OpenVINO tookit version 2020.1 or later using the official [instruction](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html)
- [FaceMesh](https://github.com/thepowerfuldeez/facemesh.pytorch)
- [BlazeFace](https://github.com/hollance/BlazeFace-PyTorch)

## Convert the PyTorch model to ONNX format
```python
net = BlazeFace()
net.load_weights("blazeface.pth")
torch.onnx.export(net, torch.randn(1, 3, 128, 128, device='cpu'), "blazeface.onnx",
    input_names=("image", ), output_names=("preds", "confs"), opset_version=9
)

net = FaceMesh()
net.load_weights("facemesh.pth")
torch.onnx.export(net, torch.randn(1, 3, 192, 192, device='cpu'), "facemesh.onnx",
    input_names=("image", ), output_names=("preds", "confs"), opset_version=9
)
```

## Convert the model from ONNX to **Intermediate Representation**
```shell script
source <path_to_openvino>/bin/setupvars.sh
```
```shell script
python3 <path_to_openvino>/deployment_tools/model_optimizer/mo.py --input_model [facemesh or blazeface].onnx
```
