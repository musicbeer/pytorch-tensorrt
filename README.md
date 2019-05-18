# pytorch model -> onnx -> tensorrt engine
## Convert a Pytorch model to TensorRT engin (optional: int8 engin)

### Required:

Python packages: in `requirements.txt` 

External: 

- CUDA == 9.0
- CUDNN == 7.3.1
- TensorRT == 4.0.2.6

### Steps
- python model2trt.py
- python trt_inference.py

### thanks to
1. https://github.com/modricwang/Pytorch-Model-to-TensorRT
