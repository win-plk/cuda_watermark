# cuda_watermark
CUDA Image Addition with Watermark

This is the project in ITCS443 Parallel and Distributed Systems

Requirement: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), [OpenCV](https://opencv.org/)

## Could not use CUDA?
CUDA might not work by the following cause
* __Your computer is not compatible with CUDA new version__
  - Just install CUDA older version
* __Your computer did not install NVIDIA Driver__
  - Download and Install [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx)

## How to compile
```
nvcc -o watermark watermark.cu -I "__somewhere__\opencv\build\include" -l "__somewhere__\opencv\build\x64\vc14\lib\opencv_world341" 
```
__Please note that:__ \_\_somewhere\_\_ is the path where you install OpenCV
## How to run
```
watermark.exe waterpaper.jpg watermark.png 
```