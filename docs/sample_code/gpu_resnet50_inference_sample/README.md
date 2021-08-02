# README

Usage:

```bash
cd gpu_resnet50_inference_sample/
bash build.sh
cd out/
export LD_PRELOAD=/home/miniconda3/lib/libpython37m.so
export LD_LIBRARY_PATH=/usr/local/TensorRT-7.2.2.3/lib/:$LD_LIBRARY_PATH
./main model.mindir 1000 10
```
