# README

Usage:

```bash
cd ascend910_resnet50_preprocess_sample
cmake . -DMINDSPORE_PATH=`pip3 show mindspore | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
make
./resnet50_sample
```
