# README

Usage:

```bash
cd ascend310_single_op_sample
cmake . -DMINDSPORE_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
make
./tensor_add_sample
```
