# 比较与torch.randperm的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/Randperm.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.randperm

```python
class torch.randperm(
    n,
    out=None,
    dtype=torch.int64,
    layout=torch.strided,
    device=None,
    requires_grad=False
)
```

更多内容详见 [torch.randperm](https://pytorch.org/docs/1.5.0/torch.html#torch.randperm)。

## mindspore.ops.Randperm

```python
class mindspore.ops.Randperm(
    max_length=1,
    pad=-1,
    dtype=mstype.int32
)(n)
```

更多内容详见 [mindspore.ops.Randperm](https://mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Randperm.html#mindspore.ops.Randperm)。

## 使用方式

PyTorch: 返回从0到n-1的整数的随机排列。

MindSpore: 生成从0到n-1的n个随机样本，不重复。如果max_length大于n，最后的max_length-n个元素将用参数值pad填充。

## 代码示例

```python
import torch
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype

# MindSpore
# The result of every execution is different because this operator will generate n random samples.
randperm = ops.Randperm(max_length=30, pad=-1)
n = Tensor([20], dtype=mstype.int32)
output = randperm(n)
print(output)
# Out:
# [15 6 11 19 14 16 9 5 13 18 4 10 8 0 17 2 1 12 3 7
#  -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]

# PyTorch
torch.randperm(30)
# Out:
# tensor([ 1, 25, 20,  0, 26, 16, 21, 27, 12,  7,  8, 15, 14, 23,  4,  3, 17, 11,
#          9, 13,  5,  6,  2, 28, 19, 22, 24, 10, 29, 18])
```
