# Function Differences with torch.randperm

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/Randperm.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see  [torch.randperm](https://pytorch.org/docs/1.5.0/torch.html#torch.randperm).

## mindspore.ops.Randperm

```python
class mindspore.ops.Randperm(
    max_length=1,
    pad=-1,
    dtype=mstype.int32
)(n)
```

For more information, see  [mindspore.ops.Randperm](https://mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.Randperm.html#mindspore.ops.Randperm).

## Differences

PyTorch: Returns a random permutation of integers from 0 to n - 1.

MindSpore: Generates n random samples from 0 to n-1 without repeating. If the max_length greater than n, the last max_length-n element will be filled with pad.

## Code Example

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
