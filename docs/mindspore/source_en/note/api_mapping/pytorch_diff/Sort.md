# Function Differences with torch.argsort

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Sort.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

## torch.argsort

```python
class torch.argsort(
    input,
    dim=-1,
    descending=False
)
```

For more information, see  [torch.argsort](https://pytorch.org/docs/1.5.0/torch.html#torch.argsort).

## mindspore.ops.Sort

```python
class mindspore.ops.Sort(
    axis=-1,
    descending=False
)(x)
```

For more information, see  [mindspore.ops.Sort](https://mindspore.cn/docs/en/r2.0.0-alpha/api_python/ops/mindspore.ops.Sort.html#mindspore.ops.Sort).

## Differences

PyTorch: Returns the indices that sort a tensor along a given dimension in ascending order by value.

MindSpore: Sorts the elements of the input tensor along a given dimension in ascending order by value. Returns a tensor whose values are the **sorted** values, and the indices of the elements in the original input tensor.

## Code Example

```python
import numpy as np
import torch
from mindspore import ops
import mindspore as ms

# MindSpore
x = ms.Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), ms.float16)
sort = ops.Sort()
output = sort(x)
print(output)
# Out:
# (Tensor(shape=[3, 3], dtype=Float16, value=
# [[ 1.0000e+00,  2.0000e+00,  8.0000e+00],
#  [ 3.0000e+00,  5.0000e+00,  9.0000e+00],
#  [ 4.0000e+00,  6.0000e+00,  7.0000e+00]]), Tensor(shape=[3, 3], dtype=Int32, value=
# [[2, 1, 0],
#  [2, 0, 1],
#  [0, 1, 2]]))

# Pytorch
a = torch.tensor([[8, 2, 1], [5, 9, 3], [4, 6, 7]], dtype=torch.int8)
torch.argsort(a, dim=1)
# Out:
# tensor([[2, 1, 0],
#         [2, 0, 1],
#         [0, 1, 2]])
```
