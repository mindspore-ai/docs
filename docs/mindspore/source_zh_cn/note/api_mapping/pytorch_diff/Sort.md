# 比较与torch.argsort的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Sort.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

## torch.argsort

```python
class torch.argsort(
    input,
    dim=-1,
    descending=False
)
```

更多内容详见 [torch.argsort](https://pytorch.org/docs/1.5.0/torch.html#torch.argsort)。

## mindspore.ops.Sort

```python
class mindspore.ops.Sort(
    axis=-1,
    descending=False
)(x)
```

更多内容详见 [mindspore.ops.Sort](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Sort.html#mindspore.ops.Sort)。

## 使用方式

PyTorch: 返回按值升序沿给定维度对张量进行排序的索引。

MindSpore: 按值升序沿给定维度对输入张量的元素进行排序。 返回一个张量，其值为排序后的值，以及原始输入张量中元素的索引。

## 代码示例

```python
import numpy as np
import torch
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype

# MindSpore
x = Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), mstype.float16)
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
