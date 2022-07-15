# Function Differences with torch.logical_or

<a href="https://gitee.com/mindspore/docs/blob/r1.8/docs/mindspore/source_en/note/api_mapping/pytorch_diff/LogicalOr.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source_en.png"></a>

## torch.logical_or

```python
class torch.logical_or(input, other, out=None)
```

For more information, see  [torch.logical_or](https://pytorch.org/docs/1.5.0/torch.html#torch.logical_or).

## mindspore.ops.LogicalOr

```python
class class mindspore.ops.LogicalOr()(x, y)
```

For more information, see  [mindspore.ops.LogicalOr](https://mindspore.cn/docs/en/r1.8/api_python/ops/mindspore.ops.LogicalOr.html#mindspore.ops.LogicalOr).

## Differences

PyTorch: Computes the element-wise logical OR of the given input tensors. Zeros are treated as `False` and nonzeros are treated as `True`.

MindSpore: Computes the “logical OR” of two tensors element-wise. The input should be a bool or a tensor whose data type is bool.

## Code Example

```python
import numpy as np
import torch
import mindspore as ms
from mindspore import ops

# MindSpore
x = ms.Tensor(np.array([True, False, True]), ms.bool_)
y = ms.Tensor(np.array([True, True, False]), ms.bool_)
logical_or = ops.LogicalOr()
print(logical_or(x, y))
# [ True  True  True]
x = ms.Tensor(np.array([True, False, True]), ms.int32)
y = ms.Tensor(np.array([True, True, False]), ms.bool_)
logical_or = ops.LogicalOr()
print(logical_or(x, y))
# TypeError: For primitive[LogicalOr], the input argument[x] must be a type of {Tensor[Bool],}, but got Int32.

# PyTorch
print(torch.logical_or(torch.tensor([True, False, True]), torch.tensor([True, False, False])))
# tensor([ True, False,  True])
a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
print(torch.logical_or(a, b))
# tensor([ True,  True,  True, False])
print(torch.logical_or(a.double(), b.double()))
# tensor([ True,  True,  True, False])
print(torch.logical_or(a.double(), b))
# tensor([ True,  True,  True, False])
print(torch.logical_or(a, b, out=torch.empty(4, dtype=torch.bool)))
# tensor([ True,  True,  True, False])
```
