# Function Differences with torch.logical_and

<a href="https://gitee.com/mindspore/docs/blob/r1.8/docs/mindspore/source_en/note/api_mapping/pytorch_diff/LogicalAnd.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source_en.png"></a>

## torch.logical_and

```python
class torch.logical_and(input, other, out=None)
```

For more information, see  [torch.logical_and](https://pytorch.org/docs/1.5.0/torch.html#torch.logical_and).

## mindspore.ops.LogicalAnd

```python
class class mindspore.ops.LogicalAnd()(x, y)
```

For more information, see  [mindspore.ops.LogicalAnd](https://mindspore.cn/docs/en/r1.8/api_python/ops/mindspore.ops.LogicalAnd.html#mindspore.ops.LogicalAnd).

## Differences

PyTorch: Computes the element-wise logical AND of the given input tensors. Zeros are treated as `False` and nonzeros are treated as `True`.

MindSpore: Computes the “logical AND” of two tensors element-wise. The input should be a bool or a tensor whose data type is bool.

## Code Example

```python
import numpy as np
import torch
import mindspore.ops as ops
import mindspore as ms

# MindSpore
x = ms.Tensor(np.array([True, False, True]), ms.bool_)
y = ms.Tensor(np.array([True, True, False]), ms.bool_)
logical_and = ops.LogicalAnd()
print(logical_and(x, y))
# [ True False False]
x = ms.Tensor(np.array([True, False, True]), ms.int32)
y = ms.Tensor(np.array([True, True, False]), ms.bool_)
logical_and = ops.LogicalAnd()
print(logical_and(x, y))
# TypeError: For primitive[LogicalAnd], the input argument[x] must be a type of {Tensor[Bool],}, but got Int32.

# Pytorch
print(torch.logical_and(torch.tensor([True, False, True]), torch.tensor([True, False, False])))
# tensor([ True, False, False])
a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
print(torch.logical_and(a, b))
# tensor([False, False,  True, False])
print(torch.logical_and(a.double(), b.double()))
# tensor([False, False,  True, False])
print(torch.logical_and(a.double(), b))
# tensor([False, False,  True, False])
print(torch.logical_and(a, b, out=torch.empty(4, dtype=torch.bool)))
# tensor([False, False,  True, False])
```
