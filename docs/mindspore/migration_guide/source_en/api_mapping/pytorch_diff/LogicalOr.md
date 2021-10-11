# Function Differences with torch.logical_or

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/LogicalOr.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## torch.logical_or

```python
class torch.logical_or(input, other, out=None)
```

For more information, see [torch.logical_or](https://pytorch.org/docs/1.5.0/torch.html#torch.logical_or).

## mindspore.ops.LogicalOr

```python
class class mindspore.ops.LogicalOr(x, y)
```

For more information, see [mindspore.ops.LogicalOr](https://mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.LogicalOr.html#mindspore.ops.LogicalOr).

## Differences

PyTorch: Computes the element-wise logical OR of the given input tensors. Zeros are treated as `False` and nonzeros are treated as `True`.

MindSpore: The input should be a bool or a tensor whose data type is bool.

## Code Example

```python
import numpy as np
import torch
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype

# MindSpore
x = Tensor(np.array([True, False, True]), mstype.bool_)
y = Tensor(np.array([True, True, False]), mstype.bool_)
logical_or = ops.LogicalOr()
logical_or(x, y)
# [ True  True  True]
x = Tensor(np.array([True, False, True]), mstype.bool_)
y = Tensor(np.array([True, True, False]), mstype.int32)
logical_or = ops.LogicalOr()
logical_or(x, y)
# TypeError: For 'LogicalOr', the type of `x` should be subclass of Tensor[Bool], but got Tensor[Int32] .

# PyTorch
torch.logical_or(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
# tensor([ True, False,  True])
a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
torch.logical_or(a, b)
# tensor([ True,  True,  True, False])
torch.logical_or(a.double(), b.double())
# tensor([ True,  True,  True, False])
torch.logical_or(a.double(), b)
# tensor([ True,  True,  True, False])
torch.logical_or(a, b, out=torch.empty(4, dtype=torch.bool))
# tensor([ True,  True,  True, False])
```
