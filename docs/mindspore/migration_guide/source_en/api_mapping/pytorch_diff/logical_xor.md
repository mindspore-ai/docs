# Function Differences with torch.logical_xor

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/logical_xor.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## torch.logical_xor

```python
class torch.logical_xor(input, other, out=None)
```

For more information, see  [torch.logical_xor](https://pytorch.org/docs/1.5.0/torch.html#torch.logical_xor).

## mindspore.numpy.logical_xor

```python
class mindspore.numpy.logical_xor(x1, x2, dtype=None)
```

For more information, see  [mindspore.numpy.logical_xor](https://mindspore.cn/docs/api/en/r1.5/api_python/numpy/mindspore.numpy.logical_xor.html#mindspore.numpy.logical_xor).

## Differences

PyTorch: Computes the element-wise logical XOR of the given input tensors. Zeros are treated as `False` and nonzeros are treated as `True`.

MindSpore: The input should be a bool or a tensor whose data type is bool.

## Code Example

```python
import mindspore.numpy as np
import torch

# MindSpore
x1 = np.array([True, False])
x2 = np.array([False, False])
np.logical_xor(x1, x2)
# [True False]
x1 = np.array([0, 1, 10, 0])
x2 = np.array([4, 0, 1, 0])
np.logical_xor(x1, x2)
# TypeError: For 'LogicalOr', the type of `x` should be subclass of Tensor[Bool], but got Tensor[Int32].

# PyTorch
torch.logical_xor(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
# tensor([False, False,  True])
a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
torch.logical_xor(a, b)
# tensor([ True,  True, False, False])
torch.logical_xor(a.double(), b.double())
# tensor([ True,  True, False, False])
torch.logical_xor(a.double(), b)
# tensor([ True,  True, False, False])
torch.logical_xor(a, b, out=torch.empty(4, dtype=torch.bool))
# tensor([ True,  True, False, False])
```
