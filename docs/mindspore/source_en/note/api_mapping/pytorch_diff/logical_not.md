# Function Differences with torch.logical_not

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.8/docs/mindspore/source_en/note/api_mapping/pytorch_diff/logical_not.md)

## torch.logical_not

```python
class torch.logical_not(input, out=None)
```

For more information, see  [torch.logical_not](https://pytorch.org/docs/1.5.0/torch.html#torch.logical_not).

## mindspore.numpy.logical_not

```python
class mindspore.numpy.logical_not(a, dtype=None)
```

For more information, see  [mindspore.numpy.logical_not](https://mindspore.cn/docs/en/r1.8/api_python/numpy/mindspore.numpy.logical_not.html#mindspore.numpy.logical_not).

## Differences

PyTorch: If not specified, the output tensor will have the bool dtype. If the input tensor is not a bool tensor, zeros are treated as `False` and non-zeros are treated as `True`.

MindSpore: Calculate the logical negation of the input tensor element-wise. The input should be a tensor whose dtype is bool.

## Code Example

```python
import mindspore.numpy as np
import torch

# MindSpore
print(np.logical_not(np.array([True, False])))
# Tensor(shape=[2], dtype=Bool, value= [False,  True])
print(np.logical_not(np.array([0, 1, -10])))
# TypeError: For primitive[LogicalNot], the input argument[x] must be a type of {Tensor[Bool],}, but got Int32.

# PyTorch
print(torch.logical_not(torch.tensor([True, False])))
# tensor([False,  True])
print(torch.logical_not(torch.tensor([0, 1, -10], dtype=torch.int8)))
# tensor([ True, False, False])
print(torch.logical_not(torch.tensor([0., 1.5, -10.], dtype=torch.double)))
# tensor([ True, False, False])
print(torch.logical_not(torch.tensor([0., 1., -10.], dtype=torch.double), out=torch.empty(3, dtype=torch.int16)))
# tensor([1, 0, 0], dtype=torch.int16)
```
