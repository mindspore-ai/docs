# Comparing the function difference with torch.nn.Parameter.data

## torch.nn.Parameter.data

```python
torch.nn.Parameter.data
```

## mindspore.Parameter.data

```python
mindspore.Parameter.data
```

## Differences

PyTorch: Return Tensor. You can directly use `torch.nn.Parameter.data` to assign values to parameters.

MindSpore：Return the Parameter instance, including its four attributes, `name`, `shape`, `dtype`, and `requirements_grad`. Use the `set_data` interface to assign values to parameters.

## Code Example

```python
from mindspore import Parameter, Tensor
import mindspore
import numpy as np

x = Parameter(Tensor(np.ones([2, 2]), mindspore.float32), name="x", requires_grad=True)
print(x.data)
print(Tensor(x), "\n")

x.set_data(data=Tensor(np.zeros([2, 2]), mindspore.float32))
print(x.data)
print(Tensor(x))

# out
# Parameter (name=x, shape=(2, 2), dtype=Float32, requires_grad=True)
# [[1. 1.]
#  [1. 1.]]
# Parameter (name=x, shape=(2, 2), dtype=Float32, requires_grad=True)
# [[0. 0.]
#  [0. 0.]]
```

```python
from torch import nn
import torch

x = nn.Parameter(torch.ones(2, 2))
print(x.data)

x.data = torch.zeros(2, 2)
print(x.data)

# out
# tensor([[1., 1.],
#         [1., 1.]])
# tensor([[0., 0.],
#         [0., 0.]])
```
