# Function Differences with torch.nn.MaxPool1d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/MaxPool1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.MaxPool1d

```text
class torch.nn.MaxPool1d(
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    return_indices=False,
    ceil_mode=False
)(input) -> Tensor
```

For more information, see [torch.nn.MaxPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxPool1d.html).

## mindspore.nn.MaxPool1d

```text
class mindspore.nn.MaxPool1d(
    kernel_size=1,
    stride=1,
    pad_mode='valid'
)(x) -> Tensor
```

For more information, see [mindspore.nn.MaxPool1d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.MaxPool1d.html).

## Differences

PyTorch：Perform maximum pooling operations on temporal data.

MindSpore：The implementation function of API in MindSpore is basically the same as that of PyTorch, but it lacks padding, dilation, and return_indices for parameter setting.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | kernel_size | kernel_size | Consistent function, no default values for PyTorch |
| | Parameter 2 | stride | stride | Consistent function, different default value  |
| | Parameter 3 | padding | - | The number of elements to fill. The default value is 0 (no padding), and the value cannot exceed kernel_size/2 (rounded down) |
| | Parameter 4 | dilation | - | Span length between elements in the window: the default value is 1, when the elements in the window are contiguous. If the value > 1, the elements in the window are spaced |
| | Parameter 5 | return_indices | - | Return index: If the value is True, the index of the corresponding element will be returned along with the maximum pooling result. Useful for subsequent calls to torch.nn.MaxUnpool1d |
| | Parameter 6 | ceil_mode | - | Control the output shape(N, C, L_{out}) in L_{out} to round up or down. In MindSpore, default: round down |
| | Parameter 7 | input | x | Consistent function, different parameter names |
| | Parameter 8 | - | pad_mode | Control the filling mode, and PyTorch does not have this parameter |

### Code Example 1

> Construct a pooling layer with a convolution kernel size of 1x3 and a step size of 1. The padding defaults to 0 and no element filling is performed. The default value of dilation is 1, and the elements in the window are contiguous. The default value of pooling padding mode is valid, which returns the output from a valid calculation without padding, and any extra pixels that do not satisfy the calculation are discarded. With the same parameter settings, the two APIs achieve the same function and perform the maximum pooling operation on the data.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

max_pool = torch.nn.MaxPool1d(kernel_size=3, stride=1)
x = tensor(np.random.randint(0, 10, [1, 2, 4]), dtype=torch.float32)
output = max_pool(x)
result = output.shape
print(tuple(result))
# (1, 2, 2)

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

max_pool = mindspore.nn.MaxPool1d(kernel_size=3, stride=1)
x = Tensor(np.random.randint(0, 10, [1, 2, 4]), mindspore.float32)
output = max_pool(x)
result = output.shape
print(result)
# (1, 2, 2)
```

### Code Example 2

> When ceil_mode=True and pad_mode='same', both APIs achieve the same function.

```python
# PyTorch
import torch
from torch import tensor

max_pool = torch.nn.MaxPool1d(kernel_size=3, stride=2, ceil_mode=True)
x = torch.Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]])
output = max_pool(x)
print(output.numpy())
# [[[ 3.  5.  7.  9. 10.]
#   [ 3.  5.  7.  9. 10.]]]

# MindSpore
import mindspore
from mindspore import Tensor

max_pool = mindspore.nn.MaxPool1d(kernel_size=3, stride=2, pad_mode='same')
x = Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], mindspore.float32)
output = max_pool(x)
print(output)
# [[[ 3.  5.  7.  9. 10.]
#   [ 3.  5.  7.  9. 10.]]]
```

### Code Example 3

> When padding=1 is set in PyTorch, the negative infinite tensor with shape (1, 2, 1) is constructed by replacing 10 in the shape (1, 2, 10) of the input tensor x with the value of padding in PyTorch in MindSpore. Both sides of the original input tensor x are spliced at axis=2 by the concat function, and the new x tensor is used to calculate the result of the maximum pooling so that the two APIs achieve the same function.

```python
# PyTorch
import torch

max_pool = torch.nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
x = torch.Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]])
output = max_pool(x)
result = output.shape
print(output.numpy())
# [[[ 3.  5.  7.  9. 10.]
#   [ 3.  5.  7.  9. 10.]]]
print(tuple(result))
# (1, 2, 5)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

max_pool = mindspore.nn.MaxPool1d(kernel_size=4, stride=2)
x = Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], mindspore.float32)
pad = ops.Pad(((0, 0), (0, 0), (1, 1)))
data = pad(Tensor(x))
output = max_pool(data)
result = output.shape
print(output)
# [[[ 3.  5.  7.  9. 10.]
#   [ 3.  5.  7.  9. 10.]]]
print(result)
# (1, 2, 5)
```
