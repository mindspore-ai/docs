# Function Differences with torch.nn.MaxPool1d

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_en/note/api_mapping/pytorch_diff/MaxPool1d.md)

## torch.nn.MaxPool1d

```text
torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)(input) -> Tensor
```

For more information, see [torch.nn.MaxPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxPool1d.html).

## mindspore.nn.MaxPool1d

```text
mindspore.nn.MaxPool1d(kernel_size=1, stride=1, pad_mode="valid", padding=0, dilation=1, return_indices=False, ceil_mode=False)(x) -> Tensor
```

For more information, see [mindspore.nn.MaxPool1d](https://www.mindspore.cn/docs/en/r1.11/api_python/nn/mindspore.nn.MaxPool1d.html).

## Differences

PyTorch: Perform maximum pooling operations on temporal data.

MindSpore: This API implementation function of MindSpore is compatible with TensorFlow and PyTorch, When `pad_mode` is "valid" or "same", the function is consistent with TensorFlow, and when `pad_mode` is "pad", the function is consistent with PyTorch, MindSpore additionally supports 2D input, which is consistent with PyTorch 1.12.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | kernel_size | kernel_size | Consistent function, no default values for PyTorch |
| | Parameter 2 | stride | stride | Consistent function, different default value  |
| | Parameter 3 | padding |padding | Consistent |
| | Parameter 4 | dilation | dilation | Consistent |
| | Parameter 5 | return_indices | return_indices | Consistent |
| | Parameter 6 | ceil_mode | ceil_mode | Consistent |
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

> Use pad mode to ensure functional consistency.

```python
import mindspore as ms
from mindspore import Tensor
import mindspore.nn as nn
import torch
import numpy as np

np_x = np.random.randint(0, 10, [1, 2, 4])

x = Tensor(np_x, ms.float32)
max_pool = nn.MaxPool1d(kernel_size=2, stride=1, pad_mode='pad', padding=1, dilation=1, return_indices=False)
output = max_pool(x)
result = output.shape
print(result)
# (1, 2, 5)
x = torch.tensor(np_x, dtype=torch.float32)
max_pool = torch.nn.MaxPool1d(kernel_size=2, stride=1, padding=1, dilation=1, return_indices=False)
output = max_pool(x)
result = output.shape
print(result)
# torch.Size([1, 2, 5])
```