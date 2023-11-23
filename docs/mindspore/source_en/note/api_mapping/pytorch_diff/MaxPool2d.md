# Differences with torch.nn.MaxPool2d

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/MaxPool2d.md)

## torch.nn.MaxPool2d

```text
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)(input) -> Tensor
```

For more information, see [torch.nn.MaxPool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxPool2d.html).

## mindspore.nn.MaxPool2d

```text
mindspore.nn.MaxPool2d(kernel_size=1, stride=1, pad_mode="valid", padding=0, dilation=1, return_indices=False, ceil_mode=False, data_format="NCHW")(x) -> Tensor
```

For more information, see [mindspore.nn.MaxPool2d](https://www.mindspore.cn/docs/en/r2.3/api_python/nn/mindspore.nn.MaxPool2d.html).

## Differences

PyTorch: Perform two-dimensional maximum pooling operations on the input multidimensional data.

MindSpore: This API implementation function of MindSpore is compatible with TensorFlow and PyTorch, When `pad_mode` is "valid" or "same", the function is consistent with TensorFlow, and when `pad_mode` is "pad", the function is consistent with PyTorch, MindSpore additionally supports 2D input, which is consistent with PyTorch 1.12.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | kernel_size | kernel_size |Consistent function, no default values for PyTorch |
| | Parameter 2 | stride | stride |Consistent function, different default value |
| | Parameter 3 | padding |padding| Consistent |
| | Parameter 4 | dilation | dilation | Consistent |
| | Parameter 5 | return_indices | return_indices | Consistent|
| | Parameter 6 | ceil_mode | ceil_mode | Consistent |
| | Parameter 7 | input | x | Consistent function, different parameter names |
| | Parameter 8 | - | pad_mode | Control the padding mode, and PyTorch does not have this parameter |
| | Parameter 9 | - | data_format | The input data format can be "NHWC" or "NCHW". Default value: "NCHW" |

### Code Example 1

> Construct a pooling layer with a convolution kernel size of 1x3 and a step size of 1. The padding defaults to 0 and no element filling is performed. The default value of dilation is 1, and the elements in the window are contiguous. Pooling padding mode returns the output from a valid calculation without padding, and excess pixels that do not satisfy the calculation are discarded. With the same parameter settings, the two APIs achieve the same function to perform two-dimensional maximum pooling operations on the input multidimensional data.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

pool = torch.nn.MaxPool2d(kernel_size=3, stride=1)
x = tensor(np.random.randint(0, 10, [1, 2, 4, 4]), dtype=torch.float32)
output = pool(x)
result = output.shape
print(tuple(result))
# (1, 2, 2, 2)

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

pool = mindspore.nn.MaxPool2d(kernel_size=3, stride=1)
x = Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
output = pool(x)
result = output.shape
print(result)
# (1, 2, 2, 2)
```

### Code Example 2

> Use `pad` mode to ensure functional consistency.

```python
# PyTorch
import torch
import numpy as np

np_x = np.random.randint(0, 10, [1, 2, 4, 4])
x = torch.tensor(np_x, dtype=torch.float32)
max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=1, return_indices=False)
output = max_pool(x)
result = output.shape
print(tuple(result))
# (1, 2, 5, 5)

# MindSpore
import mindspore as ms
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

np_x = np.random.randint(0, 10, [1, 2, 4, 4])
x = Tensor(np_x, ms.float32)
max_pool = nn.MaxPool2d(kernel_size=2, stride=1, pad_mode='pad', padding=1, dilation=1, return_indices=False)
output = max_pool(x)
result = output.shape
print(result)
# (1, 2, 5, 5)
```
