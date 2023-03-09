# Function Differences with torch.nn.AvgPool3d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/AvgPool3d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.AvgPool3d

```text
torch.nn.AvgPool3d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)(input) -> Tensor
```

For more information, see [torch.nn.AvgPool3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AvgPool3d.html).

## mindspore.nn.AvgPool3d

```text
mindspore.nn.AvgPool3d(kernel_size=1, stride=1, pad_mode='valid', padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)(x) -> Tensor
```

For more information, see [mindspore.nn.AvgPool3d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.AvgPool3d.html).

## Differences

PyTorch: Perform averaging pooling operations on a one-dimensional plane on the input multidimensional data.

MindSpore: This API implementation function of MindSpore is compatible with TensorFlow and PyTorch, When `pad_mode` is "valid" or "same", the function is consistent with TensorFlow, and when `pad_mode` is "pad", the function is consistent with PyTorch, compared with PyTorch 1.8.1, MindSpore additionally supports 4D input, which is consistent with PyTorch 1.12.

| Categories | Subcategories   | PyTorch     | MindSpore   | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | kernel_size  | kernel_size | Consistent function, no default values for PyTorch    |
|      | Parameter 2 | stride     | stride  | Consistent function, different default values of parameters            |
|      | Parameter 3 | padding     | -    | Consistent|
|      | Parameter 4 | ceil_mode             | -           | Consistent|
|      | Parameter 5 | count_include_pad     | -           | Consistent |
|      | Parameter 6 | divisor_override | -           | Consistent |
|      | Parameter 7 | -                     | pad_mode    | MindSpore specifies how the pooling will be filled, with optional values of "same", "valid" or "pad". PyTorch does not have this parameter|
| Input | Single input | input                 | x           | Same function, different parameter names                               |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
import torch.nn as nn

m = nn.AvgPool3d(kernel_size=6, stride=1)
input_x = torch.tensor([[[1,2,3,4,5,6,7]]], dtype=torch.float32)
print(m(input_x).numpy())
# [[[3.5 4.5]]]

# MindSpore
import mindspore
import mindspore.nn as nn
from mindspore import Tensor

pool = nn.AvgPool3d(kernel_size=6, stride=1)
x = Tensor([[[1,2,3,4,5,6,7]]], dtype=mindspore.float32)
output = pool(x)
print(output)
# [[[3.5 4.5]]]
```

### Code Example 2

> Use pad mode to ensure functional consistency.

```python
import torch
import mindspore.nn as nn
import mindspore.ops as ops

pool = nn.AvgPool3d(4, stride=1, ceil_mode=True, pad_mode='pad', padding=2)
x1 = ops.randn(6, 6, 8, 8, 8)
output = pool(x1)
print(output.shape)
# (6, 6, 9, 9, 9)

pool = torch.nn.AvgPool3d(4, stride=1, ceil_mode=True, padding=2)
x1 = torch.randn(6, 6, 8, 8, 8)
output = pool(x1)
print(output.shape)
# torch.Size([6, 6, 9, 9, 9])
```