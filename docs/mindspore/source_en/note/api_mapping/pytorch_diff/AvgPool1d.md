# Function Differences with torch.nn.AvgPool1d

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/AvgPool1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

## torch.nn.AvgPool1d

```text
torch.nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)(input) -> Tensor
```

For more information, see [torch.nn.AvgPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AvgPool1d.html).

## mindspore.nn.AvgPool1d

```text
mindspore.nn.AvgPool1d(kernel_size=1, stride=1, pad_mode="valid", padding=0, ceil_mode=False, count_include_pad=True)(x) -> Tensor
```

For more information, see [mindspore.nn.AvgPool1d](https://www.mindspore.cn/docs/en/r2.0/api_python/nn/mindspore.nn.AvgPool1d.html).

## Differences

PyTorch: Perform averaging pooling operations on a one-dimensional plane on the input multidimensional data.

MindSpore: This API implementation function of MindSpore is compatible with TensorFlow and PyTorch, When `pad_mode` is "valid" or "same", the function is consistent with TensorFlow, and when `pad_mode` is "pad", the function is consistent with PyTorch, MindSpore additionally supports 2D input, which is consistent with PyTorch 1.12.

| Categories | Subcategories   | PyTorch     | MindSpore   | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | kernel_size | kernel_size | Consistent function, no default values for PyTorch                             |
|      | Parameter 2 | stride            | stride      | The functions are the same, but the default values of parameters are different |
|      | Parameter 3 | padding           | padding    | Consistent |
|      | Parameter 4 | ceil_mode         | ceil_mode    | Consistent |
|      | Parameter 5 | count_include_pad | count_include_pad   | Consistent |
|      | Parameter 6 | -        | pad_mode          | MindSpore specifies how the pooling will be filled, with optional values of "same", "valid" or "pad". PyTorch does not have this parameter         |
| Input | Single input | input             | x           | Interface input, same function, different parameter names                               |

### Code Example 1

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
import torch.nn as nn

m = nn.AvgPool1d(kernel_size=6, stride=1)
input_x = torch.tensor([[[1,2,3,4,5,6,7]]], dtype=torch.float32)
print(m(input_x).numpy())
# [[[3.5 4.5]]]

# MindSpore
import mindspore
import mindspore.nn as nn
from mindspore import Tensor

pool = nn.AvgPool1d(kernel_size=6, stride=1)
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

pool = nn.AvgPool1d(4, stride=1, ceil_mode=True, pad_mode='pad', padding=2)
x1 = ops.randn(6, 6, 8)
output = pool(x1)
print(output.shape)
# (6, 6, 9)

pool = torch.nn.AvgPool1d(4, stride=1, ceil_mode=True, padding=2)
x1 = torch.randn(6, 6, 8)
output = pool(x1)
print(output.shape)
# torch.Size([6, 6, 9])
```