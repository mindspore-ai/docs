# Function Differences with torch.nn.AvgPool1d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/AvgPool1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.AvgPool1d

```text
torch.nn.AvgPool1d(
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True
)(input) -> Tensor
```

For more information, see [torch.nn.AvgPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AvgPool1d.html).

## mindspore.nn.AvgPool1d

```text
mindspore.nn.AvgPool1d(
    kernel_size=1,
    stride=1,
    pad_mode='valid'
)(x) -> Tensor
```

For more information, see [mindspore.nn.AvgPool1d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.AvgPool1d.html).

## Differences

PyTorch: Perform averaging pooling operations on a one-dimensional plane on the input multidimensional data.

MindSpore: MindSpore API implements the same function as PyTorch. MindSpore does not have padding, ceil_mode, or count_include_pad parameters, and PyTorch does not have pad_mode parameters.

| Categories | Subcategories   | PyTorch     | MindSpore   | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | kernel_size | kernel_size | Consistent function, no default values for PyTorch                             |
|      | Parameter 2 | stride            | stride      | Same function, no different default values for parameters                                |
|      | Parameter 3 | padding           | -           | This parameter in PyTorch is used to indicate the number of layers in which each edge of the input is complemented by 0. MindSpore does not have this parameter |
|      | Parameter 4 | ceil_mode         | -           | This parameter in PyTorch is used to determine whether to take the upper bound ceil value or to discard the fractional part and take the floor value when L{out} is a decimal in the output shape: (N, C, L{out}). MindSpore does not have this parameter and takes the floor value by default. |
|      | Parameter 5 | count_include_pad | -           | This parameter in PyTorch is used to decide whether to include padding in the averaging calculation. MindSpore does not have this parameter |
|      | Parameter 6 | -        | pad_mode          | MindSpore specifies how the pooling will be filled, with optional values of "same" or "valid". PyTorch does not have this parameter         |
| Input | Single input | input             | x           | Interface input, same function, different parameter names                               |

### Code Example

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
