# Function Differences with torch.nn.AvgPool2d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/AvgPool2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.AvgPool2d

```text
torch.nn.AvgPool2d(
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None
)(input) -> Tensor
```

For more information, see [torch.nn.AvgPool2d](https://PyTorch.org/docs/1.8.1/generated/torch.nn.AvgPool2d.html).

## mindspore.nn.AvgPool2d

```text
mindspore.nn.AvgPool2d(
    kernel_size=1,
    stride=1,
    pad_mode='valid',
    data_format='NCHW'
)(x) -> Tensor
```

For more information, see [mindspore.nn.AvgPool2d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.AvgPool2d.html).

## Differences

PyTorch: Apply two-dimensional averaging pooling to an input signal consisting of multiple input planes.

MindSpore: MindSpore API implements the same function as PyTorch.

| Categories | Subcategories   | PyTorch     | MindSpore   | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | kernel_size  | kernel_size | Consistent function, no default values for PyTorch    |
|      | Parameter 2 | stride     | stride  | Consistent function, different default values of parameters            |
|      | Parameter 3 | padding     | -    | This parameter in PyTorch is used to add implicit zero padding. MindSpore does not have this parameter   |
|      | Parameter 4 | ceil_mode             | -           | This parameter is used in PyTorch to determine the output shape: ($N$,$C$,$H_{out}$,$W_{out}$) where $H_{out}$,$W_{out}$ are fractional, whether to take the upper bound ceil value or to discard the fractional part and take the floor value. MindSpore does not have this parameter and takes the floor value by default. |
|      | Parameter 5 | count_include_pad     | -           | This parameter in PyTorch is used to decide whether to include zero padding in the averaging calculation. MindSpore does not have this parameter |
|      | Parameter 6 | divisor_override=None | -           | If specified in PyTorch, it will be used as a divisor, otherwise kernel_size will be used. MindSpore does not have this parameter |
|      | Parameter 7 | -                     | pad_mode    | Specify the pooling filling mode in MindSpore, optionally with "same" or "valid". PyTorch does not have this parameter             |
|      | Parameter 8 | -                     | data_format    | Specify the input data format in MindSpore, either "NHWC" or "NCHW". PyTorch does not have this parameter |
| Input | Single input | input                 | x           | Same function, different parameter names                               |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
import torch.nn as nn

m = nn.AvgPool2d(kernel_size=1, stride=1)
input_x = torch.tensor([[[[1, 0, 1], [0, 1, 1]]]],dtype=torch.float32)
output = m(input_x)
print(output.numpy())
# [[[[1. 0. 1.]
#    [0. 1. 1.]]]]

# MindSpore
import mindspore
import mindspore.nn as nn
from mindspore import Tensor

pool = nn.AvgPool2d(kernel_size=1, stride=1)
x = Tensor([[[[1, 0, 1], [0, 1, 1]]]], dtype=mindspore.float32)
output = pool(x)
print(output)
# [[[[1. 0. 1.]
#    [0. 1. 1.]]]]
```
