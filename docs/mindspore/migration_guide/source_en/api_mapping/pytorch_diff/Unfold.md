# Function Differences with torch.nn.Unfold

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/Unfold.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## torch.nn.Unfold

```python
class torch.nn.Unfold(
    kernel_size,
    dilation=1,
    padding=0,
    stride=1
)
```

For more information, see [torch.nn.Unfold](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Unfold).

## mindspore.nn.Unfold

```python
class mindspore.nn.Unfold(
    ksizes,
    strides,
    rates,
    padding="valid"
)(x)
```

For more information, see [mindspore.nn.Unfold](https://mindspore.cn/docs/api/en/r1.6/api_python/nn/mindspore.nn.Unfold.html#mindspore.nn.Unfold).

## Differences

PyTorch：The shape of output, (N,C×∏(kernel_size),L) -> The tensor of output, a 3-D tensor whose shape is (N, C×∏(kernel_size), L).

MindSpore：The tensor of output, a 4-D tensor whose data type is same as x, and the shape is [out_batch, out_depth, out_row, out_col]
where out_batch is the same as the in_batch.

## Code Example

```python
from mindspore import Tensor
import mindspore.nn as nn
from mindspore import dtype as mstype
import torch
import numpy as np

unfold = torch.nn.Unfold(kernel_size=(2, 3))
input = torch.ones(2, 5, 3, 4)
output = unfold(input)
print(output.size())
# Out：
# torch.Size([2, 30, 4])

net = nn.Unfold(ksizes=[1, 2, 2, 1], strides=[1, 2, 2, 1], rates=[1, 2, 2, 1])
image = Tensor(np.ones([2, 5, 3, 4]), dtype=mstype.float16)
output = net(image)
print(output.shape)
# Out：
# (2, 20, 1, 1)
```