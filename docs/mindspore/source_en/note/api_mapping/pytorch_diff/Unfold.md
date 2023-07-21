# Function Differences with torch.nn.Unfold

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.8/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Unfold.md)

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

For more information, see [mindspore.nn.Unfold](https://mindspore.cn/docs/en/r1.8/api_python/nn/mindspore.nn.Unfold.html#mindspore.nn.Unfold).

## Differences

PyTorch：The shape of output, (N,C×∏(kernel_size),L) -> The tensor of output, a 3-D tensor whose shape is (N, C×∏(kernel_size), L).

MindSpore：The tensor of output, a 4-D tensor whose data type is same as x, and the shape is [out_batch, out_depth, out_row, out_col]
where out_batch is the same as the in_batch.

## Code Example

```python
import mindspore as ms
import mindspore.nn as nn
import torch
import numpy as np

unfold = torch.nn.Unfold(kernel_size=(2, 3))
input = torch.ones(2, 5, 3, 4)
output = unfold(input)
print(output.size())
# Out：
# torch.Size([2, 30, 4])

net = nn.Unfold(ksizes=[1, 2, 2, 1], strides=[1, 2, 2, 1], rates=[1, 2, 2, 1])
image = ms.Tensor(np.ones([2, 5, 3, 4]), dtype=ms.float16)
output = net(image)
print(output.shape)
# Out：
# (2, 20, 1, 1)
```