# 比较与torch.nn.Unfold的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Unfold.md)

## torch.nn.Unfold

```python
class torch.nn.Unfold(
    kernel_size,
    dilation=1,
    padding=0,
    stride=1
)
```

更多内容详见[torch.nn.Unfold](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Unfold)。

## mindspore.nn.Unfold

```python
class mindspore.nn.Unfold(
    ksizes,
    strides,
    rates,
    padding="valid"
)(x)
```

更多内容详见[mindspore.nn.Unfold](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Unfold.html#mindspore.nn.Unfold)。

## 使用方式

PyTorch：输出的形状，(N,C×∏(kernel_size),L) -> 输出的张量是形状为(N,C×∏(kernel_size),L)的3维张量。

MindSpore：输出张量，数据类型与x相同的4维张量，形状为[out_batch, out_depth, out_row, out_col] 其中 out_batch 与 in_batch 相同。

## 代码示例

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