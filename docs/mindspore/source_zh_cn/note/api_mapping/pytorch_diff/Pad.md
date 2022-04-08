# 比较与torch.nn.functional.pad的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/Pad.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.functional.pad

```python
class torch.nn.functional.pad(
    input
    pad,
    mode='constant',
    value=0.0
)
```

更多内容详见[torch.nn.functional.pad](https://pytorch.org/docs/1.5.0/nn.functional.html#torch.nn.functional.pad)。

## mindspore.nn.Pad

```python
class mindspore.nn.Pad(
    paddings,
    mode="CONSTANT"
)(x)
```

更多内容详见[mindspore.nn.Pad](https://mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.Pad.html#mindspore.nn.Pad)。

## 使用方式

PyTorch：pad参数是一个有m个值的tuple，m/2小于等于输入数据的维度，且m为偶数。支持填充负维度。假设pad=(k1, k2, ..., kl, km)，输入x的shape为(d1, d2..., dg)，则dg维的两边分别填充长度为k1，k2的值。依此类推，d1维的两边分别填充长度为kl，km的值。

MindSpore：paddings参数是一个shape为(n, 2)的tuple，n为输入数据的维度。对于输入x的D维，对应输出D维的大小等于paddings[D, 0] + x.dim_size(D) + paddings[D, 1]。当前不支持填充负维度，可用ops.Slice切小。假设输入x为的shape为(1, 2, 2, 3)，Pytorch的pad参数为(1, 1, 2, 2)，要使MindSpore的输出shape与Pytorch的一致，则paddings参数应为((0, 0), (0, 0), (2, 2), (1, 1))，输出的shape为(1, 2, 6, 5)。

## 代码示例

```python
# In MindSpore.
import numpy as np
import torch
import mindspore.nn as nn
from mindspore import Tensor

x = Tensor(np.ones([1, 2, 2, 3]).astype(np.float32))
pad_op = nn.Pad(paddings=((0, 0), (0, 0), (2, 2), (1, 1)))
output = pad_op(x)
print(output.shape)
# Out:
# (1, 2, 6, 5)

# In Pytorch.
x = torch.empty(1, 2, 2, 3)
pad = (1, 1, 2, 2)
output = torch.nn.functional.pad(x, pad)
print(output.size())
# Out:
# torch.Size([1, 2, 6, 5])
```
