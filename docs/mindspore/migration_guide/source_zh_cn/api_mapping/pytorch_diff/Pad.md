# 比较与torch.nn.functional.pad的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/Pad.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

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

更多内容详见[mindspore.nn.Pad](https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Pad.html#mindspore.nn.Pad)。

## 使用方式

PyTorch：pad参数是一个有m个值的tuple，m/2小于等于输入数据的维度，且m为偶数。支持填充负维度。

MindSpore：paddings参数是一个shape为（n， 2）的tuple，n为输入数据的维度。当前不支持填充负维度，可用ops.Slice切小。

## 代码示例

```python
# In MindSpore.
import numpy as np
import torch
import mindspore.nn as nn
from mindspore import Tensor

x = Tensor(np.ones(3, 3).astype(np.float32))
pad_op = nn.Pad(paddings=((0, 0), (1, 1)))
output = pad_op(x)
print(output.shape)
# Out:
# (3, 5)

# In Pytorch.
x = torch.empty(3, 3)
pad = (1, 1)
output = torch.nn.functional.pad(x, pad)
print(output.size())
# Out:
# (3, 5)
```
