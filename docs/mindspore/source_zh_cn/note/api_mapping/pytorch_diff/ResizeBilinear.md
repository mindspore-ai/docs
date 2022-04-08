# 比较与torch.nn.Upsample的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/ResizeBilinear.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.Upsample

```python
torch.nn.Upsample(
    size=None,
    scale_factor=None,
    mode='nearest',
    align_corners=None
)(input)
```

更多内容详见[torch.nn.Upsample](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Upsample)。

## mindspore.nn.ResizeBilinear

```python
class mindspore.nn.ResizeBilinear()(x, size=None, scale_factor=None, align_corners=False)
```

更多内容详见[mindspore.nn.ResizeBilinear](https://mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.ResizeBilinear.html#mindspore.nn.ResizeBilinear)。

## 使用方式

PyTorch：对数据进行上采样时有多种模式可以选择。

MindSpore：仅支持`bilinear`模式对数据进行采样。

## 代码示例

```python
from mindspore import Tensor
import mindspore.nn as nn
import torch
import numpy as np

# In MindSpore, it is predetermined to use bilinear to resize the input image.
x = np.random.randn(1, 2, 3, 4).astype(np.float32)
resize = nn.ResizeBilinear()
tensor = Tensor(x)
output = resize(tensor, (5, 5))
print(output.shape)
# Out：
# (1, 2, 5, 5)

# In torch, parameter mode should be passed to determine which method to apply for resizing input image.
x = np.random.randn(1, 2, 3, 4).astype(np.float32)
resize = torch.nn.Upsample(size=(5, 5), mode='bilinear')
tensor = torch.tensor(x)
output = resize(tensor)
print(output.shape)
# Out：
# torch.Size([1, 2, 5, 5])
```