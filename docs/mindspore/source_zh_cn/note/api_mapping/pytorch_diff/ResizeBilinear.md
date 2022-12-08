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
class mindspore.nn.ResizeBilinear(half_pixel_centers=False)(
    x,
    size=None,
    scale_factor=None,
    align_corners=False)
```

更多内容详见[mindspore.nn.ResizeBilinear](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.ResizeBilinear.html#mindspore.nn.ResizeBilinear)。

## 使用方式

PyTorch：对数据进行上采样，有多种模式可以选择。

MindSpore：仅当前仅支持`bilinear`模式对数据进行采样，如果想要实现其他模式的采样，请参考[mindspore.ops.interpolate](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.interpolate.html)。
half_pixel_centers默认值为False，设为True后和PyTorch实现功能一致。

## 代码示例

```python
import mindspore as ms
import mindspore.nn as nn
import torch
import numpy as np

# In MindSpore, it is predetermined to use bilinear to resize the input image.
x = np.random.randn(1, 2, 3, 4).astype(np.float32)
resize = nn.ResizeBilinear(half_pixel_centers=True)
tensor = ms.Tensor(x)
output = resize(tensor, (5, 5))
print(output.shape)
# Out：
# (1, 2, 5, 5)

# In torch, parameter mode should be passed to determine which method to apply for resizing input image.
x = np.random.randn(1, 2, 3, 4).astype(np.float32)
resize = torch.nn.Upsample(size=(5, 5), mode='bilinear')
tensor = torch.tensor(x)
output = resize(tensor)
print(output.detach().numpy().shape)
# Out：
# (1, 2, 5, 5)
```