# 比较与torch.nn.MaxPool1d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/MaxPool1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.MaxPool1d

```python
torch.nn.MaxPool1d(
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    return_indices=False,
    ceil_mode=False
)
```

更多内容详见[torch.nn.MaxPool1d](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.MaxPool1d)。

## mindspore.nn.MaxPool1d

```python
class mindspore.nn.MaxPool1d(
    kernel_size=1,
    stride=1,
    pad_mode="valid"
)
```

更多内容详见[mindspore.nn.MaxPool1d](https://mindspore.cn/docs/zh-CN/r1.7/api_python/nn/mindspore.nn.MaxPool1d.html#mindspore.nn.MaxPool1d)。

## 使用方式

PyTorch：可以通过padding参数调整输出的shape。若输入的shape为 $ (N, C, L_{in}) $，则输出的shape为 $ (N, C, L_{out}) $，其中

$$
        L_{out} = \left\lfloor \frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                    \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor
$$

MindSpore：没有padding参数，仅通过pad_mode参数控制pad模式。若输入的shape为 $ (N, C, L_{in}) $，则输出的shape为 $ (N, C, L_{out}) $，其中

1. pad_mode为"valid"：

   $$
        L_{out} = \left\lfloor \frac{L_{in} - (\text{kernel_size} - 1)}{\text{stride}}\right\rfloor
   $$

2. pad_mode为"same"：

   $$
        L_{out} = \left\lfloor \frac{L_{in}}{\text{stride}}\right\rfloor
   $$

## 代码示例

```python
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import torch
import numpy as np

# In MindSpore, pad_mode="valid"
pool = nn.MaxPool1d(kernel_size=3, stride=2, pad_mode="valid")
input_x = Tensor(np.random.randn(20, 16, 50).astype(np.float32))
output = pool(input_x)
print(output.shape)
# Out：
# (20, 16, 24)

# In MindSpore, pad_mode="same"
pool = nn.MaxPool1d(kernel_size=3, stride=2, pad_mode="same")
input_x = Tensor(np.random.randn(20, 16, 50).astype(np.float32))
output = pool(input_x)
print(output.shape)
# Out：
# (20, 16, 25)


# In torch, padding=1
m = torch.nn.MaxPool1d(3, stride=2, padding=1)
input_x = torch.randn(20, 16, 50)
output = m(input_x)
print(output.shape)
# Out：
# torch.Size([20, 16, 25])
```