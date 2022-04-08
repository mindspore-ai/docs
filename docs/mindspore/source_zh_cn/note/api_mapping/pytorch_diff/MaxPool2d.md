# 比较与torch.nn.MaxPool2d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/MaxPool2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.MaxPool2d

```python
torch.nn.MaxPool2d(
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    return_indices=False,
    ceil_mode=False
)
```

更多内容详见[torch.nn.MaxPool2d](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.MaxPool2d)。

## mindspore.nn.MaxPool2d

```python
class mindspore.nn.MaxPool2d(
    kernel_size=1,
    stride=1,
    pad_mode="valid",
    data_format="NCHW"
)
```

更多内容详见[mindspore.nn.MaxPool2d](https://mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.MaxPool2d.html#mindspore.nn.MaxPool2d)。

## 使用方式

PyTorch：可以通过padding参数调整输出的shape。若输入的shape为 $ (N, C, H_{in}, W_{in}) $，则输出的shape为 $ (N, C, H_{out}, W_{out}) $，其中

$$
        H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                  \times (\text{kernel_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor
$$

$$
        W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                  \times (\text{kernel_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor
$$

MindSpore：没有padding参数，仅通过pad_mode参数控制pad模式。若输入的shape为 $ (N, C, H_{in}, W_{in}) $，则输出的shape为 $ (N, C, H_{out}, W_{out}) $，其中

1. pad_mode为"valid"：

   $$
        H_{out} = \left\lfloor\frac{H_{in} - ({kernel\_size[0]} - 1)}{\text{stride[0]}}\right\rfloor
   $$

   $$
        W_{out} = \left\lfloor\frac{W_{in} - ({kernel\_size[1]} - 1)}{\text{stride[1]}}\right\rfloor
   $$

2. pad_mode为"same"：

   $$
        H_{out} = \left\lfloor\frac{H_{in}}{\text{stride[0]}}\right\rfloor
   $$

   $$
        W_{out} = \left\lfloor\frac{W_{in}}{\text{stride[1]}}\right\rfloor
   $$

## 代码示例

```python
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import torch
import numpy as np

# In MindSpore, pad_mode="valid"
pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
input_x = Tensor(np.random.randn(20, 16, 50, 32).astype(np.float32))
output = pool(input_x)
print(output.shape)
# Out：
# (20, 16, 24, 15)

# In MindSpore, pad_mode="same"
pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
input_x = Tensor(np.random.randn(20, 16, 50, 32).astype(np.float32))
output = pool(input_x)
print(output.shape)
# Out：
# (20, 16, 25, 16)


# In torch, padding=1
m = torch.nn.MaxPool2d(3, stride=2, padding=1)
input_x = torch.randn(20, 16, 50, 32)
output = m(input_x)
print(output.shape)
# Out：
# torch.Size([20, 16, 25, 16])
```