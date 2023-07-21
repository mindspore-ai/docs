# Function Differences with torch.nn.MaxPool2d

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_en/note/api_mapping/pytorch_diff/MaxPool2d.md)

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

For more information, see [torch.nn.MaxPool2d](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.MaxPool2d).

## mindspore.nn.MaxPool2d

```python
class mindspore.nn.MaxPool2d(
    kernel_size=1,
    stride=1,
    pad_mode="valid",
    data_format="NCHW"
)
```

For more information, see [mindspore.nn.MaxPool2d](https://mindspore.cn/docs/en/r1.9/api_python/nn/mindspore.nn.MaxPool2d.html#mindspore.nn.MaxPool2d).

## Differences

PyTorch：The output shape can be adjusted through the padding parameter. If the shape of input is $ (N, C, H_{in}, W_{in}) $，the shape of output is $ (N, C, H_{out}, W_{out}) $, where

$$
        H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                  \times (\text{kernel_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor
$$

$$
        W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                  \times (\text{kernel_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor
$$

MindSpore：There is no padding parameter, the pad mode is controlled by the pad_mode parameter only.  If the shape of input is $ (N, C, H_{in}, W_{in}) $，the shape of output is $ (N, C, H_{out}, W_{out}) $, where

1. pad_mode is "valid"：

   $$
        H_{out} = \left\lceil\frac{H_{in} - ({kernel\_size[0]} - 1)}{\text{stride[0]}}\right\rceil
   $$

   $$
        W_{out} = \left\lceil\frac{W_{in} - ({kernel\_size[1]} - 1)}{\text{stride[1]}}\right\rceil
   $$

2. pad_mode is "same"：

   $$
        H_{out} = \left\lceil\frac{H_{in}}{\text{stride[0]}}\right\rceil
   $$

   $$
        W_{out} = \left\lceil\frac{W_{in}}{\text{stride[1]}}\right\rceil
   $$

## Code Example

```python
import mindspore as ms
import mindspore.nn as nn
import torch
import numpy as np

# In MindSpore, pad_mode="valid"
pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
input_x = ms.Tensor(np.random.randn(20, 16, 50, 32).astype(np.float32))
output = pool(input_x)
print(output.shape)
# Out：
# (20, 16, 24, 15)

# In MindSpore, pad_mode="same"
pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
input_x = ms.Tensor(np.random.randn(20, 16, 50, 32).astype(np.float32))
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