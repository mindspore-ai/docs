# Function Differences with torch.nn.MaxPool1d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/MaxPool1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [torch.nn.MaxPool1d](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.MaxPool1d).

## mindspore.nn.MaxPool1d

```python
class mindspore.nn.MaxPool1d(
    kernel_size=1,
    stride=1,
    pad_mode="valid"
)
```

For more information, see [mindspore.nn.MaxPool1d](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.MaxPool1d.html#mindspore.nn.MaxPool1d).

## Differences

PyTorch：The output shape can be adjusted through the padding parameter. If the shape of input is $ (N, C, L_{in}) $，the shape of output is $ (N, C, L_{out}) $, where

$$
        L_{out} = \left\lfloor \frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                    \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor
$$

MindSpore：There is no padding parameter, the pad mode is controlled by the pad_mode parameter only.  If the shape of input is $ (N, C, L_{in}) $，the shape of output is $ (N, C, L_{out}) $, where

1. pad_mode is "valid"：

   $$
        L_{out} = \left\lfloor \frac{L_{in} - (\text{kernel_size} - 1)}{\text{stride}}\right\rfloor
   $$

2. pad_mode is "same"：

   $$
        L_{out} = \left\lfloor \frac{L_{in}}{\text{stride}}\right\rfloor
   $$

## Code Example

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