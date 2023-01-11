# Function Differences with torch.nn.Conv1d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Conv1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Conv1d

```text
class torch.nn.Conv1d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode='zeros'
)(input) -> Tensor
```

For more information, see [torch.nn.Conv1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Conv1d.html).

## mindspore.nn.Conv1d

```text
class mindspore.nn.Conv1d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    pad_mode='same',
    padding=0,
    dilation=1,
    group=1,
    has_bias=False,
    weight_init='normal',
    bias_init='zeros'
)(x) -> Tensor
```

For more information, see [mindspore.nn.Conv1d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Conv1d.html).

## Differences

PyTorch: To compute a one-dimensional convolution on the input Tensor. The output values of input size $\left(N, C_{\text {in }}, L\right)$ and output size $\left(N, C_{\text {out }}, L_{\text {out }}\right)$ can be described as
$\operatorname{out}\left(N_{i}, C_{\text {out }_{j}}\right)=\operatorname{bias}\left(C_{\text {out }_{j}}\right)+\sum_{k=0}^{C_{i n}-1} \ text { weight }\left(C_{\text {out }_{j}}, k\right) \star \operatorname{input}\left(N_{i}, k\right)$, where $\star$ is the cross-correlation operator, $N$ is the batch size, $C$ is the number of channels, and $L$ is sequence length, respectively.

MindSpore: Implement basically the same function as PyTorch, but does not add bias parameters by default, in contrast to PyTorch. MindSpore pads the input by default, while PyTorch does not by default. Also MindSpore padding mode options are different from PyTorch. The padding_mode options of PyTorch are 'zeros', 'reflect', 'replicate', and 'circular', with the following meanings:

zero: Constant padding (default zero padding).

reflect: Reflection padding.

replicate: replicate padding.

circular: Circular padding

MindSpore parameter pad_mode can be optionally 'same', 'valid', 'pad', with the following meanings:

same: The width of the output is the same as the value after dividing stride by the input.

valid: No padding

pad: Zero padding.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | in_channels | in_channels |- |
| | Parameter 2 | out_channels | out_channels |- |
| | Parameter 3 | kernel_size | kernel_size |- |
| | Parameter 4 | stride | stride |- |
| | Parameter 5 | padding | padding |-|
| | Parameter 6 | dilation | dilation |-|
| | Parameter 7 | groups | group |Same function, different parameter names|
| | Parameter 8 | bias | has_bias |Same function, different parameter names, different default value |
| | Parameter 9 | padding_mode | pad_mode |PyTorch and MindSpore have different options and different default values|
| | Parameter 10 | - | weight_init |Initialization method for weight parameters|
| | Parameter 11 | - | bias_init |Initialization method for bias parameters|
| | Parameter 12  | input  | x | Interface input, same function, only different parameter names|

### Code Example 1

> The default value of PyTorch parameter bias is True, which means that bias parameters are added by default, while the default value of MindSpore parameter has_bias is False, which means that bias functions are not added by default. If the bias parameter is added, you need to set the value of has_bias to True.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((1, 120, 640))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv1d(120, 240, 4)
output = net(x).detach().numpy().shape
print(output)
# (1, 240, 637)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((1, 120, 640))
x = Tensor(x_, mindspore.float32)
net = nn.Conv1d(120, 240, 4, has_bias=True, pad_mode='valid')
output = net(x).shape
print(output)
# (1, 240, 637)
```

### Code Example 2

> PyTorch parameter padding_mode is 'zero', which means zero padding for the input, while implementing zero padding in MindSpore requires setting the parameter pad_mode to 'pad'.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((1, 120, 640))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv1d(120, 240, 4, padding=1, padding_mode='zeros')
output = net(x).detach().numpy().shape
print(output)
# (1, 240, 639)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((1, 120, 640))
x = Tensor(x_, mindspore.float32)
net = nn.Conv1d(120, 240, 4, padding=1, pad_mode='pad')
output = net(x).shape
print(output)
# (1, 240, 639)
```

### Code Example 3

> When PyTorch parameter padding_mode is 'reflect', it means reflective padding of the input, and the reflective padding in MindSpore needs to be implemented through a combination of APIs, first calling nn.Pad to reflectively pad the input x, and then convolution of the padded result.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((1, 120, 640))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv1d(120, 240, 4, padding=1, padding_mode='reflect')
output = net(x).detach().numpy().shape
print(output)
# (1, 240, 639)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np
x_ = np.ones((1, 120, 640))
x = Tensor(x_, mindspore.float32)
pad = nn.Pad(paddings=((0,0),(1,1),(1,1)), mode="REFLECT")
x_pad = pad(x)
net = nn.Conv1d(122, 240, 4, padding=0, pad_mode='valid')
output = net(x_pad).shape
print(output)
# (1, 240, 639)
```

### Code Example 4

> PyTorch does not pad the input by default, while MindSpore does by default. If don't pad the input, you need to set pad_mode to 'valid'.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((1, 120, 640))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv1d(120, 240, 4)
output = net(x).detach().numpy().shape
print(output)
# (1, 240, 637)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((1, 120, 640))
x = Tensor(x_, mindspore.float32)
net = nn.Conv1d(120, 240, 4, pad_mode='valid')
output = net(x).shape
print(output)
# (1, 240, 637)
```
