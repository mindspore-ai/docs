# Function Differences with torch.nn.Conv2d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Conv2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Conv2d

```text
class torch.nn.Conv2d(
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

For more information, see [torch.nn.Conv2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Conv2d.html).

## mindspore.nn.Conv2d

```text
class mindspore.nn.Conv2d(
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
    bias_init='zeros',
    data_format='NCHW'
)(x) -> Tensor
```

For more information, see [mindspore.nn.Conv2d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Conv2d.html).

## Differences

PyTorch: To compute a two-dimensional convolution on the input Tensor. Typically the output values with input size $\left(N, C_{\mathrm{in}}, H, W\right)$ and output size $\left(N, C_{\text {out }}, H_{\text {out }}, W_{\text {out }}\right)$ can be described as $\operatorname{out}\left(N_{i}, C_{\text {out }_{j}}\right)=\operatorname{bias}\left(C_{\text {out }_{j}}\right)+\sum_{k=0}^{C_{i n}-1} \text { weight }\left(C_{\text {out }_{j}}, k\right) \star \operatorname{input}\left(N_{i}, k\right)$, where $\star$ is the 2D cross-correlation operator, $N$ is the batch size, $C$ is the number of channels, and $H$ and $W$ are the height and width of the feature layer, respectively.

MindSpore: Essentially the same as the PyTorch implementation, but with bias differences and padding differences.

1. bias differences: MindSpore does not add bias parameters by default, in contrast to PyTorch.
2. padding differences: MindSpore pads the input by default, while PyTorch does not pad by default. The MindSpore padding mode options are different from PyTorch, which has 'zeros', 'reflect', 'replicate', and 'circular' as parameters, with the following meanings:

zero: Constant padding (default zero padding)

reflect: reflection padding

replicate: replication padding

circular: Circular padding

MindSpore parameter pad_mode can be optionally 'same', 'valid', 'pad', with the following meanings:

same: Using 0 padding makes the height and width of the output the same as the value of the input after dividing by stride, respectively. Using this mode, the value of padding must be 0.

valid: No padding

pad: Zero padding. Using this mode, padding must be greater than or equal to 0.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | in_channels | in_channels |- |
| | Parameter 2 | out_channels | out_channels |- |
| | Parameter 3 | kernel_size | kernel_size |- |
| | Parameter 4 | stride | stride |- |
| | Parameter 5 | padding | padding |-|
| | Parameter 6 | padding_mode | pad_mode |PyTorch and MindSpore have different options and different default values|
| | Parameter 7 | dilation | dilation |-|
| | Parameter 8 | groups | group |Same function, different parameter names|
| | Parameter 9 | bias | has_bias |Same function, different parameter names, different default values|
| | Parameter 10 | - | weight_init |The initialization method for the weight parameter. PyTorch can use the init function to initialize the weights|
| | Parameter 11 | - | bias_init |Initialization method for the bias parameter, which is not available for PyTorch|
| | Parameter 12 | - | data_format |Specifies the input data format. PyTorch does not have this parameter|
| |Parameter 13  | input | x  | Interface input, same function, only different parameter names |

### Code Example 1

> The default value of PyTorch parameter bias is True, which means that bias parameters are added by default, while the default value of MindSpore parameter has_bias is False, which means that bias functions are not added by default.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((1, 120, 1024, 640))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv2d(120, 240, 4)
output = net(x).detach().numpy().shape
print(output)
# (1, 240, 1021, 637)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((1, 120, 1024, 640))
x = Tensor(x_, mindspore.float32)
net = nn.Conv2d(120, 240, 4, has_bias=True, pad_mode='valid')
output = net(x).shape
print(output)
# (1, 240, 1021, 637)
```

### Code Example 2

> PyTorch parameter padding_mode is 'zero', which means zero padding for the input, while implementing zero padding in MindSpore requires setting the parameter pad_mode to 'pad'.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((1, 120, 1024, 640))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv2d(120, 240, 4, padding=1, padding_mode='zeros')
output = net(x).detach().numpy().shape
print(output)
# (1, 240, 1023, 639)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((1, 120, 1024, 640))
x = Tensor(x_, mindspore.float32)
net = nn.Conv2d(120, 240, 4, padding=1, pad_mode='pad')
output = net(x).shape
print(output)
# (1, 240, 1023, 639)
```

### Code Example 3

> When PyTorch parameter padding_mode is 'reflect', it means reflective padding of the input, and the reflective padding in MindSpore needs to be implemented through a combination of APIs, first calling nn.Pad to reflectively pad the input x, and then convolution of the padded result.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((1, 120, 1024, 640))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv2d(120, 240, 4, padding=1, padding_mode='reflect')
output = net(x).detach().numpy().shape
print(output)
# (1, 240, 1023, 639)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((1, 120, 1024, 640))
x = Tensor(x_, mindspore.float32)
pad = nn.Pad(paddings=((0,0),(1,1),(1,1),(1,1)), mode="REFLECT")
x_pad = pad(x)
net = nn.Conv2d(122, 240, 4, padding=0, pad_mode='valid')
output = net(x_pad).shape
print(output)
# (1, 240, 1023, 639)
```

### Code Example 4

> PyTorch does not pad the input by default, while MindSpore requires it by default. If don't pad the input, you need to set pad_mode to 'valid'.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((1, 120, 1024, 640))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv2d(120, 240, 4)
output = net(x).detach().numpy().shape
print(output)
# (1, 240, 1021, 637)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((1, 120, 1024, 640))
x = Tensor(x_, mindspore.float32)
net = nn.Conv2d(120, 240, 4, pad_mode='valid')
output = net(x).shape
print(output)
# (1, 240, 1021, 637)
```
