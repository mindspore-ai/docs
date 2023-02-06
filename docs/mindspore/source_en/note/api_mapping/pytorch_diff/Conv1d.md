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

MindSpore: It is basically the same as the functions implemented by PyTorch, but there are bias differences and filling differences.

1. Offset difference: MindSpore does not add offset parameters by default, contrary to PyTorch.
2. Fill difference: MindSpore fills the input by default, while PyTorch does not fill by default. At the same time, MindSpore filling mode options and behavior are different from PyTorch. The specific differences in filling behavior are as follows.

### Filling Behavior Difference

1. The parameter "padding_mode" of PyTorch can be selected as 'zero', 'reflect', 'replicate', and 'circular'. The default is 'zero'. The parameter "padding" can be selected as 'int', 'tuple of ints', 'valid', and 'same'. The default is 0. The four padding mode of the parameter "padding_mode" is consistent with that of the "torch.nn.functional.pad" interface. After setting, the convolution input will be filled according to the specified filling mode, as follows:

    - zero: constant fill (default zero fill).

    - reflect: reflection fill.

    - replicate: Edge replication fill.

    - circular: circular fill.

    After the filling method is determined by "padding _mode", the "padding" parameter is used to control the number and position of filling. For "Conv1d", when "padding" is specified as 'int', "padding" times will be filled in the left and right sides of the input (if the default value is 0, it means no filling). When "padding" is specified as tuple, the specified number of filling will be filled in the left and right sides according to the input of tuple. When "padding" is set to the 'valid' mode, it will not be filled, but will only be convolved within the range of the feature map. When "padding" is set to the 'same' mode, if the number of elements requiring "padding" is even, padding elements are evenly distributed on the top, bottom, left, and right of the feature map. If the number of elements requiring "padding" is odd, PyTorch will fill the left and upper sides of the feature map first.

2. The parameter "pad_mode" of MindSpore can be selected as 'same','valid', and 'pad'. The parameter "padding" can only be input as "int". The detailed meaning of the filling parameter is as follows:

    When "pad_mode" is set to 'pad', "MindSpore" can set the "padding" parameter to a positive integer greater than or equal to 0. Zero filling will be carried out "padding" times around the input(if it is the default value of 0, it will not fill). When "pad_mode" is the other two modes, the "padding" parameter must be set to 0 only. When "pad_mode" is set to 'valid' mode, it will not fill, and the convolution will only be carried out within the range of the feature map. If "pad_mode" is set to 'same' mode, when the padding element is an even number, padding elements are evenly distributed on the top, bottom, left, and right of the feature map. If the number of elements requiring "padding" is odd, "MindSpore" will preferentially fill the right and lower sides of the feature map (different from PyTorch, similar to TensorFlow).

    Therefore, if "MindSpore" wants to achieve the same filling mode as "PyTorch", it needs to manually fill the input with "nn.Pad" or "ops.pad" interface.

### Weight Initialization Difference

1. mindspore.nn.Conv1d (weight：$\mathcal{N}(0, 1)$，bias：zeros)
2. torch.nn.Conv1d (weight：$\mathcal{U} (-\sqrt{k},\sqrt{k} )$，bias：$\mathcal{U} (-\sqrt{k},\sqrt{k} )$)

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | in_channels | in_channels |- |
| | Parameter 2 | out_channels | out_channels |- |
| | Parameter 3 | kernel_size | kernel_size |- |
| | Parameter 4 | stride | stride |- |
| | Parameter 5 | padding | padding |Refer to the above for specific differences|
| | Parameter 6 | dilation | dilation |-|
| | Parameter 7 | groups | group |Same function, different parameter names|
| | Parameter 8 | bias | has_bias |Same function, different parameter names, different default value |
| | Parameter 9 | padding_mode | pad_mode |Refer to the above for specific differences|
| | Parameter 10 | - | weight_init |Initialization method for weight parameters|
| | Parameter 11 | - | bias_init |Initialization method for bias parameters|
|Input | Single input | input | x | Same functions, different parameter names |

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
