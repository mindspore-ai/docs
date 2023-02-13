# # Function Differences with torch.nn.Conv3d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Conv3d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Conv3d

```text
class torch.nn.Conv3d(
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

For more information, see [torch.nn.Conv3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Conv3d.html).

## mindspore.nn.Conv3d

```text
class mindspore.nn.Conv3d(
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
    data_format='NCDHW'
)(x) -> Tensor
```

For more information, see [mindspore.nn.Conv3d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Conv3d.html).

## Differences

PyTorch: To compute a 3D convolution on the input Tensor, the output value with input size $\left(N, C_{i n}, D, H, W\right)$ and output size $\left(N, C_{\text {out }}, D_{\text {out }}, H_{\text {out }}, W_{\text {out }}\right)$ can usually be described as:

$$
\operatorname{out}\left(N_{i}, C_{\text {out }_{j}}\right)=\operatorname{bias}\left(C_{\text {out }_{j}}\right)+\sum_{k=0}^{C_{i n}-1} \text { weight }\left(C_{\text {out }_{j}}, k\right) \star \operatorname{input}\left(N_{i}, k\right)
$$

where $\star$ is the 3d cross-correlation operator, $N$ is the batch size, $C$ is the number of channels, and $D$, $H$, and $W$ are the depth, height, and width of the feature layer, respectively.

MindSpore: It is basically the same as the functions implemented by PyTorch, but there are bias differences and filling differences.

1. Offset difference: MindSpore does not add offset parameters by default, contrary to PyTorch.
2. Fill difference: MindSpore fills the input by default, while PyTorch does not fill by default. At the same time, MindSpore filling mode options and behavior are different from PyTorch. The specific differences in filling behavior are as follows.

### Filling Behavior Difference

1. The parameter "padding_mode" of PyTorch can be selected as 'zero','reflect', 'replicate', and 'circular'. The default is 'zero'. The parameter "padding" can be selected as 'int', 'tuple of ints', 'valid', and 'same'. The default is 0. The four padding mode of the parameter "padding_mode" is consistent with that of the "torch.nn.functional.pad" interface. After setting, the convolution input will be filled according to the specified filling mode, as follows:

    - zero: constant fill (zero fill by default).

    - reflect: reflection fill.

    - replicate: Edge replication fill.

    - circular: circular fill.

    After the filling method is determined by "padding _mode", the "padding" parameter is used to control the number and position of filling. For "Conv2d", when "padding" is specified as 'int', "padding" times will be filled in the top, bottom, left, right，front and back sides of the input (if the default value is 0, it means no filling). When "padding" is specified as tuple, the specified number of filling will be filled in the top, bottom, left, and right front and back sides according to the input of tuple. When "padding" is set to the 'valid' mode, it will not be filled, but will only be convolved within the range of the feature map. When "padding" is set to the 'same' mode, if the number of elements requiring "padding" is even, padding elements are evenly distributed on the top, bottom, left, and right of the feature map. If the number of elements requiring "padding" is odd, PyTorch will fill the left and upper sides of the feature map first.

2. The parameter "pad_mode" of MindSpore can be selected as 'same', 'valid', and 'pad'. The parameter "padding" can only be input as "int". The detailed meaning of the filling parameter is as follows:

    When "pad_mode" is set to 'pad', "MindSpore" can set the "padding" parameter to a positive integer greater than or equal to 0. Zero filling will be carried out "padding" times around the input(if it is the default value of 0, it will not fill). When "pad_mode" is the other two modes, the "padding" parameter must be set to 0 only. When "pad_mode" is set to 'valid' mode, it will not fill, and the convolution will only be carried out within the range of the feature map. If pad_mode is set to 'same' mode, when the padding element is an even number, padding elements are evenly distributed on the top, bottom, left, right of the feature map. If the number of elements requiring "padding" is odd, MindSpore will preferentially fill the right and lower sides of the feature map (different from PyTorch, similar to TensorFlow).

    Therefore, if "MindSpore" wants to achieve the same filling mode as "PyTorch", it needs to manually fill the input with "nn.Pad" or "ops.pad" interface.

### Weight Initialization Difference

1. mindspore.nn.Conv2d (weight：$\mathcal{N}(0, 1)$，bias：zeros)
2. torch.nn.Conv2d (weight：$\mathcal{U} (-\sqrt{k},\sqrt{k} )$，bias：$\mathcal{U} (-\sqrt{k},\sqrt{k} )$)

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | in_channels | in_channels |- |
| | Parameter 2 | out_channels | out_channels |- |
| | Parameter 3 | kernel_size | kernel_size |- |
| | Parameter 4 | stride | stride |- |
| | Parameter 5 | padding | padding |-|
| | Parameter 6 | dilation | dilation |-|
| | Parameter 7 | groups | group |Same function, different parameter names|
| | Parameter 8 | bias | has_bias |Same function, different parameter names, different default values |
| | Parameter 9 | padding_mode | pad_mode |Refer to the above for specific differences|
| | Parameter 10 | - | weight_init |The initialization method for the weight parameter. Refer to the above for specific differences|
| | Parameter 11 | - | bias_init |Initialization method for the bias parameter. Refer to the above for specific differences|
| | Parameter 12 | - | data_format |Input data format, which is not available for PyTorch |
| Input | Single input | input  | x | Same function, different parameter names |

### Code Example 1

> The default value of PyTorch's parameter bias is True, which means that bias parameters are added by default, while the default value of MindSpore's parameter has_bias is False, which means that bias functions are not added by default. If you need to add a bias parameter, you need to set the value of has_bias to True.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((16, 3, 10, 32, 32))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv3d(3, 32, (4, 3, 3))
output = net(x).detach().numpy().shape
print(output)
# (16, 32, 7, 30, 30)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((16, 3, 10, 32, 32))
x = Tensor(x_, mindspore.float32)
net = nn.Conv3d(3, 32, (4, 3, 3), has_bias=True, pad_mode='valid')
output = net(x).shape
print(output)
# (16, 32, 7, 30, 30)
```

### Code Example 2

> PyTorch's parameter padding_mode is 'zero', which means zero padding for the input, while implementing zero padding in MindSpore requires setting the parameter pad_mode to 'pad'.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((16, 3, 10, 32, 32))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv3d(3, 32, (4, 3, 3), padding=1, padding_mode='zeros')
output = net(x).detach().numpy().shape
print(output)
# (16, 32, 9, 32, 32)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((16, 3, 10, 32, 32))
x = Tensor(x_, mindspore.float32)
net = nn.Conv3d(3, 32, (4, 3, 3), padding=1, pad_mode='pad')
output = net(x).shape
print(output)
# (16, 32, 9, 32, 32)
```

### Code Example 3

> PyTorch does not pad the input by default, while MindSpore requires it by default. If you don't pad the input, you need to set pad_mode to 'valid'.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((16, 3, 10, 32, 32))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv3d(3, 32, (4, 3, 3))
output = net(x).detach().numpy().shape
print(output)
# (16, 32, 7, 30, 30)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((16, 3, 10, 32, 32))
x = Tensor(x_, mindspore.float32)
net = nn.Conv3d(3, 32, (4, 3, 3), pad_mode='valid')
output = net(x).shape
print(output)
# (16, 32, 7, 30, 30)
```
