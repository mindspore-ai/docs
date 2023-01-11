# Function Differences with torch.nn.ConvTranspose1d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Conv1dTranspose.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.ConvTranspose1d

```text
class torch.nn.ConvTranspose1d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    bias=True,
    dilation=1,
    padding_mode='zeros'
)(input) -> Tensor
```

For more information, see [torch.nn.ConvTranspose1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConvTranspose1d.html).

## mindspore.nn.Conv1dTranspose

```text
class mindspore.nn.Conv1dTranspose(
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
    bias_init='zeros')
)(x) -> Tensor
```

For more information, see [mindspore.nn.Conv1dTranspose](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Conv1dTranspose.html).

## Differences

PyTorch: Computing a one-dimensional transposed convolution can be thought of as Conv1d solving for the gradient of the input, also known as deconvolution (which is not actually true deconvolution). The input shape is usually $(N,C_{in}, L_{in})$, where $N$ is the batch size, $C$ is the spatial dimension, and $L$ is the length of the sequence. The output shape is $(N,C_{out},L_{out})$, where $L_{out}=(L_{in}-1)×stride-2×padding+dilation×(kernel\_size-1)+output\_padding+1$.

MindSpore: MindSpore API implements basically the same function as PyTorch, but with the new "pad_mode" parameter. When "pad_mode" = "pad", it is the same as the PyTorch default, and the weight_init and bias_init parameters can be used to configure the initialization method.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | in_channels    | in_channels   | -  |
|      | Parameter 2 | out_channels    | out_channels   | - |
|      | Parameter 3 | kernel_size  |  kernel_size       | -  |
|      | Parameter 4 | stride   |  stride        | -  |
|      | Parameter 5 | padding   |  padding      | -  |
|      | Parameter 6 | output_padding   |   -  | Usually used with stride > 1 to adjust output shapes. For example, it is common to set padding to (kernel_size - 1)/2, where setting output_padding = (stride - 1) ensures that input shapes/output shapes = stride. MindSpore does not have this parameter  |
|      | Parameter 7 | groups   |  group           | Same function, different parameter names  |
|      | Parameter 8 | bias   |  has_bias           | PyTorch defaults to True, and MindSpore defaults to False |
|      | Parameter 9 | dilation   |  dilation     | -  |
|      | Parameter 10 |  padding_mode   |   -      | Numeric padding mode, only supports "zeros" i.e. padding 0. MindSpore does not have this parameter, but pads 0 by default|
|      | Parameter 11 |  -   | pad_mode       | Specify the padding mode. Optional values are "same", "valid", "pad". In "same" and "valid" mode, padding must be set to 0, and default is "same"  |
|      | Parameter 12 |   -  | weight_init        | The initialization method for the weight parameter. Can be Tensor, str, Initializer or numbers.Number. When using str, the values of "TruncatedNormal", "Normal", "Uniform", "HeUniform" and "XavierUniform" distributions and the constants "One" and "Zero" distributions can be selected. The default is "normal". |
|      | Parameter 13 |   -  | bias_init        | The initialization method for the bias parameter. The initialization method is the same as "weight_init", and the default is "zeros". |
|      | Parameter 14  | input               | x        | Interface input, same function, only different parameter names |

### Code Example 1

> Both APIs implement one-dimensional transposed convolutional operations and need to be instantiated first when used. When PyTorch sets output_padding to 0 and MindSpore sets pad_mode to "pad", the output width is $L_{out}=(L_{in}-1)×stride-2×padding+dilation×( kernel\_size-1)+1$. The weights are initialized to 1 in PyTorch through net.weight.data = torch.ones(), and MindSpore sets the parameter weight_init = "ones" directly.

```python
# PyTorch
import torch
from torch import tensor
import torch.nn as nn
import numpy as np

k = 4
x_ = np.ones([1, 3, 50])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose1d(3, 64, kernel_size=k, stride=1, padding=0, output_padding=0, bias=False)
net.weight.data = torch.ones(3, 64, k)
output = net(x).detach().numpy()
print(output.shape)
# (1, 64, 53)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 4
x_ = np.ones([1, 3, 50])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv1dTranspose(3, 64, kernel_size=k, weight_init='ones', pad_mode='pad')
output = net(x)
print(output.shape)
# (1, 64, 53)
```

### Code Example 2

> To make the output the same width as the input after dividing stride, PyTorch sets output_padding = stride - 1 and padding to (kernel_size - 1)/2. MindSpore sets pad_mode = "same" and padding = 0.

```python
# PyTorch
import torch
from torch import tensor
import torch.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 50])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose1d(3, 64, kernel_size=k, stride=s, padding=(k-1)//2, output_padding=s-1, bias=False)
net.weight.data = torch.ones(3, 64, k)
output = net(x).detach().numpy()
print(output.shape)
# (1, 64, 150)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 50])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv1dTranspose(3, 64, kernel_size=k, stride=s, weight_init='ones', pad_mode='same', padding=0)
output = net(x)
print(output.shape)
# (1, 64, 150)
```

### Code Example 3

> If no padding is done on the original image, a part of the data may be discarded in the case of stride>1. The output width is $L_{out}=(L_{in}-1)×stride+dilation×(kernel\_size-1)+1$. Set padding and output_padding to 0 in PyTorch and set pad_mode = "valid" in MindSpore.

```python
# PyTorch
import torch
from torch import tensor
import torch.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 50])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose1d(3, 64, kernel_size=k, stride=s, padding=0, output_padding=0, bias=False)
net.weight.data = torch.ones(3, 64, k)
output = net(x).detach().numpy()
print(output.shape)
# (1, 64, 152)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 50])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv1dTranspose(3, 64, kernel_size=k, stride=s, weight_init='ones', pad_mode='valid', padding=0)
output = net(x)
print(output.shape)
# (1, 64, 152)
```
