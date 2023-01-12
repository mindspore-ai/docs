# # Function Differences with torch.nn.ConvTranspose3d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Conv3dTranspose.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.ConvTranspose3d

```text
class torch.nn.ConvTranspose3d(
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

For more information, see [torch.nn.ConvTranspose3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConvTranspose3d.html).

## mindspore.nn.Conv3dTranspose

```text
class mindspore.nn.Conv3dTranspose(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    pad_mode='same',
    padding=0,
    dilation=1,
    group=1,
    output_padding=0,
    has_bias=False,
    weight_init='normal',
    bias_init='zeros',
    data_format='NCDHW'
)(x) -> Tensor
```

For more information, see [mindspore.nn.Conv3dTranspose](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Conv3dTranspose.html).

## Differences

PyTorch: Computing the 3D transposed convolution can be thought of as Conv3d solving for the gradient of the input, also known as deconvolution (which is not really deconvolution). The input shape is usually $(N,C_{in},D_{in},H_{in},W_{in})$, where $N$ is the batch size, $C$ is the spatial dimension, and $D_{in},H_{in},W_{in}$ are the depth, height and width of the feature layer, respectively. The output shape is $(N,C_{out},D_{out},H_{out},W_{out})$, which is calculated as follows.
$D_{out}=(D_{in}-1)×stride[0]-2×padding[0]+dilation[0]×(kernel\underline{ }size[0]-1)+output\ underline{ }padding[0]+1$
$H_{out}=(H_{in}−1)×stride[1]−2×padding[1]+dilation[1]×(kernel\underline{ }size[1]−1)+output\underline{ }padding[1]+1$
$W_{out}=(W_{in}−1)×stride[2]−2×padding[2]+dilation[2]×(kernel\underline{ }size[2]−1)+output\underline{ }padding[2]+1$

MindSpore: MindSpore API implements essentially the same function as PyTorch, but with the new "pad_mode" parameter. When "pad_mode" = "pad", it is the same as the PyTorch default, and the weight_init and bias_init parameters can be used to configure the initialization method.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | in_channels    | in_channels          | -  |
|      | Parameter 2 | out_channels    | out_channels      | - |
|      | Parameter 3 | kernel_size  |  kernel_size          | -  |
|      | Parameter 4 | stride   |  stride           | -  |
|      | Parameter 5 | padding   |  padding           | The function is the same, PyTorch can only pad the same values on each side of the three dimensions, which can be tuples of length 3. MindSpore can set the number of padding for the front, tail, top, bottom, left and right respectively, and can be a tuples of length 6.  |
|      | Parameter 6 | output_padding   |  output_padding      | -  |
|      | Parameter 7 | groups   |  group           | Same function, different parameter names  |
|      | Parameter 8 | bias   |  has_bias           | PyTorch defaults to True, and MindSpore defaults to False  |
|      | Parameter 9 | dilation   |  dilation     | -  |
|      | Parameter 10 |  padding_mode   |   -      | Numeric padding mode, only support "zeros" that is, padding 0. MindSpore does not have this parameter, but the default padding 0 |
|      | Parameter 11 |  -   | pad_mode       | Specify the padding mode. Optional values are "same", "valid", "pad", in "same" and "valid" mode. padding must be set to 0, and default is "same". |
|      | Parameter 12 |    - | weight_init        | The initialization method for the weight parameter. Can be Tensor, str, Initializer or numbers.Number. When using str, the values of "TruncatedNormal", "Normal", "Uniform", "HeUniform" and "XavierUniform" distributions and the constants "One" and "Zero" distributions can be selected. The default is "normal". |
|     | Parameter 13 | -  | bias_init  | The initialization method for the bias parameter. Optional parameter is the same as "weight_init", and default is "zeros". |
|      | Parameter 14 |   -   |   data_format  |Optional value for the data format. Currently only "NCDHW" is supported, in the same order as the default in PyTorch.|
| | Parameter 15 | input | x | Interface input, same function, only different parameter names |

### Code Example 1

> Both APIs implement 3D transposed convolutional operations and need to be instantiated first when used. To make the output the same width as the input after dividing stride, PyTorch sets output_padding = stride - 1 and padding to (kernel_size - 1)/2. MindSpore sets pad_mode = "same" and padding = 0.

```python
# PyTorch
import torch
from torch import tensor
import torch.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 4, 9, 16])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose3d(3, 32, kernel_size=k, stride=s, padding=(k-1)//2, output_padding=s-1, bias=False)
net.weight.data = torch.ones(3, 32, k, k, k)
output = net(x).detach().numpy()
print(output.shape)
# (1, 32, 12, 27, 48)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 4, 9, 16])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv3dTranspose(3, 32, kernel_size=k, stride=s, weight_init='ones', pad_mode='same')
output = net(x)
print(output.shape)
# (1, 32, 12, 27, 48)
```

### Code Example 2

> Both APIs implement 3D transposed convolutional operations and need to be instantiated first when used. If you do not do any padding on the original image, you may discard part of the data if stride > 1. Set padding and output_padding to 0 in PyTorch, and set pad_mode = "valid" in MindSpore, while padding = 0.

```python
# PyTorch
import torch
from torch import tensor
import torch.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 4, 9, 16])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose3d(3, 32, kernel_size=k, stride=s, bias=False)
net.weight.data = torch.ones(3, 32, k, k, k)
output = net(x).detach().numpy()
print(output.shape)
# (1, 32, 14, 29, 50)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 4, 9, 16])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv3dTranspose(3, 32, kernel_size=k, stride=s, weight_init='ones', pad_mode='valid')
output = net(x)
print(output.shape)
# (1, 32, 14, 29, 50)
```

