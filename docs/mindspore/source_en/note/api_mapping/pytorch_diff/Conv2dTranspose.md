# Function Differences with torch.nn.ConvTranspose2d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Conv2dTranspose.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.ConvTranspose2d

```text
class torch.nn.ConvTranspose2d(
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

For more information, see [torch.nn.ConvTranspose2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConvTranspose2d.html).

## mindspore.nn.Conv2dTranspose

```text
class mindspore.nn.Conv2dTranspose(
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

For more information, see [mindspore.nn.Conv2dTranspose](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Conv2dTranspose.html).

## Differences

PyTorch: Computing a two-dimensional transposed convolution can be thought of as Conv2d solving for the gradient of the input, also known as deconvolution (which is not actually true deconvolution). The input shape is usually $(N,C_{in},H_{in},W_{in})$, where $N$ is the batch size, $C$ is the spatial dimension, and $H_{in},W_{in}$ is the height and width, respectively. The output shape is $(N,C_{out},H_{out},W_{out})$, where the height and width are
$H_{out}=(H_{in}−1)×stride[0]−2×padding[0]+dilation[0]×(kernel\underline{ }size[0]−1)+output\underline{ }padding[0]+1$ and
$W_{out}=(W_{in}−1)×stride[1]−2×padding[1]+dilation[1]×(kernel\underline{ }size[1]−1)+output\underline{ }padding[1]+1$

MindSpore: MindSpore API implements basically the same function as PyTorch, but with the new "pad_mode" parameter. When "pad_mode" = "pad", it is the same as the PyTorch default, and the weight_init and bias_init parameters can be used to configure the initialization method. In addition, torch.nn.ConvTranspose2d has an output_padding parameter that functions as a one-sided zero-padding (right and bottom side) of the feature map after deconvolution, which is currently not available in mindspore.nn.Conv2dTranspose and can be used for the output result by using [nn.Pad] (https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Pad.html#mindspore.nn.Pad) for the output to perform the complementary dimensioning instead.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | in_channels    | in_channels   | -  |
|      | Parameter 2 | out_channels    | out_channels      | - |
|      | Parameter 3 | kernel_size  |  kernel_size          | -  |
|      | Parameter 4 | stride   |  stride           | -  |
|      | Parameter 5 | padding   |  padding           |Consistent Function. PyTorch can only pad the same values on each side of the two dimensions, which can be tuples of length 2. MindSpore allows you to set the number of fills for the top, bottom, left and right, respectively, for a tuple of length 4 |
|      | Parameter 6 | output_padding   |     -       | One-sided complementary zeros (right and bottom side) are applied to the feature map after deconvolution, which is usually used to adjust output shapes if stride > 1. For example, the padding is usually set to (kernel_size - 1)/2, and setting output_padding = (stride - 1) ensures that input shapes/output shapes = stride. MindSpore does not have this parameter.|
|      | Parameter 7 | groups   |  group           | Same function, different parameter names  |
|      | Parameter 8 | bias   |  has_bias           | PyTorch defaults to True, and MindSpore defaults to False|
|      | Parameter 9 | dilation   |  dilation        | -  |
|      | Parameter 10 |  padding_mode   |    -     | Numeric padding mode, only support "zeros" that is, pad 0. MindSpore does not have this parameter, but the default padding is 0|
|      | Parameter 11 |   -  | pad_mode       | Specify the padding mode. Optional values are "same", "valid", "pad". In "same" and "valid" mode, padding must be set to 0, default is "same". PyTorch does not have this parameter.|
|      | Parameter 12 |   -  | weight_init        | The initialization method for the weight parameter. Can be Tensor, str, Initializer or numbers.Number. When using str, the values of "TruncatedNormal", "Normal", "Uniform", "HeUniform" and "XavierUniform" distributions and the constants "One" and "Zero" distributions can be selected. The default is "normal". PyTorch does not have this parameter. |
|      | Parameter 13 |  -   | bias_init        | The initialization method for the bias parameter. The initialization method is the same as "weight_init", and the default is "zeros". PyTorch does not have this parameter.|
| Input |Single input  | input | x  | Same function, different parameter names |

### Code Example 1

> Both APIs implement two-dimensional transposed convolutional operations, which need to be instantiated first. Padding values for height and width in PyTorch are the same in the same direction, e.g., padding set to (2,4) means 2 rows and 4 columns of 0 on both sides of the height and width respectively, which corresponds to setting pad_mode to "pad" in MindSpore. Padding is set to (2,2,4,4). PyTorch uses net.weight.data = torch.ones() to initialize the weights to 1, with a shape of $(in\underline{ }channels,\frac {out\underline{ }channels}{groups} , kernel\ underline{ }size[0], kernel\underline{ }size[1])$. MindSpore directly sets the parameter weight_init = "ones".

```python
# PyTorch
import torch
from torch import tensor
import torch.nn as nn
import numpy as np

k = 4
x_ = np.ones([1, 3, 16, 50])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose2d(3, 64, kernel_size=k, stride=1, padding=(2, 4), output_padding=0, bias=False)
net.weight.data = torch.ones(3, 64, k, k)
output = net(x).detach().numpy()
print(output.shape)
# (1, 64, 15, 45)

# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 4
x_ = np.ones([1, 3, 16, 50])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv2dTranspose(3, 64, kernel_size=k, weight_init='ones', pad_mode='pad', padding=(2, 2, 4, 4))
output = net(x)
print(output.shape)
# (1, 64, 15, 45)
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
x_ = np.ones([1, 3, 16, 50])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose2d(3, 64, kernel_size=k, stride=s, padding=(k-1)//2, output_padding=s-1, bias=False)
net.weight.data = torch.ones(3, 64, k, k)
output = net(x).detach().numpy()
print(output.shape)
# (1, 64, 48, 150)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 16, 50])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv2dTranspose(3, 64, kernel_size=k, stride=s, weight_init='ones', pad_mode='same', padding=0)
output = net(x)
print(output.shape)
# (1, 64, 48, 150)
```

### Code Example 3

> If no padding is done on the original image, a part of the data may be discarded in the case of stride>1. Set padding and output_padding to 0 in PyTorch, and set pad_mode = "valid" in MindSpore, while padding = 0.

```python
# PyTorch
import torch
from torch import tensor
import torch.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 16, 50])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose2d(3, 64, kernel_size=k, stride=s, padding=0, output_padding=0, bias=False)
net.weight.data = torch.ones(3, 64, k, k)
output = net(x).detach().numpy()
print(output.shape)
# (1, 64, 50, 152)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 16, 50])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv2dTranspose(3, 64, kernel_size=k, stride=s, weight_init='ones', pad_mode='valid', padding=0)
output = net(x)
print(output.shape)
# (1, 64, 50, 152)
```

### Code Example 4

> The following example implements the deconvolution of the input tensor and outputs the size of the deconvolved feature map, where PyTorch can set the value of output_padding to make up for the lack of stride greater than 1 by adding right-side and lower-side dimensionality to the output image after deconvolution. MindSpore does not support output_padding parameter for now, and you need to use [nn.Pad](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Pad.html#mindspore.nn.Pad) again for the output result for one-sided dimensioning.

```python
# PyTorch
import torch
import torch.nn as nn
import numpy as np

m = nn.ConvTranspose2d(in_channels=3, out_channels=32,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       output_padding=1,
                       bias=False)
input = torch.tensor(np.ones([1, 3, 48, 48]), dtype=torch.float32)
output = m(input).detach().numpy()
print(output.shape)
#(1, 32, 96, 96)

# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np
input = ms.Tensor(np.ones([1, 3, 48, 48]), dtype=ms.float32)
m = nn.Conv2dTranspose(in_channels=3,
                       out_channels=32,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       pad_mode="pad",
                       has_bias=False)
output = m(input)
pad = nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1)), mode="CONSTANT")
output = pad(output)
print(output.shape)
#(1, 32, 96, 96)
```
