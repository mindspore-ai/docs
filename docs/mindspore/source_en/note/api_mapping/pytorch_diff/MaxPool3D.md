# Function Differences with torch.nn.MaxPool3D

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/MaxPool3D.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.MaxPool3D

```python
torch.nn.MaxPool3D(
    kernel_size=1,
    stride=None,
    padding=0,
    dilation=1,
    return_indices=False,
    ceil_mode=False
)
```

For more information, see [torch.nn.MaxPool3D](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.MaxPool3d).

## mindspore.ops.MaxPool3D

```python
class mindspore.ops.MaxPool3D(
    kernel_size=1,
    strides=1,
    pad_mode='valid',
    pad_list=0,
    ceil_mode=None,
    data_format='NCDHW'
)(input)
```

For more information, see [mindspore.ops.MaxPool3D](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.MaxPool3D.html#mindspore.ops.MaxPool3D).

## Differences

PyTorch: Supports both 5-dimensional (N, C, Din, Hin, Win) input data and 4-dimensional (C, Din, Hin, Win) input data.

MindSpore: Supports only 5-dimensional (N, C, Din, Hin, Win) input data.

Migration advice: If you need MindSpore MaxPool3D to calculate on 4-dimensional input data, data can be expanded to 5-dimensional using ExpandDims operator and passed into MaxPool3D. You can then use the Squeeze operator to convert the dimension from (1, C, Dout, Hout, Wout) to (C, Dout, Hout, Wout).

## Code Example

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore
net = ops.MaxPool3D((3, 2, 2), strides=2)
x = Tensor(np.ones([20, 16, 50, 44, 31]), mindspore.float32)
output = net(x).shape
print(output)
# Out:
# (20, 16, 24, 22, 15)

# In PyTorch
m = torch.nn.MaxPool3d((3, 2, 2), stride=2)
input = torch.rand(20, 16, 50, 44, 31)
output = m(input).shape
print(output)
# Out：
# torch.Size([20, 16, 24, 22, 15])
```
