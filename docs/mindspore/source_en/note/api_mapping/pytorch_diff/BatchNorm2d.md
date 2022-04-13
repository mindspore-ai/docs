# Function Differences with torch.nn.BatchNorm2d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/BatchNorm2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.BatchNorm2d

```python
class torch.nn.BatchNorm2d(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=True,
    track_running_stats=True
)
```

For more information, see [torch.nn.BatchNorm2d](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.BatchNorm2d).

## mindspore.nn.BatchNorm2d

```python
class mindspore.nn.BatchNorm2d(
    num_features,
    eps=1e-05,
    momentum=0.9,
    affine=True,
    gamma_init="ones",
    beta_init="zeros",
    moving_mean_init="zeros",
    moving_var_init="ones",
    use_batch_statistics=None,
    data_format="NCHW")
)
```

For more information, see [mindspore.nn.BatchNorm2d](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.BatchNorm2d.html#mindspore.nn.BatchNorm2d).

## Differences

PyTorch：The default value of the momentum parameter used for running_mean and running_var calculation is 0.1.

MindSpore：The default value of the momentum parameter is 0.9, and the momentum relationship with Pytorch is 1-momentum, that is, when Pytorch’s momentum value is 0.2, MindSpore’s momemtum should be 0.8. Parameter beta, gamma, moving_mean and moving_variance correspond to Pytorch's bias, weight, running_mean and running_var parameters respectively.

## Code Example

```python
# The following implements BatchNorm2d with MindSpore.
import numpy as np
import torch
import mindspore.nn as nn
from mindspore import Tensor

net = nn.BatchNorm2d(num_features=2, momentum=0.8)
x = Tensor(np.array([[[[1, 2], [1, 2]], [[3, 4], [3, 4]]]]).astype(np.float32))
output = net(x)
print(output)
# Out:
# [[[[0.999995   1.99999]
#    [0.999995   1.99999]]
#
#   [[2.999985   3.99998]
#    [2.999985   3.99998]]]]


# The following implements BatchNorm2d with torch.
input_x = torch.tensor(np.array([[[[1, 2], [1, 2]], [[3, 4], [3, 4]]]]).astype(np.float32))
m = torch.nn.BatchNorm2d(2, momentum=0.2)
output = m(input_x)
print(output)
# Out:
# tensor([[[[-1.0000,  1.0000],
#           [-1.0000,  1.0000]],
#
#          [[-1.0000,  1.0000],
#           [-1.0000,  1.0000]]]], grad_fn=<NativeBatchNormBackward>)
```
