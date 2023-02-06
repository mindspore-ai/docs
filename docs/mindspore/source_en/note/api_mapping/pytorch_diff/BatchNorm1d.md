# Function Differences with torch.nn.BatchNorm1d

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/note/api_mapping/pytorch_diff/BatchNorm1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

## torch.nn.BatchNorm1d

```text
class torch.nn.BatchNorm1d(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=True,
    track_running_stats=True
)(input) -> Tensor
```

For more information, see [torch.nn.BatchNorm1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.BatchNorm1d.html).

## mindspore.nn.BatchNorm1d

```text
class mindspore.nn.BatchNorm1d(
    num_features,
    eps=1e-5,
    momentum=0.9,
    affine=True,
    gamma_init='ones',
    beta_init='zeros',
    moving_mean_init='zeros',
    moving_var_init='ones',
    use_batch_statistics=None,
    data_format='NCHW'
)(x) -> Tensor
```

For more information, see [mindspore.nn.BatchNorm1d](https://mindspore.cn/docs/en/r2.0.0-alpha/api_python/nn/mindspore.nn.BatchNorm1d.html).

## Differences

PyTorch：perform batch normalization on the input 2D or 3D data.

MindSpore：MindSpore API implements the same function as TensorFlow. The default value of the momentum parameter is 0.9, and the momentum relationship with Pytorch is 1-momentum.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1  | num_features        | num_features         | -                                                            |
|      | Parameter 2  | eps                 | eps                  | -                                                            |
|      | Parameter 3  | momentum            | momentum             | Same functionality, but the default value is 0.1 in PyTorch and 0.9 in MindSpore, with PyTorch's momentum conversion relationship of 1-momentum and the same default as PyTorch        |
|      | Parameter 4  | affine              | affine               | -                                                            |
|      | Parameter 5  | track_running_stats              | use_batch_statistics               | Same function. Different values correspond to different default methods. Please refer to [Typical Differences with PyTorch - nn.BatchNorm2d](https://www.mindspore.cn/docs/en/master/migration_guide/typical_api_comparision.html#nn.BatchNorm2d) for detailed differences                               |
|      | Parameter 6  | -                   | gamma_init           |    PyTorch does not have this parameter. MindSpore can initialize the value of the parameter gamma    |
|      | Parameter 7  | -                   | beta_init            |    PyTorch does not have this parameter. MindSpore can initialize the value of the parameter beta     |
|      | Parameter 8  | -                   | moving_mean_init     |    PyTorch does not have this parameter. MindSpore can initialize the value of the parameter moving_mean    |
|      | Parameter 9  | -                   | moving_var_init      |    PyTorch does not have this parameter. MindSpore can initialize the value of the parameter moving_var     |
|      | Parameter 10  | -                   | data_format      |    PyTorch does not have this parameter.    |
| Input | Single input | input               | x                    | Interface input. the function is basically the same, but PyTorch allows input to be 2D or 3D, while input in MindSpore can only be 2D |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
import numpy as np
from torch import nn, tensor

net = nn.BatchNorm1d(4, affine=False, momentum=0.1)
x = tensor(np.array([[0.7, 0.5, 0.5, 0.6], [0.5, 0.4, 0.6, 0.9]]).astype(np.float32))
output = net(x)
print(output.detach().numpy())
# [[ 0.9995001   0.9980063  -0.998006   -0.99977785]
#  [-0.9995007  -0.9980057   0.998006    0.99977785]]

# MindSpore
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor

net = nn.BatchNorm1d(num_features=4, affine=False, momentum=0.9)
net.set_train()
x = Tensor(np.array([[0.7, 0.5, 0.5, 0.6], [0.5, 0.4, 0.6, 0.9]]).astype(np.float32))
output = net(x)
print(output.asnumpy())
# [[ 0.9995001  0.9980063 -0.998006  -0.9997778]
#  [-0.9995007 -0.9980057  0.998006   0.9997778]]
```
