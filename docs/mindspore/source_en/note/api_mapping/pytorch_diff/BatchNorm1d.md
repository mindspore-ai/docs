# Function Differences with torch.nn.BatchNorm1d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/BatchNorm1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [mindspore.nn.BatchNorm1d](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.BatchNorm1d.html).

## Differences

PyTorch：Batch normalization of the input 2D or 3D data.

MindSpore：The implementation function of the API in MindSpore is basically the same as that of PyTorch. The default value of the momentum parameter is 0.9, and the conversion relationship between the momentum parameter and PyTorch momentum is 1-momentum.

| Categories | Subcategories   | PyTorch             | MindSpore            | Differences                                                         |
| ---- | ------ | ------------------- | -------------------- | ------------------------------------------------------------ |
| Parameters | Parameter 1  | num_features        | num_features         | -                                                            |
|      | Parameter 2  | eps                 | eps                  | -                                                            |
|      | Parameter 3  | momentum            | momentum             | The function is the same, but the default value is 0.1 in PyTorch and 0.9 in MindSpore         |
|      | Parameter 4  | affine              | affine               | -                                                            |
|      | Parameter 5  | track_running_stats              | -               | Not involved                                |
|      | Parameter 6  | input               | x                    | The function is basically the same, but PyTorch allows inputs to be 2D or 3D, while MindSpore allows inputs to be 2D only |
|      | Parameter 7  | -                   | gamma_init           |    PyTorch does not have this parameter, while MindSpore can initialize the value of the parameter gamma    |
|      | Parameter 8  | -                   | beta_init            |    PyTorch does not have this parameter, while MindSpore can initialize the value of the parameter beta     |
|      | Parameter 9  | -                   | moving_mean_init     |    PyTorch does not have this parameter, while MindSpore can initialize the value of the parameter moving_mean   |
|      | Parameter 10  | -                   | moving_var_init      |    PyTorch does not have this parameter, while MindSpore can initialize the value of the parameter moving_var     |
|      | Parameter 11 | -                   | use_batch_statistics |    PyTorch does not have this parameter, and if True in MindSpore, the mean and variance values of the current batch of data are used      |
|      | Parameter 12  | -                   | data_format      |    PyTorch does not have this parameter    |

## Code Example

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
import numpy as np
from torch import nn, tensor

m = nn.BatchNorm1d(4, affine=False, momentum=0.1)
input = tensor(np.array([[0.7, 0.5, 0.5, 0.6], [0.5, 0.4, 0.6, 0.9]]).astype(np.float32))
output = m(input)
print(output.detach().numpy())
# [[ 0.9995001   0.9980063  -0.998006   -0.99977785]
#  [-0.9995007  -0.9980057   0.998006    0.99977785]]

# MindSpore
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor

net = nn.BatchNorm1d(num_features=4, affine=False, momentum=0.9)
net.set_train()
input = Tensor(np.array([[0.7, 0.5, 0.5, 0.6], [0.5, 0.4, 0.6, 0.9]]).astype(np.float32))
output = net(input)
print(output.asnumpy())
# [[ 0.9995001  0.9980063 -0.998006  -0.9997778]
#  [-0.9995007 -0.9980057  0.998006   0.9997778]]
```
