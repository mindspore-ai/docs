# Differences with torch.nn.BatchNorm1d

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/BatchNorm1d.md)

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

For more information, see [mindspore.nn.BatchNorm1d](https://mindspore.cn/docs/en/r2.3/api_python/nn/mindspore.nn.BatchNorm1d.html).

## Differences

PyTorch：Batch normalization of the input 2D or 3D data.

MindSpore：The implementation function of the API in MindSpore is basically the same as that of PyTorch. The default value of the momentum parameter in MindSpore is 0.9, and the momentum conversion relationship with PyTorch is 1-momentum. The behavior of the default value is the same as that of PyTorch. The parameter update strategy during training and inference is different from that of PyTorch.

| Categories | Subcategories   | PyTorch             | MindSpore            | Differences                                                         |
| ---- | ------ | ------------------- | -------------------- | ------------------------------------------------------------ |
| Parameters | Parameter 1  | num_features        | num_features         | -                                                            |
|      | Parameter 2  | eps                 | eps                  | -                                                            |
|      | Parameter 3  | momentum            | momentum             | The function is the same, but the default value in PyTorch is 0.1, and in MindSpore is 0.9. The conversion relationship with PyTorch's momentum is 1-momentum, and the default value behavior is the same as PyTorch         |
|      | Parameter 4  | affine              | affine               | -                                                            |
|      | Parameter 5  | track_running_stats              | use_batch_statistics |    The function is the same, and different values correspond to different default methods.      |
|      | Parameter 6  | -                   | gamma_init           |    PyTorch does not have this parameter, while MindSpore can initialize the value of the parameter gamma    |
|      | Parameter 7  | -                   | beta_init            |    PyTorch does not have this parameter, while MindSpore can initialize the value of the parameter beta     |
|      | Parameter 8  | -                   | moving_mean_init     |    PyTorch does not have this parameter, while MindSpore can initialize the value of the parameter moving_mean   |
|      | Parameter 9  | -                   | moving_var_init      |    PyTorch does not have this parameter, while MindSpore can initialize the value of the parameter moving_var     |
|      | Parameter 10  | -                   | data_format      |    PyTorch does not have this parameter    |
| Input | Single input | input    | x     | Same function, different parameter names  |

The detailed differences are as follows:
BatchNorm is a special regularization method in the CV field. It has different computation processes during training and inference and is usually controlled by operator attributes. BatchNorm of MindSpore and PyTorch uses two different parameter groups at this point.

- Difference 1

  `torch.nn.BatchNorm1d` status under different parameters

  |training|track_running_stats|Status|
  |:----|:----|:--------------------------------------|
  |True|True|Expected training status. `running_mean` and `running_var` trace the statistical features of the batch in the entire training process. Each group of input data is normalized based on the mean and var statistical features of the current batch, and then `running_mean` and `running_var` are updated.|
  |True|False|Each group of input data is normalized based on the statistics feature of the current batch, but the `running_mean` and `running_var` parameters do not exist.|
  |False|True|Expected inference status. The BN uses `running_mean` and `running_var` for normalization and does not update them.|
  |False|False|The effect is the same as that of the second status. The only difference is that this is the inference status and does not learn the weight and bias parameters. Generally, this status is not used.|

  `mindspore.nn.BatchNorm1d` status under different parameters

  |use_batch_statistics|Status|
  |:----|:--------------------------------------|
  |True|Expected training status. `moving_mean` and `moving_var` trace the statistical features of the batch in the entire training process. Each group of input data is normalized based on the mean and var statistical features of the current batch, and then `moving_mean` and `moving_var` are updated.
  |Fasle|Expected inference status. The BN uses `moving_mean` and `moving_var` for normalization and does not update them.
  |None|`use_batch_statistics` is automatically set. For training, set `use_batch_statistics` to `True`. For inference, set `use_batch_statistics` to `False`.

  Compared with `torch.nn.BatchNorm1d`, `mindspore.nn.BatchNorm1d` does not have two redundant states and retains only the most commonly used training and inference states.

- Difference 2

  In PyTorch, the network is in training mode by default, while in MindSpore, it is in inference mode by default (`is_training` is False). You need to use the `net.set_train()` method in MindSpore to switch the network to training mode. In this case, the parameters `mean` and `variance` are calculated during the training. Otherwise, in inference mode, the parameters are loaded from the checkpoint.

- Difference 3

  The meaning of the momentum parameter of the BatchNorm series operators in MindSpore is opposite to that in PyTorch. The relationship is as follows:

  $$momentum_{pytorch} = 1 - momentum_{mindspore}$$

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
# BatchNorm1d<num_features=4, eps=1e-05, momentum=0.9, gamma=Parameter (name=gamma, shape=(4,), dtype=Float32, requires_grad=False), beta=Parameter (name=beta, shape=(4,), dtype=Float32, requires_grad=False), moving_mean=Parameter (name=mean, shape=(4,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=variance, shape=(4,), dtype=Float32, requires_grad=False)>

x = Tensor(np.array([[0.7, 0.5, 0.5, 0.6], [0.5, 0.4, 0.6, 0.9]]).astype(np.float32))
output = net(x)
print(output.asnumpy())
# [[ 0.9995001  0.9980063 -0.998006  -0.9997778]
#  [-0.9995007 -0.9980057  0.998006   0.9997778]]
```
