# Differences with torch.nn.BatchNorm2d

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/BatchNorm2d.md)

## torch.nn.BatchNorm2d

```text
class torch.nn.BatchNorm2d(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=True,
    track_running_stats=True
)(input) -> Tensor
```

For more information, see [torch.nn.BatchNorm2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.BatchNorm2d.html).

## mindspore.nn.BatchNorm2d

```text
class mindspore.nn.BatchNorm2d(
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

For more information, see [mindspore.nn.BatchNorm2d](https://www.mindspore.cn/docs/en/r2.3/api_python/nn/mindspore.nn.BatchNorm2d.html).

## Differences

PyTorch: Apply batch normalization on four-dimensional inputs (two-dimensional input with additional mini-batch and channel channels) to avoid internal covariate bias.

MindSpore: The function of this API is basically the same as that of PyTorch, with two typical differences. The default value of the momentum parameter in MindSpore is 0.9, and the momentum conversion relationship with PyTorch is 1-momentum. The behavior of the default value is the same as that of PyTorch. The parameter update strategy during training and inference is different from that of PyTorch.

| Categories | Subcategories   |PyTorch | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | num_features | num_features | - |
| | Parameter 2 | eps | eps | - |
| | Parameter 3 | momentum | momentum | The function is the same, but the default value in PyTorch is 0.1, and in MindSpore is 0.9, the conversion relationship with PyTorch's momentum is 1-momentum, and the default value behavior is the same as PyTorch |
| | Parameter 4 | affine | affine | - |
| | Parameter 5 | track_running_stats | use_batch_statistics | The function is the same, and different values correspond to different default methods. |
| | Parameter 6 | - | gamma_init |The initialization method of the γ parameter, default value: "ones". PyTorch does not have this parameter. |
| | Parameter 7 | - | beta_init |The initialization method of the β parameter, default value: "zeros". PyTorch does not have this parameter. |
| | Parameter 8 | - | moving_mean_init |Initialization method of dynamic average, default value: "zeros". PyTorch does not have this parameter. |
| | Parameter 9 | - | moving_var_init |Initialization method of dynamic variance, default value: "ones". PyTorch does not have this parameter. |
| | Parameter 10 | - | data_format |MindSpore can specify the input data format as "NHWC" or "NCHW", default value: "NCHW". PyTorch does not have this parameter|
| Input | Single input | input | x | Same function, different parameter names |

BatchNorm is a special regularization method in the CV field. It has different computation processes during training and inference and is usually controlled by operator attributes. BatchNorm of MindSpore and PyTorch uses two different parameter groups at this point.

- Difference 1

`torch.nn.BatchNorm2d` status under different parameters

|training|track_running_stats|Status|
|:----|:----|:--------------------------------------|
|True|True|Expected training status. `running_mean` and `running_var` trace the statistical features of the batch in the entire training process. Each group of input data is normalized based on the mean and var statistical features of the current batch, and then `running_mean` and `running_var` are updated.|
|True|False|Each group of input data is normalized based on the statistics feature of the current batch, but the `running_mean` and `running_var` parameters do not exist.|
|False|True|Expected inference status. The BN uses `running_mean` and `running_var` for normalization and does not update them.|
|False|False|The effect is the same as that of the second status. The only difference is that this is the inference status and does not learn the weight and bias parameters. Generally, this status is not used.|

`mindspore.nn.BatchNorm2d` status under different parameters

|use_batch_statistics|Status|
|:----|:--------------------------------------|
|True|Expected training status. `moving_mean` and `moving_var` trace the statistical features of the batch in the entire training process. Each group of input data is normalized based on the mean and var statistical features of the current batch, and then `moving_mean` and `moving_var` are updated.
|Fasle|Expected inference status. The BN uses `moving_mean` and `moving_var` for normalization and does not update them.
|None|`use_batch_statistics` is automatically set. For training, set `use_batch_statistics` to `True`. For inference, `set use_batch_statistics` to `False`.

Compared with `torch.nn.BatchNorm2d`, `mindspore.nn.BatchNorm2d` does not have two redundant states and retains only the most commonly used training and inference states.

- Difference 2

In PyTorch, the network is in training mode by default, while in MindSpore, it is in inference mode by default (`is_training` is False). You need to use the `net.set_train()` method in MindSpore to switch the network to training mode. In this case, the parameters `mean` and `variance` are calculated during the training. Otherwise, in inference mode, the parameters are loaded from the checkpoint.

- Difference 3

The meaning of the momentum parameter of the BatchNorm series operators in MindSpore is opposite to that in PyTorch. The relationship is as follows:

$$momentum_{pytorch} = 1 - momentum_{mindspore}$$

### Code Example

> In PyTorch, the value after 1-momentum is equal to the momentum of MindSpore, both trained by using mini-batch data and learning parameters.

```python
# PyTorch
from torch import nn, tensor
import numpy as np

m = nn.BatchNorm2d(num_features=3, momentum=0.1)
input_py = tensor(np.array([[[[0.1, 0.2], [0.3, 0.4]],
                          [[0.5, 0.6], [0.7, 0.8]],
                          [[0.9, 1], [1.1, 1.2]]]]).astype(np.float32))
output = m(input_py)
print(output.detach().numpy())
# [[[[-1.3411044  -0.44703478]
#    [ 0.4470349   1.3411044 ]]
#
#   [[-1.3411043  -0.44703442]
#    [ 0.44703496  1.3411049 ]]
#
#   [[-1.3411039  -0.44703427]
#    [ 0.44703534  1.341105  ]]]]

# MindSpore
from mindspore import Tensor, nn
import numpy as np

m = nn.BatchNorm2d(num_features=3, momentum=0.9)
m.set_train()
#  BatchNorm2d<num_features=3, eps=1e-05, momentum=0.9, gamma=Parameter (name=gamma, shape=(3,), dtype=Float32, requires_grad=True), beta=Parameter (name=beta, shape=(3,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=mean, shape=(3,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=variance, shape=(3,), dtype=Float32, requires_grad=False)>

input_ms = Tensor(np.array([[[[0.1, 0.2], [0.3, 0.4]],
                          [[0.5, 0.6], [0.7, 0.8]],
                          [[0.9, 1], [1.1, 1.2]]]]).astype(np.float32))
output = m(input_ms)
print(output)
# [[[[-1.3411045  -0.4470348 ]
#    [ 0.44703496  1.3411045 ]]
#
#   [[-1.341105   -0.4470351 ]
#    [ 0.44703424  1.3411041 ]]
#
#   [[-1.3411034  -0.44703388]
#    [ 0.44703573  1.3411053 ]]]]
```
