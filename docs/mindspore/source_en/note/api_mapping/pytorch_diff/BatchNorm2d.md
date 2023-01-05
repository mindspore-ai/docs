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
)(input) -> Tensor
```

For more information, see [torch.nn.BatchNorm2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.BatchNorm2d.html).

## mindspore.nn.BatchNorm2d

```python
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

For more information, see [mindspore.nn.BatchNorm2d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.BatchNorm2d.html).

## Differences

PyTorch：Apply batch normalization on four-dimensional inputs (small batches of two-dimensional inputs with additional channel dimensionality) to avoid internal covariate bias.

MindSpore：Implement the same function as PyTorch.

| Categories | Subcategories   |PyTorch | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | input | x | Interface input, same function, only different parameter names |
| | Parameter 2 | num_features | num_features | - |
| | Parameter 3 | eps | eps | - |
| | Parameter 4 | momentum | momentum |Same function, different calculation method |
| | Parameter 5 | affine | affine |- |
| | Parameter 6 | track_running_stats | use_batch_statistics | The function is the same, and different values correspond to different default methods |
| | Parameter 7 | - | gamma_init |The initialization method of the γ parameter, default value: "ones" |
| | Parameter 8 | - | beta_init |The initialization method of the βparameter, default value: "ones" |
| | Parameter 9 | - | moving_mean_init |Initialization method of dynamic average, default value: "ones" |
| | Parameter 10 | - | moving_var_init |Initialization method of dynamic variance, default value: "ones" |
| | Parameter 11 | - | data_format |MindSpore can specify the input data format as "NHWC" or "NCHW", default value: "NCHW", PyTorch does not have this parameter|

## Code Example

> In PyTorch, the value after 1-momentum is equal to the momentum of MindSpore, both trained by using mini-batch data and learning parameters.

```python
# PyTorch
from torch import nn, Tensor
import numpy as np

m = nn.BatchNorm2d(num_features=3, momentum=0.1)
input_x = Tensor(np.array([[[[0.1, 0.2], [0.3, 0.4]],
                          [[0.5, 0.6], [0.7, 0.8]],
                          [[0.9, 1], [1.1, 1.2]]]]).astype(np.float32))
output = m(input_x)
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
input_x = Tensor(np.array([[[[0.1, 0.2], [0.3, 0.4]],
                          [[0.5, 0.6], [0.7, 0.8]],
                          [[0.9, 1], [1.1, 1.2]]]]).astype(np.float32))
output = m(input_x)
print(output)
# [[[[-1.3411045  -0.4470348 ]
#    [ 0.44703496  1.3411045 ]]
#
#   [[-1.341105   -0.4470351 ]
#    [ 0.44703424  1.3411041 ]]
#
#   [[-1.3411034  -0.44703388]
#    [ 0.44703573  1.3411053 ]]]
```
