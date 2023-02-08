# Function Differences with torch.nn.BatchNorm3d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/BatchNorm3d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.BatchNorm3d

```text
class torch.nn.BatchNorm3d(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=True,
    track_running_stats=True
)(input) -> Tensor
```

For more information, see [torch.nn.BatchNorm3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.BatchNorm3d.html).

## mindspore.nn.BatchNorm3d

```text
class mindspore.nn.BatchNorm3d(
    num_features,
    eps=1e-5,
    momentum=0.9,
    affine=True,
    gamma_init='ones',
    beta_init='zeros',
    moving_mean_init='zeros',
    moving_var_init='ones',
    use_batch_statistics=None
)(x) -> Tensor
```

For more information, see [mindspore.nn.BatchNorm3d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.BatchNorm3d.html).

## Differences

PyTorch: Apply batch normalization on five-dimensional inputs (three-dimensional input with additional mini-batch and channel channels) to avoid internal covariate bias.

MindSpore：The function of this API is basically the same as that of PyTorch, with two typical differences. The default value of the momentum parameter in MindSpore is 0.9, and the momentum conversion relationship with PyTorch is 1-momentum. The behavior of the default value is the same as that of PyTorch. The parameter update strategy during training and inference is different from that of PyTorch. For details, please refer to [Differences Between MindSpore and PyTorch - BatchNorm](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/typical_api_comparision.html#nn.BatchNorm2d).

| Categories | Subcategories   |PyTorch | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1| num_features | num_features | - |
| | Parameter 2 | eps | eps | - |
| | Parameter 3 | momentum | momentum | The function is the same, but the default value in PyTorch is 0.1, and in MindSpore is 0.9, the conversion relationship with PyTorch's momentum is 1-momentum, and the default value behavior is the same as PyTorch |
| | Parameter 4 | affine | affine | - |
| | Parameter 5 | track_running_stats | use_batch_statistics | The function is the same, and different values correspond to different default methods. For details, please refer to [Typical differences with PyTorch -nn.BatchNorm](https://www.mindspore.cn/docs/en/master/migration_guide/typical_api_comparision.html#nn.BatchNorm2d) |
| | Parameter 6 | - | gamma_init |The initialization method of the γ parameter, default value: "ones". |
| | Parameter 7 | - | beta_init |The initialization method of the β parameter, default value: "zeros". |
| | Parameter 8 | - | moving_mean_init |Initialization method of dynamic average, default value: "zeros". |
| | Parameter 9 | - | moving_var_init |Initialization method of dynamic variance, default value: "ones". |
| Input | Single input | input | x | Interface input, same function, only different parameter names |

### Code Example

> In PyTorch, the value after 1-momentum is equal to the momentum of MindSpore, both trained by using mini-batch data and learning parameters.

```python
# PyTorch
from torch import nn, tensor
import numpy as np

m = nn.BatchNorm3d(num_features=2, momentum=0.1)
input_x = tensor(np.array([[[[[0.1, 0.2], [0.3, 0.4]]],
                             [[[0.9, 1], [1.1, 1.2]]]]]).astype(np.float32))
output = m(input_x)
print(output.detach().numpy())
# [[[[[-1.3411044  -0.44703478]
#     [ 0.4470349   1.3411044 ]]]
#
#
#   [[[-1.3411034  -0.44703388]
#     [ 0.44703573  1.3411053 ]]]]]

# MindSpore
from mindspore import Tensor, nn
import numpy as np

m = nn.BatchNorm3d(num_features=2, momentum=0.9)
m.set_train()
input_x = Tensor(np.array([[[[[0.1, 0.2], [0.3, 0.4]]],
                             [[[0.9, 1], [1.1, 1.2]]]]]).astype(np.float32))
output = m(input_x)
print(output)
# [[[[[-1.3411044  -0.44703478]
#     [ 0.4470349   1.3411044 ]]]
#
#
#   [[[-1.3411039  -0.44703427]
#     [ 0.44703534  1.341105  ]]]]]
```
