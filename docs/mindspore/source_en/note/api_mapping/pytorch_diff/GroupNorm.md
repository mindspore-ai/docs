# Function Differences with torch.nn.GroupNorm

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/GroupNorm.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.GroupNorm

```text
class torch.nn.GroupNorm(
    num_groups,
    num_channels,
    eps=1e-05,
    affine=True
)(input) -> Tensor
```

For more information, see [torch.nn.GroupNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.GroupNorm.html).

## mindspore.nn.GroupNorm

```text
class mindspore.nn.GroupNorm(
    num_groups,
    num_channels,
    eps=1e-05,
    affine=True,
    gamma_init='ones',
    beta_init='zeros'
)(x) -> Tensor
```

For more information, see [mindspore.nn.GroupNorm](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.GroupNorm.html).

## Differences

PyTorch: Group normalization is performed on the mini-batch input by dividing the channels into groups and then calculating the mean and variance within each group for normalization.

MindSpore: MindSpore API implements basically the same function as PyTorch. MindSpore can also perform additional initialization of the radiating parameters that need to be learned.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1 | num_groups | num_groups   | - |
|      | Parameter 2 | num_channels | num_channels | - |
|      | Parameter 3 | eps          | eps          | -|
|      | Parameter 4 | affine       | affine       | -|
|      | Parameter 5 | -            | gamma_init   | Initialize the radial transform parameter gamma used for learning in the formula. The default is 'ones', while PyTorch cannot be set additionally, can only be 'ones'. |
|      | Parameter 6 | -           | beta_init    | Initialize the radial transform parameter beta used for learning in the formula. The default is 'ones', while PyTorch cannot be set additionally, can only be 'ones'. |
| Input | Single input | input        | x            | Interface input, same function, different parameter names          |

## Code Example 1

> MindSpore API basically implements the same function as TensorFlow, and MindSpore can also perform additional initialization of the two learning parameters.

```python
# PyTorch
import torch
import numpy as np
from torch import tensor, nn

x = tensor(np.ones([1, 2, 4, 4], np.float32))
net = nn.GroupNorm(2, 2)
output = net(x).detach().numpy()
print(output)
# [[[[0. 0. 0. 0.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]]
#
#   [[0. 0. 0. 0.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]]]]

# MindSpore
import mindspore as ms
import numpy as np
from mindspore import Tensor, nn

x = Tensor(np.ones([1, 2, 4, 4], np.float32))
net = nn.GroupNorm(2, 2)
output = net(x)
print(output)
# [[[[0. 0. 0. 0.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]]
#
#   [[0. 0. 0. 0.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]]]]
```
