# Function Differences with torch.nn.InstanceNorm2d

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/InstanceNorm2d.md)

## torch.nn.InstanceNorm2d

```text
class torch.nn.InstanceNorm2d(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=False,
    track_running_stats=False
)(input) -> Tensor
```

For more information, see [torch.nn.InstanceNorm2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.InstanceNorm2d.html).

## mindspore.nn.InstanceNorm2d

```text
class mindspore.nn.InstanceNorm2d(
    num_features,
    eps=1e-5,
    momentum=0.1,
    affine=True,
    gamma_init='ones',
    beta_init='zeros'
)(x) -> Tensor
```

For more information, see [mindspore.nn.InstanceNorm2d](https://www.mindspore.cn/docs/en/r2.0/api_python/nn/mindspore.nn.InstanceNorm2d.html).

## Differences

PyTorch: Apply normalization within each channel of the four-dimension input (2D with additional mini-batch and channel channels).

MindSpore: MindSpore API implements the same function as PyTorch, with two typical differences. The default value of the affine parameter in MindSpore is True, which learns the internal parameters γ and β, and the default value of PyTorch is False, which does not perform parameter learning. PyTorch supports the track_running_stats parameter. If set to True, it will use the mean and variance obtained from training in inference, and the default value is False. MindSpore does not have this parameter, and will use the computed mean and variance of the input data in both training and inference, with the same behavior as PyTorch default value.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Input | Single input | input | x | Interface input, same function, only different parameter names |
| Parameters | Parameter 1 | num_features | num_features | - |
| | Parameter 2 | eps | eps | - |
| | Parameter 3 | momentum | momentum | - |
| | Parameter 4 | affine | affine | The default values are different: MindSpore defaults to True, which learns the internal parameters γ and β, and PyTorch defaults to False, which does not learn the parameters |
| | Parameter 5 | track_running_stats | - | If set to True, PyTorch will use the mean and variance obtained from training in inference, and the default value is False. MindSpore does not have this parameter, and will use the computed mean and variance of the input data in both training and inference, with the same behavior as PyTorch default value. |
| | Parameter 6 | - | gamma_init | Initialize transform parameter γ for learning, default is 'ones', while PyTorch can't set additionally, only 'ones'|
| | Parameter 7 | - | beta_init |Initialize transform parameter γ for learning, default is 'zeros', while PyTorch can't set additionally, only 'zeros' |

### Code Example

> MindSpore affine, when set to False, has the same functions as PyTorch default behavior.

```python
# PyTorch
from torch import nn, tensor
import numpy as np

m = nn.InstanceNorm2d(num_features=3)
input_x = tensor(np.array([[[[0.1, 0.2], [0.3, 0.4]],
                          [[0.5, 0.6], [0.7, 0.8]],
                          [[0.9, 1], [1.1, 1.2]]]]).astype(np.float32))
output = m(input_x)
print(output.detach().numpy())
# [[[[-1.3411044  -0.44703478]
#    [ 0.4470349   1.3411044 ]]
#
#   [[-1.3411045  -0.44703463]
#    [ 0.44703472  1.3411046 ]]
#
#   [[-1.3411034  -0.44703388]
#    [ 0.44703573  1.3411053 ]]]]

# MindSpore
from mindspore import Tensor, nn
import numpy as np

m = nn.InstanceNorm2d(num_features=3, affine=False)
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
#    [ 0.44703573  1.3411053 ]]]]
```
