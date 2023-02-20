# Function Differences with torch.nn.InstanceNorm1d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/InstanceNorm1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.InstanceNorm1d

```text
class torch.nn.InstanceNorm1d(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=False,
    track_running_stats=False
)(input) -> Tensor
```

For more information, see [torch.nn.InstanceNorm1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.InstanceNorm1d.html).

## mindspore.nn.InstanceNorm1d

```text
class mindspore.nn.InstanceNorm1d(
    num_features,
    eps=1e-5,
    momentum=0.1,
    affine=True,
    gamma_init='ones',
    beta_init='zeros'
)(x) -> Tensor
```

For more information, see [mindspore.nn.InstanceNorm1d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.InstanceNorm1d.html).

## Differences

PyTorch: Apply normalization within each channel of the input 2D or 3D data (1D with additional mini-batch and channel channels or 2D input with mini-batch channels).

MindSpore: MindSpore API implements the same function as PyTorch, but currently it can only normalize 3D data, with two typical differences. The default value of the affine parameter in MindSpore is True, which learns the internal parameters γ and β, and the default value of PyTorch is False, which does not perform parameter learning. PyTorch supports the track_running_stats parameter. If set to True, it will use the mean and variance obtained from training in inference, and the default value is False. MindSpore does not have this parameter, and will use the computed mean and variance of the input data in both training and inference, with the same behavior as PyTorch default value.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1 | num_features | num_features   | - |
|      | Parameter 2 | eps | eps | - |
| | Parameter 3 | momentum | momentum | - |
| | Parameter 4 | affine | affine | The default values are different: MindSpore defaults to True, which learns the internal parameters γ and β, and PyTorch defaults to False, which does not learn the parameters |
| | Parameter 5 | track_running_stats | - | If set to True, PyTorch will use the mean and variance obtained from training in inference, and the default value is False. MindSpore does not have this parameter, and will use the computed mean and variance of the input data in both training and inference, with the same behavior as PyTorch default value. |
| | Parameter 6 | - | gamma_init | Initialize transform parameter γ for learning, default is 'ones', while PyTorch can't set additionally, only 'ones'|
| | Parameter 7 | - | beta_init |Initialize transform parameter γ for learning, default is 'zeros', while PyTorch can't set additionally, only 'zeros' |
| Input | Single input | input | x | Interface input. The function is basically the same, but PyTorch allows input to be 2D or 3D, while input in MindSpore can only be 2D |

### Code Example

> MindSpore affine, when set to False, has the same functions as PyTorch default behavior.

```python
# PyTorch
from torch import nn, tensor
import numpy as np

m = nn.InstanceNorm1d(num_features=2)
input_x = tensor(np.array([[[0.7, 0.5, 0.5, 0.6], [0.5, 0.4, 0.6, 0.9]]]).astype(np.float32))
output = m(input_x)
print(output.detach().numpy())
# [[[ 1.5064616e+00 -9.0387678e-01 -9.0387678e-01  3.0129281e-01]
#   [-5.3444624e-01 -1.0688924e+00  3.2054459e-08  1.6033382e+00]]]

# MindSpore
from mindspore import Tensor, nn
import numpy as np

m = nn.InstanceNorm1d(num_features=2, affine=False)
m.set_train()
input_x = Tensor(np.array([[[0.7, 0.5, 0.5, 0.6], [0.5, 0.4, 0.6, 0.9]]]).astype(np.float32))
output = m(input_x)
print(output)
# [[[ 1.5064610e+00 -9.0387726e-01 -9.0387726e-01  3.0129224e-01]
#   [-5.3444624e-01 -1.0688924e+00  3.2054459e-08  1.6033382e+00]]]
```

