# Function Differences with torch.nn.LayerNorm

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/LayerNorm.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.LayerNorm

```python
class torch.nn.LayerNorm(
    normalized_shape,
    eps=1e-05,
    elementwise_affine=True
)(input) -> Tensor
```

For more information, see [torch.nn.LayerNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.LayerNorm.html).

## mindspore.nn.LayerNorm

```python
class mindspore.nn.LayerNorm(
    normalized_shape,
    begin_norm_axis=-1,
    begin_params_axis=-1,
    gamma_init='ones',
    beta_init='zeros',
    epsilon=1e-7
)(x) -> Tensor
```

For more information, see [mindspore.nn.LayerNorm](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.LayerNorm.html).

## Differences

PyTorch: Layer Normalization is applied on the mini-batch input, where the parameter `elementwise_affine` is used to control whether learnable parameters are used.

MindSpore: MindSpore API basically implements the same function as PyTorch, but there is no parameter `elementwise_affine` in MindSpore, and the parameter `begin_norm_axis` is added to control the axis of the normalized start calculation. The parameter `begin_params_axis` controls the dimension of the first parameter (beta, gamma), and the parameters `gamma_init` and `beta_init` are used to control the initialization method of the `gamma` and `beta` parameters.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Input | Single input | input | x | Interface input, same function, different parameter names|
|Parameters | Parameter 1 | normalized_shape | normalized_shape |Same function, same parameter name |
| | Parameter 2 | eps | epsilon | Same function, different parameter names, different default values |
| | Parameter 3 | elementwise_affine | - | This parameter is used in PyTorch to control whether the learnable parameters are used. MindSpore does not have this parameter|
| | Parameter 4 | - | begin_norm_axis | This parameter in MindSpore controls the axis on which the normalization begins. PyTorch does not have this parameter|
| | Parameter 5 | - | begin_params_axis | This parameter in MindSpore controls the dimensionality of the first parameter (beta, gamma). PyTorch does not have this parameter |
| | Parameter 6 | - | gamma_init | This parameter in MindSpore controls how the `γ` parameter is initialized. PyTorch does not have this parameter|
| | Parameter 7 | - | beta_init | This parameter in MindSpore controls how the `β` parameter is initialized. PyTorch does not have this parameter |

### Code Example

> When the parameter `elementwise_affine` is True in this API of PyTorch, MindSpore and PyTorch implement basically the same function.

```python
# PyTorch
import torch
import torch.nn as nn

inputs = torch.ones([20, 5, 10, 10])
m = nn.LayerNorm(inputs.size()[1:])
output = m(inputs)
print(output.detach().numpy().shape)
# (20, 5, 10, 10)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.numpy as np
import mindspore.nn as nn

x = Tensor(np.ones([20, 5, 10, 10]), mindspore.float32)
shape1 = x.shape[1:]
m = nn.LayerNorm(shape1, begin_norm_axis=1, begin_params_axis=1)
output = m(x).shape
print(output)
# (20, 5, 10, 10)
```
