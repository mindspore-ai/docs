# Function Differences with torch.nn.Linear

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Dense.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Linear

```text
class torch.nn.Linear(
    in_features,
    out_features,
    bias=True
)(input) -> Tensor
```

For more information, see [torch.nn.Linear](https://pytorch.org/docs/1.8.1/generated/torch.nn.Linear.html).

## mindspore.nn.Dense

```text
class mindspore.nn.Dense(
    in_channels,
    out_channels,
    weight_init='normal',
    bias_init='zeros',
    has_bias=True,
    activation=None
)(x) -> Tensor
```

For more information, see [mindspore.nn.Dense](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Dense.html).

## Differences

Pytorch: Fully connected layer that implements the matrix multiplication operation.

MindSpore: The implementation function of the API in MindSpore is basically the same as that of PyTorch, and it is possible to add activation functions after the fully connected layer.

| Categories | Subcategories   | PyTorch             | MindSpore   | Differences    |
| ---- | ----- | ------------ | ------------ | ---------------------------- |
| Parameters | Parameter 1 | in_features  | in_channels  | Same function, different parameter names              |
|      | Parameter 2 | out_features | out_channels | Same function, different parameter names       |
|      | Parameter 3 | bias         | has_bias     | Same function, different parameter names         |
|      | Parameter 4 | -             | weight_init  | Initialization method for the weight parameter, which is not available for PyTorch         |
|      | Parameter 5 | -             | bias_init    | Initialization method for the bias parameter, which is not available for PyTorch           |
|      | Parameter 6 | -             | activation   | Activation function applied to the output of the fully connected layer, which is not available for PyTorch   |
| Input | Single input | input | x | Same function, only different parameter names |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
from torch import nn
import numpy as np

net = nn.Linear(3, 4)
x = torch.tensor(np.array([[180, 234, 154], [244, 48, 247]]), dtype=torch.float)
output = net(x)
print(output.detach().numpy().shape)
# (2, 4)

# MindSpore
import mindspore
from mindspore import Tensor, nn
import numpy as np

x = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), mindspore.float32)
net = nn.Dense(3, 4)
output = net(x)
print(output.shape)
# (2, 4)
```