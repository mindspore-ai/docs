# Function Differences with torch.nn.Linear

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Dense.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

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

For more information, see [mindspore.nn.Dense](https://mindspore.cn/docs/en/r2.0.0-alpha/api_python/nn/mindspore.nn.Dense.html).

## Differences

Pytorch: Fully connected layer that implements the matrix multiplication operation.

MindSpore: MindSpore API basically implements the same function as TensorFlow, and it is possible to add activation functions after the fully connected layer.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1 | in_features  | in_channels  | Same function, different parameter names                          |
|      | Parameter 2 | out_features | out_channels | Same function, different parameter names                        |
|      | Parameter 3 | bias         | has_bias     | Same function, different parameter names        |
|      | Parameter 4 | -             | weight_init  | Initialization method for the weight parameter. PyTorch does not have this parameter         |
|      | Parameter 5 | -             | bias_init    | Initialization method for the bias parameter. PyTorch does not have this parameter           |
|      | Parameter 6 | -             | activation   | Activation function applied to the output of the fully connected layer. PyTorch does not have this parameter   |
|  Input   | Single input | input | x | Same function, different parameter names|

## Code Example

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