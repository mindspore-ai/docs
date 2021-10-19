# Function Differences with torch.nn.Linear

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/Dense.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## torch.nn.Linear

```python
torch.nn.Linear(
    in_features,
    out_features,
    bias=True
)
```

For more information, see [torch.nn.Linear](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Linear).

## mindspore.nn.Dense

```python
class mindspore.nn.Dense(
    in_channels,
    out_channels,
    weight_init='normal',
    bias_init='zeros',
    has_bias=True,
    activation=None
)(input)
```

For more information, see [mindspore.nn.Dense](https://mindspore.cn/docs/api/en/r1.5/api_python/nn/mindspore.nn.Dense.html#mindspore.nn.Dense).

## Differences

Pytorch: Applies a linear transformation to the incoming data.

MindSpore: Applies a linear transformation to the incoming data, and applies the `activation` function before outputting the data.

## Code Example

```python
import mindspore
from mindspore import Tensor, nn
import torch
import numpy as np

# In MindSpore, default weight will be initialized through standard normal distribution.
# Default bias will be initialized by zero.
# Default none activation used.
input_net = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), mindspore.float32)
net = nn.Dense(3, 4)
output = net(input_net)
print(output.shape)
# Out：
# (2, 4)

# In torch, default weight and bias will be initialized through uniform distribution.
# No parameter to set the activation.
input_net = torch.Tensor(np.array([[180, 234, 154], [244, 48, 247]]))
net = torch.nn.Linear(3, 4)
output = net(input_net)
print(output.shape)
# Out：
# torch.Size([2, 4])
```