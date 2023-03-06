# Function Differences with torch.nn.RNNCell

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RNNCell.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.RNNCell

```text
class torch.nn.RNNCell(
    input_size,
    hidden_size,
    bias=True,
    nonlinearity='tanh')(input, hidden) -> Tensor
```

For more information, see [torch.nn.RNNCell](https://pytorch.org/docs/1.8.1/generated/torch.nn.RNNCell.html).

## mindspore.nn.RNNCell

```text
class mindspore.nn.RNNCell(
    input_size: int,
    hidden_size: int,
    has_bias: bool=True,
    nonlinearity: str = 'tanh')(x, hx) -> Tensor
```

For more information, see [mindspore.nn.RNNCell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.RNNCell.html).

## Differences

PyTorch: Recurrent Neural Network (RNN) unit.

MindSpore: MindSpore API implements the same functions as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
|Parameters | Parameter 1 | input_size | input_size |- |
| | Parameter 2 | hidden_size | hidden_size | - |
| | Parameter 3 | bias | has_bias | Same function, different parameter names |
| | Parameter 4 | nonlinearity | nonlinearity | - |
|Inputs | Input 1 | input | x | Same function, different parameter names |
| | Input 2 | hidden | hx |  Same function, different parameter names |

### Code Example 1

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

rnncell = torch.nn.RNNCell(2, 3, nonlinearity="relu", bias=False)
input = torch.tensor(np.array([[3.0, 4.0]]).astype(np.float32))
hidden = torch.tensor(np.array([[1.0, 2.0, 3]]).astype(np.float32))
output = rnncell(input, hidden)
print(output)
# tensor([[0.5022, 0.0000, 1.4989]], grad_fn=<ReluBackward0>)

# MindSpore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

rnncell = nn.RNNCell(2, 3, nonlinearity="relu", has_bias=False)
x = Tensor(np.array([[3.0, 4.0]]).astype(np.float32))
hx = Tensor(np.array([[1.0, 2.0, 3]]).astype(np.float32))
output = rnncell(x, hx)
print(output)
# [[2.4998584 0.        1.9334991]]
```
