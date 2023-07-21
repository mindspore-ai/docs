# Differences with torch.nn.GRUCell

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/GRUCell.md)

## torch.nn.GRUCell

```text
class torch.nn.GRUCell(
    input_size,
    hidden_size,
    bias=True)(input, hidden) -> Tensor
```

For more information, see [torch.nn.GRUCell](https://pytorch.org/docs/1.8.1/generated/torch.nn.GRUCell.html).

## mindspore.nn.GRUCell

```text
class mindspore.nn.GRUCell(
    input_size: int,
    hidden_size: int,
    has_bias: bool=True)(x, hx) -> Tensor
```

For more information, see [mindspore.nn.GRUCell](https://www.mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.GRUCell.html).

## Differences

PyTorch: Recurrent Neural Network unit.

MindSpore: MindSpore API implements the same functions as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
|Parameters | Parameter 1 | input_size | input_size |- |
| | Parameter 2 | hidden_size | hidden_size | - |
| | Parameter 3 | bias | has_bias | Same function, different parameter names |
|Inputs | Input 1 | input | x | Same function, different parameter names |
| | Input 2 | hidden | hx |  Same function, different parameter names |

### Code Example 1

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

grucell = torch.nn.GRUCell(2, 3, bias=False)
input = torch.tensor(np.array([[3.0, 4.0]]).astype(np.float32))
hidden = torch.tensor(np.array([[1.0, 2.0, 3]]).astype(np.float32))
output = grucell(input, hidden)
print(output)
# tensor([[ 0.9948,  0.0913, -0.1633]], grad_fn=<AddBackward0>)

# MindSpore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

grucell = nn.GRUCell(2, 3, has_bias=False)
x = Tensor(np.array([[3.0, 4.0]]).astype(np.float32))
hx = Tensor(np.array([[1.0, 2.0, 3]]).astype(np.float32))
output = grucell(x, hx)
print(output)
# [[-0.94861907  0.6191679   2.1289415 ]]
```
