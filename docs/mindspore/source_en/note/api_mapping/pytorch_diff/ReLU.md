# Function Differences with torch.nn.ReLU

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ReLU.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.ReLU

```text
class torch.nn.ReLU(inplace=False)(input) -> Tensor
```

For more information, see [torch.nn.ReLU](https://pytorch.org/docs/1.8.1/generated/torch.nn.ReLU.html).

## mindspore.nn.ReLU

```text
class mindspore.nn.ReLU()(x) -> Tensor
```

For more information, see [mindspore.nn.ReLU](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.ReLU.html).

## Differences

PyTorch: ReLU activation function.

MindSpore: MindSpore implements the same function as PyTorch, but with different parameter settings.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
|Parameter | Parameter 1 | inplace | - | Whether to execute in-place, default: False. MindSpore does not have this parameter.|
| Input | Single input | input | x | Same function, different parameter names |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
from torch import tensor
from torch import nn
import numpy as np

x = tensor(np.array([[0.1, -0.6], [-0.9, 0.8]]), dtype=torch.float32)
m = nn.ReLU()
out = m(x)
output = out.detach().numpy()
print(output)
# [[0.1 0. ]
#  [0.  0.8]]

# MindSpore
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([[0.1, -0.6], [-0.9, 0.8]]), dtype=mindspore.float32)
relu = nn.ReLU()
output = relu(x)
print(output)
# [[0.1 0. ]
#  [0.  0.8]]
```
