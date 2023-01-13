# Function Differences with torch.exp

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/exp.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.exp

```text
torch.exp(input, *, out=None) -> Tensor
```

For more information, see [torch.exp](https://pytorch.org/docs/1.8.1/generated/torch.exp.html).

## mindspore.ops.exp

```text
mindspore.ops.exp(x) -> Tensor
```

For more information, see [mindspore.ops.exp](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.exp.html).

## Differences

PyTorch: Computes the index of the input tensor `input` element-wise.

MindSpore: MindSpore API implements the same function as PyTorch, and only the parameter names are different.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| :-: | :-: | :-: | :-: |:-:|
|Parameters | Parameter 1 | input | x | Same function, different parameter names |
| | Parameter 2 | out | - |Not involved |

### Code Example 1

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
from torch import tensor

x = tensor([[0, 1, 2], [0, -1, -2]], dtype=torch.float32)
out = torch.exp(x).numpy()
print(out)
# [[1.         2.7182817  7.389056  ]
#  [1.         0.36787945 0.13533528]]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([[0, 1, 2], [0, -1, -2]]), mindspore.float32)
output = mindspore.ops.exp(x)
print(output)
# [[1.         2.718282   7.3890557 ]
#  [1.         0.36787948 0.13533528]]
```

### Code Example 2

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
from torch import tensor
import math

x = tensor([-1, 1, math.log(2.0)], dtype=torch.float32)
out = torch.exp(x).numpy()
print(out)
# [0.36787945 2.7182817  2.        ]

# MindSpore
import mindspore
from mindspore import Tensor
import math

x = Tensor([-1, 1, math.log(2.0)], mindspore.float32)
output = mindspore.ops.exp(x)
print(output)
# [0.36787948 2.718282   2.        ]
```
