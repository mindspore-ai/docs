# Function Differences with torch.atan

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/atan.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.atan

```text
torch.atan(input, *, out=None) -> Tensor
```

For more information, see [torch.atan](https://pytorch.org/docs/1.8.1/generated/torch.atan.html).

## mindspore.ops.atan

```text
mindspore.ops.atan(x) -> Tensor
```

For more information, see [mindspore.ops.atan](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.atan.html).

## Differences

PyTorch: Compute the inverse tangent of the input Tensor element-wise.

MindSpore: MindSpore API implements the same function as PyTorch, and only the parameter names are different.

| Categories | Subcategories   | PyTorch     | MindSpore   | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | input   | x | Same function, different parameter names |
|  | Parameter 2 | out | - | Not involved |

### Code Example 1

The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import numpy as np
import torch
from torch import tensor

input = torch.tensor(np.array([0.2341, 1.0, 0.0, -0.6448]), dtype=torch.float32)
output = torch.atan(input).numpy()
print(output)
# [ 0.22995889  0.7853982   0.         -0.572711  ]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.array([0.2341, 1.0, 0.0, -0.6448]), mindspore.float32)
output = ops.atan(x)
print(output)
# [ 0.22995889  0.7853982   0.         -0.572711  ]
```
