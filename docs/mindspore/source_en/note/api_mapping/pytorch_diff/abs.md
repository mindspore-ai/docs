# Function Differences with torch.abs

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/abs.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.abs

```text
torch.abs(input, *, out=None) -> Tensor
```

For more information, see [torch.abs](https://pytorch.org/docs/1.8.1/generated/torch.abs.html).

## mindspore.ops.abs

```text
mindspore.ops.abs(x) -> Tensor
```

For more information, see [mindspore.ops.abs](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.abs.html).

## Differences

PyTorch: Calculates the absolute value of the input.

MindSpore: MindSpore API implements the same function as PyTorch, and only the parameter names are different.

| Categories | Subcategories   | PyTorch     | MindSpore   | Differences   |
| ---- | ----- | ------- | --------- | --------------------- |
| Parameters | Parameter1 | input   | x | Same function, different parameter names |
|  | Parameter2 | out | - | Not involved |

### Code Example 1

The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
from torch import tensor

input = torch.tensor([-1, 1, 0], dtype=torch.float32)
output = torch.abs(input).numpy()
print(output)
# [1. 1. 0.]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.array([-1, 1, 0]), mindspore.float32)
output = ops.abs(x).asnumpy()
print(output)
# [1. 1. 0.]
```
