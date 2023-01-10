# Function Differences with torch.atan

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/atan2.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.atan2

```text
torch.atan2(input, other, *, out=None) -> Tensor
```

For more information, see [torch.atan2](https://pytorch.org/docs/1.8.1/generated/torch.atan2.html).

## mindspore.ops.atan2

```text
mindspore.ops.atan2(x, y) -> Tensor
```

For more information, see [mindspore.ops.atan2](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.atan2.html).

## Differences

PyTorch: Compute the inverse tangent of input/other for the considered quadrant element-wise, where the second parameter other is the x-coordinate and the first parameter input is the y-coordinate.

MindSpore: MindSpore API implements the same function as PyTorch, also supports Scalar input for x or y.

| Categories | Subcategories   | PyTorch     | MindSpore   | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | input   | x | Same function, different parameter names |
|  | Parameter 2 | other | y | Same function, different parameter names |
|      | Parameter 3 | out     | -    | Not involved    |

### Code Example 1

When the input x and y are both Tensor, the two APIs achieve the same function.

```python
# PyTorch
import numpy as np
import torch
from torch import tensor

input = torch.tensor(np.array([0, 1]), dtype=torch.float32)
other = torch.tensor(np.array([1, 1]), dtype=torch.int)
output = torch.atan2(input, other).numpy()
print(output)
# [0.        0.7853982]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.array([0, 1]), mindspore.float32)
y = Tensor(np.array([1, 1]), mindspore.float32)

output = ops.atan2(x, y)
print(output)
# [0.        0.7853982]
```

### Code Example 2

Note: When the input x or y is Scalar, MindSpore can achieve the corresponding function, while pytorch does not support.

```python
# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = 1
y = Tensor(np.array([1, 1]), mindspore.float32)

output = ops.atan2(x, y)
print(output)
# [0.7853982 0.7853982]
```
