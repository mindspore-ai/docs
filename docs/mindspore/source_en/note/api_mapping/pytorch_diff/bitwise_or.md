# Function Differences with torch.bitwise_or

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/bitwise_or.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.bitwise_or

```text
torch.bitwise_or(input, other, *, out=None) -> Tensor
```

For more information, see [torch.bitwise_or](https://pytorch.org/docs/1.8.1/generated/torch.bitwise_or.html).

## mindspore.ops.bitwise_or

```text
mindspore.ops.bitwise_or(input, other) -> Tensor
```

For more information, see [mindspore.ops.bitwise_or](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.bitwise_or.html).

## Differences

PyTorch: Calculates the logical or of two tensor data if the input data type is Boolean, otherwise calculates the bitwise or of two tensor data.

MindSpore: MindSpore API implements the same function as PyTorch, but MindSpore does not support tensor data of Boolean.

| Categories | Subcategories | PyTorch | MindSpore | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | input   | input         | No difference |
|      | Parameter 2 | other   | other         | No difference |
|      | Parameter 3 | out | -         | Not involved    |

### Code Example 1

The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import numpy as np
import torch
from torch import tensor

input = torch.tensor(np.array([0, 0, 1, -1, 1, 1, 1]), dtype=torch.int32)
other = torch.tensor(np.array([0, 1, 1, -1, -1, 2, 3]), dtype=torch.int32)
output = torch.bitwise_or(input, other).numpy()
print(output)
# [ 0  1  1 -1 -1  3  3]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

input = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int32)
other = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int32)
output = ops.bitwise_or(input, other)
print(output)
# [ 0  1  1 -1 -1  3  3]
```
