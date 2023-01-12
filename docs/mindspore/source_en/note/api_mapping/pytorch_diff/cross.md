# Function Differences with torch.cross

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/cross.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.cross

```text
torch.cross(input, other, dim=None, *, out=None) -> Tensor
```

For more information, see [torch.cross](https://pytorch.org/docs/1.8.1/generated/torch.cross.html).

## mindspore.ops.cross

```text
mindspore.ops.cross(input, other, dim=None) -> Tensor
```

For more information, see [mindspore.ops.cross](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.cross.html).

## Differences

PyTorch: Return the cross product of input and other vector sets.

MindSpore: MindSpore API implements essentially the same function as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | input   | input   | Same function, different parameter names |
|      | Parameter 2 | other   | other    | Same function, different parameter names |
|      | Parameter 3 | dim     | dim   | -     |
|      | Parameter 4 | out  | -  | Not involved    |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
from torch import tensor

a = tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
b = tensor([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
c = torch.cross(a, b).detach().numpy()
print(c)
# [[-1 -1 -1]
#  [-1 -2 -3]
#  [ 1  2  3]]

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype
import mindspore.ops as ops

a = Tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]], mstype.int8)
b = Tensor([[1, 2, 3], [3, 4, 5], [4, 5, 6]], mstype.int8)
c = ops.cross(a, b)
print(c)
# [[-1 -1 -1]
#  [-1 -2 -3]
#  [ 1  2  3]]
```
