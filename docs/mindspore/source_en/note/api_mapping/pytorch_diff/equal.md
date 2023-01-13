# Function Differences with torch.eq

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/equal.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.eq

```text
torch.eq(input, other, *, out=None) -> Tensor
```

For more information, see [torch.eq](https://pytorch.org/docs/1.8.1/generated/torch.eq.html).

## mindspore.ops.equal

```text
mindspore.ops.equal(x, y) -> Tensor
```

For more information, see [mindspore.ops.equal](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.equal.html).

## Differences

PyTorch: Compare two input Tensors element-wise to see if they are equal.

MindSpore: MindSpore API implements the same function as PyTorch, and only the parameter names are different.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| :-: | :-: | :-: | :-: |:-:|
|Parameters | Parameter 1 | input | x |Same function, different parameter names |
| | Parameter 2 | other | y |Same function, different parameter names |
| | Parameter 3 | out | - |Not involved |

### Code Example 1

> Implement the same function and use the same usage.

```python
# PyTorch
import torch
from torch import tensor

input1 = tensor([1, 2], dtype=torch.float32)
other = tensor([[1, 2], [0, 2], [1, 3]], dtype=torch.int64)
out = torch.eq(input1, other).numpy()
print(out)
# [[ True  True]
#  [False  True]
#  [ True False]]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([1, 2]), mindspore.float32)
y = Tensor(np.array([[1, 2], [0, 2], [1, 3]]), mindspore.int64)
output = mindspore.ops.equal(x, y)
print(output)
# [[ True  True]
#  [False  True]
#  [ True False]]
```

### Code Example 2

> Implement the same function and use the same usage.

```python
# PyTorch
import torch
from torch import tensor

input1 = tensor([1, 3, 1, 4], dtype=torch.int32)
out = torch.eq(input1, 1).numpy()
print(out)
# [ True False  True False]

# MindSpore
import mindspore
from mindspore import Tensor

x = Tensor([1, 3, 1, 4], mindspore.int32)
output = mindspore.ops.equal(x, 1)
print(output)
# [ True False  True False]
```
