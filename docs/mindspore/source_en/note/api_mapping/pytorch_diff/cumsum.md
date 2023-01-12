# Function Differences with torch.cumsum

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/cumsum.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.cumsum

```text
torch.cumsum(input, dim, *, dtype=None, out=None) -> Tensor
```

For more information, see [torch.cumsum](https://pytorch.org/docs/1.8.1/generated/torch.cumsum.html).

## mindspore.ops.cumsum

```text
mindspore.ops.cumsum(x, axis, dtype=None) -> Tensor
```

For more information, see [mindspore.ops.cumsum](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.cumsum.html).

## Differences

PyTorch: Calculates the cumulative sum of the input Tensor on the specified axis.

MindSpore: MindSpore API implements functions basically same as PyTorch, but there are differences in parameter settings.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | input | x |Same function, different parameter names |
| | Parameter 2 | dim | axis | Same function, different parameter names |
| | Parameter 3 | dtype | dtype | - |
| | Parameter 4 | out | - | Not involved |

### Code Example 1

> When the input tensor is the same and the accumulation axis is -1, the innermost layer of the tensor is accumulated from left to right, and the two APIs achieve the same function.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np
a = tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float32))
y = torch.cumsum(a, dim=-1)
print(y.numpy())
# [[ 3.  7. 13. 23.]
#  [ 1.  7. 14. 23.]
#  [ 4.  7. 15. 22.]
#  [ 1.  4. 11. 20.]]

# MindSpore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
x = Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float32))
y = ops.cumsum(x, -1)
print(y)
# [[ 3.  7. 13. 23.]
#  [ 1.  7. 14. 23.]
#  [ 4.  7. 15. 22.]
#  [ 1.  4. 11. 20.]]
```

### Code Example 2

> When the input tensor and the accumulation axis are the same, torch.cumsum and MindSpore get the same result by setting the data type of the output y to int8 through the parameter dtype.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np
a = tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float32))
y = torch.cumsum(a, dim=0, dtype=torch.int8)
print(y.numpy())
# [[ 3  4  6 10]
#  [ 4 10 13 19]
#  [ 8 13 21 26]
#  [ 9 16 28 35]]
print(y.dtype)
# torch.int8

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
x = Tensor([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]], mindspore.float32)
y = ops.cumsum(x, 0, dtype=mindspore.int8)
print(y)
# [[ 3  4  6 10]
#  [ 4 10 13 19]
#  [ 8 13 21 26]
#  [ 9 16 28 35]]
print(y.dtype)
# Int8
```
