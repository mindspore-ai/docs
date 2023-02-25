# Function Differences with torch.div

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/div.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.div

```text
torch.div(input, other, *, rounding_mode=None, out=None) -> Tensor
```

For more information, see [torch.div](https://pytorch.org/docs/1.8.1/generated/torch.div.html).

## mindspore.ops.div

```text
mindspore.ops.div(input, other, rounding_mode=None) -> Tensor
```

For more information, see [mindspore.ops.div](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.div.html).

## Differences

PyTorch: Calculate the quotient of the first input divided by the second input, where the quotient depends on the rounding_mode parameter.

MindSpore: MindSpore API achieves the same function as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
|:----|-----|:--------------|-----------|-----------------------------------|
| Parameters | Parameter 1 | input    | input  | -   |
| | Parameter 2 | other         | other  | -  |
| | Parameter 3 | rounding_mode | rounding_mode | -   |
| | Parameter 4 | out           | -    |Not involved |

### Code Example 1

> When the parameter rounding_mode of both APIs is trunc, both APIs round the result obtained by division to zero.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x = tensor(np.array([1, -3, 8, 9]), dtype=torch.float32)
y = tensor(np.array([3, -2, -7, 5]), dtype=torch.float32)
out = torch.div(x, y, rounding_mode='trunc').detach().numpy()
print(out)
# [ 0.  1. -1.  1.]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x = Tensor(np.array([1, -3, 8, 9]), mindspore.float32)
y = Tensor(np.array([3, -2, -7, 5]), mindspore.float32)
output = ops.div(x, y, rounding_mode='trunc')
print(output)
# [ 0.  1. -1.  1.]
```

### Code Example 2

> When the parameter rounding_mode of both APIs is floor, both APIs round down the result obtained by dividing.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x = tensor(np.array([1, -3, 8, 9]), dtype=torch.float32)
y = tensor(np.array([3, -2, -7, 5]), dtype=torch.float32)
out = torch.div(x, y, rounding_mode='floor').detach().numpy()
print(out)
# [ 0.  1. -2.  1.]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x = Tensor(np.array([1, -3, 8, 9]), mindspore.float32)
y = Tensor(np.array([3, -2, -7, 5]), mindspore.float32)
output = ops.div(x, y, rounding_mode='floor')
print(output)
# [ 0.  1. -2.  1.]
```

### Code Example 3

> When the parameter rounding_mode of both APIs is None by default, both APIs do not perform any rounding operation on the result obtained by division.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x = tensor(np.array([[np.arange(1, 7).reshape(2, 3), np.arange(-7, -1).reshape(2, 3)]]), dtype=torch.float32)
y = tensor(np.ones((2, 3)), dtype=torch.float32)
out = torch.div(x, y).detach().numpy()
print(out)
# [[[[ 1.  2.  3.]
#    [ 4.  5.  6.]]
#
#   [[-7. -6. -5.]
#    [-4. -3. -2.]]]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x = Tensor(np.array([[np.arange(1, 7).reshape(2, 3),np.arange(-7, -1).reshape(2, 3)]]), mindspore.float32)
y = Tensor(np.ones((2, 3)), mindspore.float32)
output = ops.div(x, y)
print(output)
# [[[[ 1.  2.  3.]
#    [ 4.  5.  6.]]
#
#   [[-7. -6. -5.]
#    [-4. -3. -2.]]]]
```
