# Function Differences with torch.unsqueeze

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/expand_dims.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.unsqueeze

```text
torch.unsqueeze(input, dim) -> Tensor
```

For more information, see [torch.unsqueeze](https://pytorch.org/docs/1.8.1/generated/torch.unsqueeze.html).

## mindspore.ops.expand_dims

```text
mindspore.ops.expand_dims(input_x, axis) -> Tensor
```

For more information, see [mindspore.ops.expand_dims](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.expand_dims.html).

## Differences

PyTorch: Add an extra dimension to the input input on the given axis.

MindSpore: MindSpore API implements the same function as PyTorch, and only the parameter names are different.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | --------------------- |
| Parameters | Parameter 1 | input   | input_x   | Same function, different parameter names |
|      | Parameter 2 | dim     | axis      | Same function, different parameter names |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
from torch import tensor

x = tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.float32)
dim = 1
out = torch.unsqueeze(x,dim).numpy()
print(out)
# [[[ 1.  2.  3.  4.]]
#
#  [[ 5.  6.  7.  8.]]
#
#  [[ 9. 10. 11. 12.]]]

# MindSpore
import mindspore
import numpy as np
import mindspore.ops as ops
from mindspore import Tensor

input_params = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), mindspore.float32)
axis = 1
output = ops.expand_dims(input_params,  axis)
print(output)
# [[[ 1.  2.  3.  4.]]
#
#  [[ 5.  6.  7.  8.]]
#
#  [[ 9. 10. 11. 12.]]]
```
