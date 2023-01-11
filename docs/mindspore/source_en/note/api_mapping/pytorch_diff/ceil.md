# Function Differences with torch.ceil

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ceil.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.ceil

```text
torch.ceil(input, *, out=None) -> Tensor
```

For more information, see [torch.ceil](https://pytorch.org/docs/1.8.1/generated/torch.ceil.html).

## mindspore.ops.ceil

```text
mindspore.ops.ceil(x) -> Tensor
```

For more information, see [mindspore.ops.ceil](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ceil.html).

## Differences

PyTorch: Return a new tensor with the ceil of the input element, which is greater than or equal to the smallest integer of each element.

MindSpore: MindSpore API implements the same function as PyTorch, and only the parameter names are different.

| Categories | Subcategories | PyTorch | MindSpore | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | input   | x         | Same function, different parameter names |
|      | Parameter 2 | out     | -         | Not involved        |

### Code Example 1

The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import numpy as np
import torch
from torch import tensor

input = torch.tensor(np.array([2.5, -1.5, 1, 1.4448, 0.5826]), dtype=torch.float32)
output = torch.ceil(input).numpy()
print(output)
# [ 3. -1.  1.  2.  1.]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.array([2.5, -1.5, 1, 1.4448, 0.5826]), mindspore.float32)
output = ops.ceil(x).asnumpy()
print(output)
# [ 3. -1.  1.  2.  1.]
```
