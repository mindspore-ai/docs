# Function Differences with torch.acos

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/acos.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The following mapping relationships can be found in this file.

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|   torch.acos      |   mindspore.ops.acos|
|    torch.arccos     |  mindspore.ops.arccos   |
|   torch.Tensor.acos   |   mindspore.Tensor.acos    |
| torch.Tensor.arccos | mindspore.Tensor.arccos |

## torch.acos

```text
torch.acos(input, *, out=None) -> Tensor
```

For more information, see [torch.acos](https://pytorch.org/docs/1.8.1/generated/torch.acos.html).

## mindspore.ops.acos

```text
mindspore.ops.acos(x) -> Tensor
```

For more information, see [mindspore.ops.acos](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.acos.html).

## Differences

PyTorch: Compute the inverse cosine of the input Tensor element-wise.

MindSpore: MindSpore API implements the same function as PyTorch, and only the parameter names are different.

| Categories | Subcategories   | PyTorch     | MindSpore   | Differences   |
| ---- | ----- | ------- | --------- | --------------------- |
| Parameters | Parameter 1 | input   | x | Same function, different parameter names |
|  | Parameter 2 | out | - | Not involved |

### Code Example 1

The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import numpy as np
import torch
from torch import tensor

input = torch.tensor(np.array([0.74, 0.04, 0.30, 0.56]), dtype=torch.float32)
output = torch.acos(input).numpy()
print(output)
# [0.737726  1.5307857 1.2661036 0.9764105]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float32)
output = ops.acos(x)
print(output)
# [0.737726  1.5307857 1.2661036 0.9764105]
```
