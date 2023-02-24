# Function Differences with torch.asin

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/asin.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.asin

```text
torch.asin(input, *, out=None) -> Tensor
```

For more information, see [torch.asin](https://pytorch.org/docs/1.8.1/generated/torch.asin.html).

## mindspore.ops.asin

```text
mindspore.ops.asin(x) -> Tensor
```

For more information, see [mindspore.ops.asin](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.asin.html).

## Differences

PyTorch: Compute the inverse sine of the input Tensor element-wise.

MindSpore: MindSpore API implements the same function as PyTorch, and only the parameter names are different.

| Categories | Subcategories| PyTorch | MindSpore |Differences |
| ---- | ----- | ------- | --------- |------------------ |
| Parameters | Parameter 1 | input   | x | Same function, different parameter names |
|  | Parameter 2 | out | - | Not involved |

### Code Example 1

The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import numpy as np
import torch
from torch import tensor

input = torch.tensor(np.array([-0.14, 0.14, 0.30, 0.314]), dtype=torch.float32)
output = torch.asin(input).numpy()
print(output)
# [-0.14046142  0.14046142  0.30469266  0.3194032 ]

# MindSpore
import numpy as np
import mindspore
import mindspore.context as context
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.array([-0.14, 0.14, 0.30, 0.314]), mindspore.float32)
output = ops.asin(x)
print(output)
# [-0.14046142  0.14046142  0.30469266  0.3194032 ]
```
