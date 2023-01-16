# Function Differences with torch.expm1

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/expm1.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.expm1

```text
torch.expm1(input, *, out=None) -> Tensor
```

For more information, see [torch.expm1](https://pytorch.org/docs/1.8.1/generated/torch.expm1.html).

## mindspore.ops.expm1

```text
mindspore.ops.expm1(x) -> Tensor
```

For more information, see [mindspore.ops.expm1](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.expm1.html).

## Differences

PyTorch: Calculate the value of the index of the input tensor minus 1, element-wise.

MindSpore: MindSpore API implements the same function as PyTorch, and only the parameter names are different.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
|Parameters | Parameter 1 | input | x |Same function, different parameter names |
|  | Parameter 2 | out | - | Not involved |

### Code Example 1

> Both APIs implement the same function and have the same usage.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

inputx_ = np.array([0.0, 1.0, 2.0, 4.0])
inputx = tensor(inputx_, dtype=torch.float32)
output = torch.expm1(inputx)
output_m = output.detach().numpy()
print(output_m)
#[ 0.         1.7182817  6.389056  53.59815  ]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x_ = np.array([0.0, 1.0, 2.0, 4.0])
x = Tensor(x_, mindspore.float32)
output = ops.expm1(x)
print(output)
#[0.        1.7182819  6.389056  53.598152]
```
