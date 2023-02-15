# Function Differences with torch.nn.GaussianNLLLoss

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/GaussianNLLLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.GaussianNLLLoss

```text
class torch.nn.GaussianNLLLoss(
    *,
    full=False,
    eps=1e-06,
    reduction='mean'
)(input, target, var) -> Tensor/Scalar
```

For more information, see [torch.nn.GaussianNLLLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.GaussianNLLLoss.html).

## mindspore.nn.GaussianNLLLoss

```text
class mindspore.nn.GaussianNLLLoss(
    *,
    full=False,
    eps=1e-06,
    reduction='mean'
)(logits, labels, var) -> Tensor/Scalar
```

For more information, see [mindspore.nn.GaussianNLLLoss](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.GaussianNLLLoss.html).

## Differences

PyTorch: Obey the negative log-likelihood loss of the Gaussian distribution.

MindSpore: Implements the same function as PyTorch. If there is a number less than 0 in var, PyTorch will report an error directly, while MindSpore will calculate max(var, eps) and then pass the result to log for calculation.

| Categories | Subcategories |PyTorch | MindSpore | Differences |
|-----|-----|-----------|-----------|------------|
| Parameters  | Parameter 1 | full      | full      | Same function       |
|     | Parameter 2 | eps       | -         | Same function       |
|     | Parameter 3 | reduction | -         | Same function       |
| Inputs  | Input 1 | input     | logits    | Same function, different parameter names |
|     | Input 2 | target    | labels    | Same function, different parameter names |
|     | Input 3 | var       | var       | Same function       |

### Code Example

> The two APIs implement basically the same functionality and usage, but PyTorch and MindSpore handle the case of input `var<0` differently.

```python
# PyTorch
import torch
from torch import nn
import numpy as np

arr1 = np.arange(8).reshape((4, 2))
arr2 = np.array([2, 3, 1, 4, 6, 4, 4, 9]).reshape((4, 2))
logits = torch.tensor(arr1, dtype=torch.float32)
labels = torch.tensor(arr2, dtype=torch.float32)
loss = nn.GaussianNLLLoss(reduction='mean')
var = torch.tensor(np.ones((4, 1)), dtype=torch.float32)
output = loss(logits, labels, var)
# tensor(1.4375)

# If there are elements in the var that are less than 0, PyTorch will directly report an error
var[0] = -1
output2 = loss(logits, labels, var)
# ValueError: var has negative entry/entries

# MindSpore
import numpy as np
from mindspore import Tensor
import mindspore.nn as nn
from mindspore import dtype as mstype

arr1 = np.arange(8).reshape((4, 2))
arr2 = np.array([2, 3, 1, 4, 6, 4, 4, 9]).reshape((4, 2))
logits = Tensor(arr1, mstype.float32)
labels = Tensor(arr2, mstype.float32)
loss = nn.GaussianNLLLoss(reduction='mean')
var = Tensor(np.ones((4, 1)), mstype.float32)
output = loss(logits, labels, var)
print(output)
# 1.4374993

# If there are elements that are less than 0 in var, MindSpore will use the result of max(var, eps)
var[0] = -1
output2 = loss(logits, labels, var)
print(output2)
# 499999.22
```
