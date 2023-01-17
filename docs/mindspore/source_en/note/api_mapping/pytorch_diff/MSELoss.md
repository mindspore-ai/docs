# Function Differences with torch.nn.MSELoss

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/MSELoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.MSELoss

```text
torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')(input, target) -> Tensor
```

For more information, see [torch.nn.MSELoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.MSELoss.html).

## mindspore.nn.MSELoss

```text
class mindspore.nn.MSELoss(reduction='mean')(logits, labels) -> Tensor
```

For more information, see [mindspore.nn.MSELoss](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.MSELoss.html).

## Differences

PyTorch: Used to calculate the mean square error for each element of the input and target. The reduction parameter specifies the type of statute applied to the loss.

MindSpore: Implement functions consistent with PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1 | size_average | -        | Deprecated, replaced by reduction |
| | Parameter 2 | reduce | - |  Deprecated, replaced by reduction |
| | Parameter 3 | reduction | reduction | - |
|Inputs | Input 1 | input        | logits       | Same function, different parameter names |
|      | Input 2 | target       | labels      | Same function, different parameter names |

### Code Example 1

> Compute the mean square error of `input` and `target`. By default, `reduction='mean'`.

```python
# PyTorch
import torch
from torch import nn
from torch import tensor
import numpy as np

loss = nn.MSELoss()
input_ = np.array([1, 1, 1, 1]).reshape((2, 2))
inputs = tensor(input_, dtype=torch.float32)
target_ = np.array([1, 2, 2, 1]).reshape((2, 2))
target = tensor(target_, dtype=torch.float32)
output = loss(inputs, target)
print(output.numpy())
# 0.5

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

loss = nn.MSELoss()
input_ = np.array([1, 1, 1, 1]).reshape((2, 2))
inputs = Tensor(input_, dtype=mindspore.float32)
target_ = np.array([1, 2, 2, 1]).reshape((2, 2))
target = Tensor(target_, dtype=mindspore.float32)
output = loss(inputs, target)
print(output)
# 0.5
```

### Code Example 2

> Compute the mean square error of `input` and `target` for the summation mode statute.

```python
# PyTorch
import torch
from torch import nn
from torch import tensor
import numpy as np

loss = nn.MSELoss(reduction='sum')
input_ = np.array([1, 1, 1, 1]).reshape((2, 2))
inputs = tensor(input_, dtype=torch.float32)
target_ = np.array([1, 2, 2, 1]).reshape((2, 2))
target = tensor(target_, dtype=torch.float32)
output = loss(inputs, target)
print(output.numpy())
# 2.0

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

loss = nn.MSELoss(reduction='sum')
input_ = np.array([1, 1, 1, 1]).reshape((2, 2))
inputs = Tensor(input_, dtype=mindspore.float32)
target_ = np.array([1, 2, 2, 1]).reshape((2, 2))
target = Tensor(target_, dtype=mindspore.float32)
output = loss(inputs, target)
print(output)
# 2.0
```
