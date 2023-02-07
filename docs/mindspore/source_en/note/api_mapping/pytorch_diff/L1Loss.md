# Function Differences with torch.nn.L1Loss

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/L1Loss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.L1Loss

```text
torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')(input, target) -> Tensor
```

For more information, see [torch.nn.L1Loss](https://pytorch.org/docs/1.8.1/generated/torch.nn.L1Loss.html).

## mindspore.nn.L1Loss

```text
mindspore.nn.L1Loss(reduction='mean')(logits, labels) -> Tensor
```

For more information, see [mindspore.nn.L1Loss](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.L1Loss.html).

## Differences

PyTorch: L1Loss is used to calculate the average absolute error between the predicted and target values.

MindSpore: Includes PyTorch function, which can still run when logits and labels have different shapes but can broadcast to each other, while PyTorch cannot.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1 | size_average     | -    | Deprecated, function taken over by reduction |
|      | Parameter 2 | reduce    | -    | Deprecated, function taken over by reduction|
|      | Parameter 3 | reduction | reduction | - |
| Input  | Input 1 | input     | logits    | Same function, different parameter names |
|      | Input 2 | target    | labels    | Same function, different parameter names |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
import torch.nn as nn

loss = nn.L1Loss()
input = torch.tensor([2, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 2], dtype=torch.float32)
output = loss(input, target)
output = output.detach().numpy()
print(output)
# 0.6666667

# MindSpore
import mindspore
from mindspore import Tensor, nn
import numpy as np

loss = nn.L1Loss()
logits = Tensor(np.array([2, 2, 3]), mindspore.float32)
labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
output = loss(logits, labels)
print(output)
# 0.6666667
```
