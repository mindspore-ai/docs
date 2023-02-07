# Function Differences with torch.nn.functional.l1_loss

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/L1Loss_func.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.functional.l1_loss

```text
torch.nn.functional.l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor
```

For more information, see [torch.nn.functional.l1_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#l1-loss).

## mindspore.nn.L1Loss

```text
mindspore.nn.L1Loss(reduction='mean')(logits, labels) -> Tensor
```

For more information, see [mindspore.nn.L1Loss](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.L1Loss.html).

## Differences

PyTorch: functional.l1_loss is equivalent to L1Loss.

MindSpore: Includes PyTorch function, which can still run when logits and labels have different shapes but can broadcast to each other, while PyTorch cannot.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters  | Parameter 1| input     | logits    | Same function, different parameter names |
|      | Parameter 2| target    | labels    | Same function, different parameter names |
|      | Parameter 3 | size_average     | -    | Deprecated, function taken over by reduction |
|      | Parameter 4 | reduce    | -    | Deprecated, function taken over by reduction|
|      | Parameter 5 | reduction | reduction | - |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
import torch.nn as nn

loss = nn.functional.l1_loss
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

