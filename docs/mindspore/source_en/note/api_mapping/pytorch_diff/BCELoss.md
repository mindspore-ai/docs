# Function Differences with torch.nn.functional.binary_cross_entropy

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/BCELoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.functional.binary_cross_entropy

```text
torch.nn.functional.binary_cross_entropy(
    input,
    target,
    weight=None,
    size_average=None,
    reduce=None,
    reduction='mean'
) -> Tensor
```

For more information, see [torch.nn.functional.binary_cross_entropy](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.binary_cross_entropy).

## mindspore.nn.BCELoss

```text
class mindspore.nn.BCELoss(
    weight=None,
    reduction='none'
)(logits, labels) -> Tensor
```

For more information, see [mindspore.nn.BCELoss](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.BCELoss.html).

## Differences

PyTorch: Compute the binary cross-entropy loss value between the target and predicted values.

MindSpore: MindSpore API basically implements the same function as PyTorch.

| Categories | Subcategories | PyTorch | MindSpore | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | input     | logits    | Same function, different parameter names                  |
|      | Parameter 2 | target    | labels    | Same function, different parameter names                 |
|      | Parameter 3 | weight    | weight    | -  |
|      | Parameter 4 | size_average    | -    | PyTorch deprecated parameters, functionally replaced by the reduction parameter          |
|      | Parameter 5 | reduce    | -    | PyTorch deprecated parameters, functionally replaced by the reduction parameter        |
|      | Parameter 6 | reduction | reduction | Same function, specifying how the output result is calculated. PyTorch defaults to "mean" and MindSpore defaults to None. |

### Code Example 1

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
import torch.nn.functional as F
from torch import tensor

input = tensor([0.1, 0.2, 0.3], requires_grad=True)
target = tensor([1., 1., 1.])
loss = F.binary_cross_entropy(input, target)
print(loss.detach().numpy())
# 1.7053319

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor
from mindspore import nn

loss = nn.BCELoss(reduction='mean')
logits = Tensor(np.array([0.1, 0.2, 0.3]), mindspore.float32)
labels = Tensor(np.array([1., 1., 1.]), mindspore.float32)
output = loss(logits, labels)
print(output)
# 1.7053319
```
