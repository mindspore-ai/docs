# Function Differences with torch.nn.functional.binary_cross_entropy

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/binary_cross_entropy.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

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

## mindspore.ops.binary_cross_entropy

```text
mindspore.ops.binary_cross_entropy(
    logits,
    labels,
    weight=None,
    reduction='mean'
) -> Tensor
```

For more information, see [mindspore.ops.binary_cross_entropy](https://mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.binary_cross_entropy.html).

## Differences

PyTorch: Compute the binary cross-entropy loss value between the target and predicted values.

MindSpore: MindSpore API basically implements the same function as PyTorch.

| Categories | Subcategories | PyTorch | MindSpore | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | input     | logits    | Same function, different parameter names                  |
|      | Parameter 2 | target    | labels    | Same function, different parameter names                 |
|      | Parameter 3 | weight    | weight    | Same function  |
|      | Parameter 4 | size_average    | -    | PyTorch deprecated parameters, functionally replaced by the reduction parameter          |
|      | Parameter 5 | reduce    | -    | PyTorch deprecated parameters, functionally replaced by the reduction parameter        |
|      | Parameter 6 | reduction | reduction | Same function |

### Code Example 1

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
import torch.nn.functional as F
from torch import tensor

logits = tensor([0.1, 0.2, 0.3], requires_grad=True)
labels = tensor([1., 1., 1.])
loss = F.binary_cross_entropy(logits, labels)
print(loss.detach().numpy())
# 1.7053319

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor
from mindspore import ops

logits = Tensor(np.array([0.1, 0.2, 0.3]), mindspore.float32)
labels = Tensor(np.array([1., 1., 1.]), mindspore.float32)
loss = ops.binary_cross_entropy(logits, labels)
print(loss)
# 1.7053319
```
