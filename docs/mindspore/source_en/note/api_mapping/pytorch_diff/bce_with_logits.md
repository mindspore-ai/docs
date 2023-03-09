# Function Differences with torch.nn.functional.binary_cross_entropy_with_logits

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/bce_with_logits.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.functional.binary_cross_entropy_with_logits

```text
torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)

```

For more information, see [torch.nn.functional.binary_cross_entropy_with_logits](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.binary_cross_entropy_with_logits).

## mindspore.nn.BCELoss

```text
mindspore.ops.binary_cross_entropy_with_logits(logits, label, weight, pos_weight, reduction='mean')
```

For more information, see [mindspore.ops.binary_cross_entropy_with_logits](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.binary_cross_entropy_with_logits.html).

## Differences

PyTorch: Compute the binary cross-entropy loss value between the target and predicted values.

MindSpore: MindSpore API basically implements the same function as PyTorch, but not set default value of `weight` and `pos_weight`.

| Categories | Subcategories | PyTorch | MindSpore | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | input     | logits    | Same function, different parameter names                  |
|      | Parameter 2 | target    | label    | Same function, different parameter names                 |
|      | Parameter 3 | weight    | weight    | default value is not set  |
|      | Parameter 4 | size_average    | -    | PyTorch deprecated parameters, functionally replaced by the reduction parameter          |
|      | Parameter 5 | reduce    | -    | PyTorch deprecated parameters, functionally replaced by the reduction parameter        |
|      | Parameter 6 | reduction | reduction | Same function, different default values. |
|      | Parameter 7 | pos_weight    | pos_weight    | default value is not set  |

### Code Example 1

```python
import numpy as np
import mindspore
from mindspore import Tensor

logits = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]), mindspore.float32)
label = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]), mindspore.float32)
weight = Tensor(np.array([1.0, 1.0, 1.0]), mindspore.float32)
pos_weight = Tensor(np.array([1.0, 1.0, 1.0]), mindspore.float32)
output = ops.binary_cross_entropy_with_logits(logits, label, weight, pos_weight)
print(output)
# 0.34636116

import torch

logits = torch.tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]))
label = torch.tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]))
output = torch.nn.functional.binary_cross_entropy_with_logits(logits, label)
print(output)
# tensor(0.3464, dtype=torch.float64)
```
