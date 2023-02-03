# Function Differences with torch.nn.functional.soft_margin_loss

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/SoftMarginLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.functional.soft_margin_loss

```text
torch.nn.functional.soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean')  -> Tensor/Scalar
```

For more information, see [torch.nn.functional.soft_margin_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#soft-margin-loss).

## mindspore.nn.SoftMarginLoss

```text
class mindspore.nn.SoftMarginLoss(reduction='mean')(logits, labels)  -> Tensor/Scalar
```

For more information, see [mindspore.nn.SoftMarginLoss](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.SoftMarginLoss.html).

## Differences

PyTorch: Loss function for the binary classification problem, used to calculate the binary loss value for the input Tensor x and the target value Tensor y (containing 1 or -1).

MindSpore: There are no functional differences except for two parameters that have been deprecated in PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1 | input | logits | Same function, different parameter names|
| | Parameter 2 | target | labels | Same function, different parameter names|
| | Parameter 3 | size_average | -         | Deprecated, replaced by reduction. MindSpore does not have this parameter  |
| | Parameter 4 | reduce | - | Deprecated, replaced by reduction. MindSpore does not have this Parameter |
| | Parameter 5 | reduction | reduction | - |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
from torch import tensor
import torch.nn as nn

logits = torch.FloatTensor([[0.3, 0.7], [0.5, 0.5]])
labels = torch.FloatTensor([[-1, 1], [1, -1]])
output = torch.nn.functional.soft_margin_loss(logits, labels)
print(output.numpy())
# 0.6764238

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

loss = mindspore.nn.SoftMarginLoss()
logits = Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]), mindspore.float32)
labels = Tensor(np.array([[-1, 1], [1, -1]]), mindspore.float32)
output = loss(logits, labels)
print(output)
# 0.6764238
```

