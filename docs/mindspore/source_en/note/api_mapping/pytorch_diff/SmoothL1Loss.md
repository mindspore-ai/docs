# Function Differences with torch.nn.SmoothL1Loss

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/SmoothL1Loss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.SmoothL1Loss

```text
class torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)(input, target) -> Tensor/Scalar
```

For more information, see [torch.nn.SmoothL1Loss](https://pytorch.org/docs/1.8.1/generated/torch.nn.SmoothL1Loss.html).

## mindspore.nn.SmoothL1Loss

```text
class mindspore.nn.SmoothL1Loss(beta=1.0, reduction='none')(logits, labels) -> Tensor/Scalar
```

For more information, see [mindspore.nn.SmoothL1Loss](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.SmoothL1Loss.html).

## Differences

PyTorch: SmoothL1Loss loss function. If the element-wise absolute error between the predicted and target values is less than a set threshold beta, use a squared term, otherwise with an absolute error term.

MindSpore: There are no functional differences except for two parameters that have been deprecated in PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters| Parameter 1 | size_average | -         | Deprecated, replaced by reduction. MindSpore does not have this parameter |
| | Parameter 2 | reduce | - | Deprecated, replaced by reduction. MindSpore does not have this parameter |
| | Parameter 3 | reduction | reduction | - |
| | Parameter 4 | beta         | beta      | -                                        |
| | Parameter 5 | input | logits | Same function, different parameter names|
| | Parameter 6 | target | labels | Same function, different parameter names|

### Code Example 1

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
import torch.nn as nn

beta = 1
loss = nn.SmoothL1Loss(reduction="none", beta=beta)
logits = torch.FloatTensor([1, 2, 3])
labels = torch.FloatTensor([1, 2, 2])
output = loss(logits, labels)
print(output.numpy())
# [0.  0.  0.5]

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

loss = mindspore.nn.SmoothL1Loss()
logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
output = loss(logits, labels)
print(output)
# [0.  0.  0.5]
```

### Code Example 2

> The two APIs achieve the same function and have the same usage.

```python
import torch
import torch.nn as nn

beta = 1
loss = nn.SmoothL1Loss(reduction="none", beta=beta)
logits = torch.FloatTensor([1, 1, 1, 1])
labels = torch.FloatTensor([0, 2, 3, 4])
output = loss(logits, labels)
print(output.numpy())
# [0.5 0.5 1.5 2.5]

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

loss = mindspore.nn.SmoothL1Loss()
logits = mindspore.Tensor(np.array([1, 1, 1, 1]), mindspore.float32)
labels = mindspore.Tensor(np.array([0, 2, 3, 4]), mindspore.float32)
output = loss(logits, labels)
print(output)
# [0.5 0.5 1.5 2.5]
```
