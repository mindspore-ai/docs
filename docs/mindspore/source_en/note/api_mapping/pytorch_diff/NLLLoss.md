# Function Differences with torch.nn.NLLLoss

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/pytorch_diff/NLLLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.NLLLoss

```python
torch.nn.NLLLoss(
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction='mean'
)
```

For more information, see [torch.nn.NLLLoss](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.NLLLoss).

## mindspore.ops.NLLLoss

```python
class mindspore.ops.NLLLoss(
    reduction='mean'
)(logits, labels, weight)
```

For more information, see [mindspore.ops.NLLLoss](https://mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.NLLLoss.html#mindspore.ops.NLLLoss).

## Differences

PyTorch: Supports both 2-dimensional (N, C) input data and n-dimensional (N, C, d1, d2, ..., dK) input data.

MindSpore: Supports only 2-dimensional (N, C) input data.

Migration advice: If you need MindSpore NLLLoss operator to calculate on input of higher dimensions, separate data in dimensions higher than 2, calculate each piece of data with NLLLoss operator, then pack the outputs together.

## Code Example

```python
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore
m = nn.LogSoftmax(axis=1)
loss = ops.NLLLoss()
input = Tensor(np.random.randn(3, 5), mindspore.float32)
labels = Tensor([1, 0, 4], mindspore.int32)
weight = Tensor(np.random.rand(5), mindspore.float32)
loss, weight = loss(m(input), labels, weight)
print(loss)
# Out:
# 1.3557988


# In PyTorch
m = torch.nn.LogSoftmax(dim=1)
loss = torch.nn.NLLLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.tensor([1, 0, 4])
output = loss(m(input), target)
output.backward()
print(output)
# Out：
# tensor(1.7451, grad_fn=<NllLossBackward>)
```
