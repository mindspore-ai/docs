# Function Differences with torch.nn.NLLLoss

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_en/note/api_mapping/pytorch_diff/NLLLoss.md)

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

For more information, see [mindspore.ops.NLLLoss](https://mindspore.cn/docs/en/r1.9/api_python/ops/mindspore.ops.NLLLoss.html#mindspore.ops.NLLLoss).

## Differences

PyTorch: Supports both 2-dimensional (N, C) input data and n-dimensional (N, C, d1, d2, ..., dK) input data.

MindSpore: Supports only 2-dimensional (N, C) input data.

Migration advice: If you need MindSpore NLLLoss operator to calculate on input of higher dimensions, separate data in dimensions higher than 2, calculate each piece of data with NLLLoss operator, then pack the outputs together.

## Code Example

```python
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore
m = nn.LogSoftmax(axis=1)
loss = ops.NLLLoss()
input = ms.Tensor(np.random.randn(3, 5), ms.float32)
labels = ms.Tensor([1, 0, 4], ms.int32)
weight = ms.Tensor(np.random.rand(5), ms.float32)
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
