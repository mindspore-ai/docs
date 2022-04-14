# 比较与torch.nn.NLLLoss的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/NLLLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

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

更多内容详见[torch.nn.NLLLoss](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.NLLLoss)。

## mindspore.ops.NLLLoss

```python
class mindspore.ops.NLLLoss(
    reduction='mean'
)(logits, labels, weight)
```

更多内容详见[mindspore.ops.NLLLoss](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.NLLLoss.html#mindspore.ops.NLLLoss)。

## 使用方式

PyTorch：同时支持二维数据 (N, C) 和多维数据(N, C, d1, d2, ..., dK)。

MindSpore：仅支持二维数据 (N, C)。

迁移建议：如需要处理高维度输入数据，可以自行封装将d1, d2, ..., dK维度拆分计算loss后再拼接的NLLLoss接口。

## 代码示例

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
