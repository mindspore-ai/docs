# 比较与torch.nn.SmoothL1Loss的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SmoothL1Loss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.SmoothL1Loss

```text
class torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)(input, target) -> Tensor/Scalar
```

更多内容详见 [torch.nn.SmoothL1Loss](https://pytorch.org/docs/1.8.1/generated/torch.nn.SmoothL1Loss.html)。

## mindspore.nn.SmoothL1Loss

```text
class mindspore.nn.SmoothL1Loss(beta=1.0, reduction='none')(logits, labels) -> Tensor/Scalar
```

更多内容详见 [mindspore.nn.SmoothL1Loss](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.SmoothL1Loss.html)。

## 差异对比

PyTorch：SmoothL1Loss损失函数，如果预测值和目标值的逐个元素绝对误差小于设定阈值beta则用平方项，否则用绝对误差项。

MindSpore：除两个在PyTorch已弃用的参数不同外，功能上无差异。

| 分类 | 子类  | PyTorch      | MindSpore | 差异                                                         |
| ---- | ----- | ------------ | --------- | ------------------------------------------------------------ |
| 参数| 参数1 | size_average | -         | 已弃用，被reduction取代，MindSpore无此参数 |
| | 参数2 | reduce | - | 已弃用，被reduction取代，MindSpore无此参数 |
| | 参数3 | reduction | reduction | - |
| | 参数4 | beta         | beta      | -                                        |
| | 参数5 | input | logits | 功能一致，参数名不同|
| | 参数6 | target | labels | 功能一致，参数名不同|

### 代码示例1

> 两API实现功能一致，用法相同。

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
#[0.  0.  0.5]

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

loss = mindspore.nn.SmoothL1Loss()
logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
output = loss(logits, labels)
print(output)
#[0.  0.  0.5]
```

### 代码示例2

> 两API实现功能一致，用法相同。

```python
import torch
import torch.nn as nn

beta = 1
loss = nn.SmoothL1Loss(reduction="none", beta=beta)
logits = torch.FloatTensor([1, 1, 1,1])
labels = torch.FloatTensor([0, 2, 3,4])
output = loss(logits, labels)
print(output.numpy())
#[0.5 0.5 1.5 2.5]

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

loss = mindspore.nn.SmoothL1Loss()
logits = mindspore.Tensor(np.array([1, 1, 1,1]), mindspore.float32)
labels = mindspore.Tensor(np.array([0, 2, 3,4]), mindspore.float32)
output = loss(logits, labels)
print(output)
#[0.5 0.5 1.5 2.5]
```
