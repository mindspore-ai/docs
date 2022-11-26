# 比较与torch.nn.functional.binary_cross_entropy的功能差异

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

更多内容详见 [torch.nn.functional.binary_cross_entropy](https://pytorch.org/docs/1.8.1/nn.functional.html)。

## mindspore.nn.BCELoss

```text
class mindspore.nn.BCELoss(
    weight=None,
    reduction='none'
)(logits, labels) -> Tensor
```

更多内容详见 [mindspore.nn.BCELoss](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.BCELoss.html)。

## 差异对比

PyTorch：计算目标值和预测值之间的二值交叉熵损失值。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。

| 分类 | 子类  | PyTorch   | MindSpore | 差异                                                         |
| ---- | ----- | --------- | --------- | ------------------------------------------------------------ |
| 参数 | 参数1 | input     | logits    | 功能一致，参数名不同                                         |
|      | 参数2 | target    | labels    | 功能一致，参数名不同                                         |
|      | 参数3 | weight    | weight    | -                                                            |
|      | 参数4 | reduction | reduction | 功能一致，指定输出结果的计算方式，PyTorch默认值是"mean"，MindSpore默认值是none |

## 代码示例1

> 两API实功能一致， 用法相同。

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