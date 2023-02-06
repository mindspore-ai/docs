# 比较与torch.nn.functional.binary_cross_entropy的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/BCELoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

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

更多内容详见[torch.nn.functional.binary_cross_entropy](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.binary_cross_entropy)。

## mindspore.nn.BCELoss

```text
class mindspore.nn.BCELoss(
    weight=None,
    reduction='none'
)(logits, labels) -> Tensor
```

更多内容详见[mindspore.nn.BCELoss](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.BCELoss.html)。

## 差异对比

PyTorch：计算目标值和预测值之间的二值交叉熵损失值。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。

| 分类 | 子类  | PyTorch   | MindSpore | 差异                                                         |
| ---- | ----- | --------- | --------- | ------------------------------------------------------------ |
| 参数 | 参数1 | input     | logits    | 功能一致，参数名不同                                         |
|      | 参数2 | target    | labels    | 功能一致，参数名不同                                         |
|      | 参数3 | weight    | weight    | -                                                            |
|      | 参数4 | size_average    | -    | PyTorch的已弃用参数，功能由reduction参数取代          |
|      | 参数5 | reduce    | -    | PyTorch的已弃用参数，功能由reduction参数取代                 |
|      | 参数6 | reduction | reduction | 功能一致，默认值不同 |

### 代码示例1

> 两API实现功能一致，用法相同。

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
from mindspore import nn

loss = nn.BCELoss(reduction='mean')
logits = Tensor(np.array([0.1, 0.2, 0.3]), mindspore.float32)
labels = Tensor(np.array([1., 1., 1.]), mindspore.float32)
output = loss(logits, labels)
print(output)
# 1.7053319
```
