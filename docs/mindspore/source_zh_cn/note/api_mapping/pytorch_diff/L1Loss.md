# 比较与torch.nn.L1Loss的功能差异

## torch.nn.L1Loss

``` text
torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')(input, target) -> Tensor
```

更多内容详见[torch.nn.L1Loss](https://pytorch.org/docs/1.8.1/generated/torch.nn.L1Loss.html#torch.nn.L1Loss)。

## mindspore.nn.L1Loss

```text
mindspore.nn.L1Loss(reduction='mean')(logits, labels) -> Tensor
```

更多内容详见[MindSpore.nn.L1Loss](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.L1Loss.html)。

## 差异对比

PyTorch:L1Loss用于计算预测值和目标值之间的平均绝对误差。

MindSpore:包含PyTorch功能，当logits和labels的shape不同但可以互相传播时，仍可运行，PyTorch不可以。

| 分类 | 子类  | PyTorch   | MindSpore | 差异                 |
| ---- | ----- | --------- | --------- | -------------------- |
| 参数 | 参数1 | size_average     | -    | 已弃用，功能由reduction接替 |
|      | 参数2 | reduce    | -    | 已弃用，功能由reduction接替|
|      | 参数3 | reduction | reduction | - |
|      | 参数4 | input     | logits    | 功能一致，参数名不同 |
|      | 参数5 | target    | labels    | 功能一致，参数名不同 |

### 代码示例1

> 两API功能一致，用法相同。

```python
# PyTorch
import torch
import torch.nn as nn

loss = nn.L1Loss()
input = torch.tensor([2,2,3], dtype=torch.float32)
target = torch.tensor([1,2,2], dtype=torch.float32)
output = loss(input, target)
output = output.detach().numpy()
print(output)
# 0.6666667

# MindSpore
import mindspore
from mindspore import Tensor, nn
import numpy as np

loss = nn.L1Loss()
logits = Tensor(np.array([2, 2, 3]), mindspore.float32)
labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
output = loss(logits, labels)
print(output)
# 0.6666667
```
