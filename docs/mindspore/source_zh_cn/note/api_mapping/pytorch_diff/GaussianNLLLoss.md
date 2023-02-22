# 比较与torch.nn.GaussianNLLLoss的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/GaussianNLLLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.GaussianNLLLoss

```text
class torch.nn.GaussianNLLLoss(
    *,
    full=False,
    eps=1e-06,
    reduction='mean'
)(input, target, var) -> Tensor/Scalar
```

更多内容详见[torch.nn.GaussianNLLLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.GaussianNLLLoss.html)。

## mindspore.nn.GaussianNLLLoss

```text
class mindspore.nn.GaussianNLLLoss(
    *,
    full=False,
    eps=1e-06,
    reduction='mean'
)(logits, labels, var) -> Tensor/Scalar
```

更多内容详见[mindspore.nn.GaussianNLLLoss](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.GaussianNLLLoss.html)。

## 差异对比

PyTorch：服从高斯分布的负对数似然损失。

MindSpore：与PyTorch实现同样的功能。如果var中存在小于0的数字，PyTorch会直接报错，而MindSpore则会计算max(var, eps)
之后，将结果传给log进行计算。

| 分类  | 子类  | PyTorch   | MindSpore | 差异         |
|-----|-----|-----------|-----------|------------|
| 参数  | 参数1 | full      | full      | 功能一致       |
|     | 参数2 | eps       | eps       | 功能一致       |
|     | 参数3 | reduction | reduction | 功能一致       |
| 输入  | 输入1 | input     | logits    | 功能一致，参数名不同 |
|     | 输入2 | target    | labels    | 功能一致，参数名不同 |
|     | 输入3 | var       | var       | 功能一致       |

### 代码示例

> 两API实现功能和使用方法基本相同，但PyTorch和MindSpore针对输入 `var<0` 的情况做了不同处理。

```python
# PyTorch
import torch
from torch import nn
import numpy as np

arr1 = np.arange(8).reshape((4, 2))
arr2 = np.array([2, 3, 1, 4, 6, 4, 4, 9]).reshape((4, 2))
logits = torch.tensor(arr1, dtype=torch.float32)
labels = torch.tensor(arr2, dtype=torch.float32)
loss = nn.GaussianNLLLoss(reduction='mean')
var = torch.tensor(np.ones((4, 1)), dtype=torch.float32)
output = loss(logits, labels, var)
# tensor(1.4375)

# 如果var中有小于0的元素，PyTorch会直接报错
var[0] = -1
output2 = loss(logits, labels, var)
# ValueError: var has negative entry/entries

# MindSpore
import numpy as np
from mindspore import Tensor
import mindspore.nn as nn
from mindspore import dtype as mstype

arr1 = np.arange(8).reshape((4, 2))
arr2 = np.array([2, 3, 1, 4, 6, 4, 4, 9]).reshape((4, 2))
logits = Tensor(arr1, mstype.float32)
labels = Tensor(arr2, mstype.float32)
loss = nn.GaussianNLLLoss(reduction='mean')
var = Tensor(np.ones((4, 1)), mstype.float32)
output = loss(logits, labels, var)
print(output)
# 1.4374993

# 如果var中有小于0的元素，MindSpore会使用max(var, eps)的结果
var[0] = -1
output2 = loss(logits, labels, var)
print(output2)
# 499999.22
```
