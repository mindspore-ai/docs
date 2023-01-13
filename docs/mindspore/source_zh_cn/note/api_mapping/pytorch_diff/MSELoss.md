# 比较与torch.nn.MSELoss的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/MSELoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.MSELoss

```text
torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')(input, target) -> Tensor
```

更多内容详见[torch.nn.MSELoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.MSELoss.html)。

## mindspore.nn.MSELoss

```text
class mindspore.nn.MSELoss(reduction='mean')(logits, labels) -> Tensor
```

更多内容详见[mindspore.nn.MSELoss](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.MSELoss.html)。

## 差异对比

PyTorch：用于计算输入input和target每一个元素的均方误差，reduction参数指定应用于loss的规约类型。

MindSpore：实现与PyTorch一致的功能。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 | size_average | -        | 已弃用，被reduction替代 |
| | 参数2 | reduce | - |  已弃用，被reduction替代 |
| | 参数3 | reduction | reduction | - |
|输入 | 输入1 | input        | logits       | 功能一致，参数名不同 |
|      | 输入2 | target       | labels      | 功能一致，参数名不同 |

### 代码示例1

> 计算`input`和`target`的均方误差。默认情况下，`reduction='mean'`。

```python
# PyTorch
import torch
from torch import nn
from torch import tensor
import numpy as np

loss = nn.MSELoss()
input_ = np.array([1, 1, 1, 1]).reshape((2, 2))
inputs = tensor(input_, dtype=torch.float32)
target_ = np.array([1, 2, 2, 1]).reshape((2, 2))
target = tensor(target_, dtype=torch.float32)
output = loss(inputs, target)
print(output.numpy())
# 0.5

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

loss = nn.MSELoss()
input_ = np.array([1, 1, 1, 1]).reshape((2, 2))
inputs = Tensor(input_, dtype=mindspore.float32)
target_ = np.array([1, 2, 2, 1]).reshape((2, 2))
target = Tensor(target_, dtype=mindspore.float32)
output = loss(inputs, target)
print(output)
# 0.5
```

### 代码示例2

> 计算`input`和`target`的均方误差，以求和方式规约。

```python
# PyTorch
import torch
from torch import nn
from torch import tensor
import numpy as np

loss = nn.MSELoss(reduction='sum')
input_ = np.array([1, 1, 1, 1]).reshape((2, 2))
inputs = tensor(input_, dtype=torch.float32)
target_ = np.array([1, 2, 2, 1]).reshape((2, 2))
target = tensor(target_, dtype=torch.float32)
output = loss(inputs, target)
print(output.numpy())
# 2.0

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

loss = nn.MSELoss(reduction='sum')
input_ = np.array([1, 1, 1, 1]).reshape((2, 2))
inputs = Tensor(input_, dtype=mindspore.float32)
target_ = np.array([1, 2, 2, 1]).reshape((2, 2))
target = Tensor(target_, dtype=mindspore.float32)
output = loss(inputs, target)
print(output)
# 2.0
```
