# 比较与torch.nn.MSELoss的功能差异

## torch.nn.MSELoss

```text
torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean') -> Tensor
```

更多内容详见[torch.nn.MSELoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.MSELoss.html)。

## mindspore.nn.MSELoss

```text
mindspore.nn.MSELoss(reduction='mean') -> Tensor
```

更多内容详见[mindspore.nn.MSELoss](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.MSELoss.html)。

## 差异对比

PyTorch: 用于计算输入 x 和 y 每一个元素的均方误差。reduction参数指定应用于loss的reduction类型。

MindSpore:除两个在Pytorch已弃用的参数不同外，功能上无差异。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | size_average | - |被`reduction`替代，MindSpore无此参数 |
| | 参数2 | reduce | - | 被`reduction`替代，MindSpore无此参数 |
| | 参数3 | reduction | reduction | - |

## 差异分析与示例

### 代码示例1

> 计算`input`和`target`的均方误差。

```python
# pytoch
import torch
from torch import nn
from torch import tensor
import numpy as np

# 默认情况，reduction='mean'
loss = nn.MSELoss()
input_ = np.array([1,1,1,1]).reshape((2,2))
input = tensor(input_, dtype=torch.float32)
target_ = np.array([1,2,2,1]).reshape((2,2))
target = tensor(target_, dtype=torch.float32)
output = loss(input, target)
print(output.numpy())
# 0.5

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

loss = nn.MSELoss()
input_ = np.array([1,1,1,1]).reshape((2,2))
input = Tensor(input_, dtype=mindspore.float32)
target_ = np.array([1,2,2,1]).reshape((2,2))
target = Tensor(target_, dtype=mindspore.float32)
output = loss(input, target)
print(output)
# 0.5
```

### 代码示例2

> 计算`input`和`target`的均方误差，以求和方式规约。

```python
# pytoch
import torch
from torch import nn
from torch import tensor
import numpy as np

# redcution='sum'
loss = nn.MSELoss(reduction='sum')
input_ = np.array([1,1,1,1]).reshape((2,2))
input = tensor(input_, dtype=torch.float32)
target_ = np.array([1,2,2,1]).reshape((2,2))
target = tensor(target_, dtype=torch.float32)
output = loss(input, target)
print(output.numpy())
# 2.0


# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

loss = nn.MSELoss(reduction='sum')
input_ = np.array([1,1,1,1]).reshape((2,2))
input = Tensor(input_, dtype=mindspore.float32)
target_ = np.array([1,2,2,1]).reshape((2,2))
target = Tensor(target_, dtype=mindspore.float32)
output = loss(input, target)
print(output)
# 2.0
```
