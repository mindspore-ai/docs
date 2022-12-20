# 比较与torch.nn.functional.mse_loss的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/MSELoss_func.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.functional.mse_loss

```text
torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor
```

更多内容详见[torch.nn.functional.mse_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.mse_loss)。

## mindspore.nn.MSELoss

```text
class mindspore.nn.MSELoss(reduction='mean')(logits, labels) -> Tensor
```

更多内容详见[mindspore.nn.MSELoss](https://www.mindspore.cn/docs/zh-CN/r1.8/api_python/nn/mindspore.nn.MSELoss.html)。

## 差异对比

PyTorch：用于计算输入x和y每一个元素的均方误差，reduction参数指定应用于loss的规约类型。

MindSpore：实现与PyTorch一致的功能。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input        | logits       | 功能一致，参数名不同 |
|      | 参数2 | target       | labels      | 功能一致，参数名不同 |
|      | 参数3 | size_average | -        | 被reduction替代|
| | 参数4 | reduce | - | 被reduction替代 |
| | 参数5 | reduction | reduction | - |

### 代码示例1

> 计算input和target的均方误差。

```python
# PyTorch
import torch
from torch.nn.functional import mse_loss
from torch import tensor
import numpy as np

# 默认情况，reduction='mean'
input_ = np.array([1, 1, 1, 1]).reshape((2, 2))
input = tensor(input_, dtype=torch.float32)
target_ = np.array([1, 2, 2, 1]).reshape((2, 2))
target = tensor(target_, dtype=torch.float32)
output = mse_loss(input, target)
print(output.numpy())
# 0.5

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

loss = nn.MSELoss()
input_ = np.array([1, 1, 1, 1]).reshape((2, 2))
input = Tensor(input_, dtype=mindspore.float32)
target_ = np.array([1, 2, 2, 1]).reshape((2, 2))
target = Tensor(target_, dtype=mindspore.float32)
output = loss(input, target)
print(output)
# 0.5
```

### 代码示例2

> 计算`input`和`target`的均方误差，以求和方式规约。

```python
# PyTorch
import torch
from torch.nn.functional import mse_loss
from torch import tensor
import numpy as np

# redcution='sum'
input_ = np.array([1, 1, 1, 1]).reshape((2, 2))
input = tensor(input_, dtype=torch.float32)
target_ = np.array([1, 2, 2, 1]).reshape((2, 2))
target = tensor(target_, dtype=torch.float32)
print(target)
output = mse_loss(input, target, reduction='sum')
print(output.numpy())
# 2.0


# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

loss = nn.MSELoss(reduction='sum')
input_ = np.array([1, 1, 1, 1]).reshape((2, 2))
input = Tensor(input_, dtype=mindspore.float32)
target_ = np.array([1, 2, 2, 1]).reshape((2, 2))
target = Tensor(target_, dtype=mindspore.float32)
output = loss(input, target)
print(output)
# 2.0
```
