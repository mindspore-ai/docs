# 比较与torch.cross的功能差异

## torch.cross

``` text
torch.cross(input, other, dim=None, *, out=None) -> Tensor
```

更多内容详见 [torch.cross](https://pytorch.org/docs/1.8.1/generated/torch.cross.html)。

## mindspore.ops.cross

``` text
mindspore.ops.cross(input, other, dim=None) -> Tensor
```

更多内容详见 [mindspore.ops.cross](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cross.html)。

## 差异对比

Pytorch：返回 input 和 other 两个向量组的叉积。

MindSpore：MindSpore此API实现功能与PyTorch一致。

| 分类 | 子类  | Pytorch | MindSpore | 差异                 |
| ---- | ----- | ------- | --------- | -------------------- |
| 参数 | 参数1 | input   | input        | 功能一致，参数名不同 |
|      | 参数2 | other   | other        | 功能一致，参数名不同 |
|      | 参数3 | dim     | dim       | -                    |

### 代码示例

> 两API实功能一致，用法相同。

```python
# PyTorch
import torch
from torch import tensor

a = tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
b = tensor([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
c = torch.cross(a, b).detach().numpy()
d = torch.cross(a, b, dim = 1).detach().numpy()
print(c)
print(d)
# [[-1, -1, -1],
#  [-1, -2, -3],
#  [ 1,  2,  3]]
# [[ 0,  0,  0],
#  [-1,  2, -1],
#  [-1,  2, -1]]

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype
import mindspore.ops as ops

a = Tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]], mstype.int8)
b = Tensor([[1, 2, 3], [3, 4, 5], [4, 5, 6]], mstype.int8)
c = ops.cross(a, b)
d = ops.cross(a, b, dim = 1)
print(c)
print(d)
# [[-1, -1, -1],
#  [-1, -2, -3],
#  [ 1,  2,  3]]
# [[ 0,  0,  0],
#  [-1,  2, -1],
#  [-1,  2, -1]]
```