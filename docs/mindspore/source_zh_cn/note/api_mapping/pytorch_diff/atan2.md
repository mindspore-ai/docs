# 比较与torch.atan2的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/atan2.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.atan2

```text
torch.atan2(input, other, *, out=None) -> Tensor
```

更多内容详见[torch.atan2](https://pytorch.org/docs/1.8.1/generated/torch.atan2.html)。

## mindspore.ops.atan2

```text
mindspore.ops.atan2(x, y) -> Tensor
```

更多内容详见[mindspore.ops.atan2](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.atan2.html)。

## 差异对比

PyTorch：逐元素计算考虑象限的input/other的反正切值，其中第二个参数other是x坐标，第一个参数input是y坐标。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，不过也支持x或者y为Scalar的输入。

| 分类 | 子类  | PyTorch | MindSpore | 差异                 |
| ---- | ----- | ------- | --------- | -------------------- |
| 参数 | 参数1 | input   | x         | 功能一致，参数名不同 |
|      | 参数2 | other   | y         | 功能一致，参数名不同 |
|      | 参数3 | out     | -         | 不涉及               |

### 代码示例1

当输入的x和y均为Tenor的时候，两API实现相同的功能。

```python
# PyTorch
import numpy as np
import torch
from torch import tensor

input = torch.tensor(np.array([0, 1]), dtype=torch.float32)
other = torch.tensor(np.array([1, 1]), dtype=torch.int)
output = torch.atan2(input, other).numpy()
print(output)
# [0.        0.7853982]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.array([0, 1]), mindspore.float32)
y = Tensor(np.array([1, 1]), mindspore.float32)

output = ops.atan2(x, y)
print(output)
# [0.        0.7853982]
```

### 代码示例2

说明：当输入的x或y是Scalar的时候，MindSpore能实现对应功能，pytorch不支持。

```python
# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = 1
y = Tensor(np.array([1, 1]), mindspore.float32)

output = ops.atan2(x, y)
print(output)
# [0.7853982 0.7853982]
```

