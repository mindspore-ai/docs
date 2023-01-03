# 比较与torch.exp的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/exp.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.exp

```text
torch.exp(input, *, out=None) -> Tensor
```

更多内容详见[torch.exp](https://pytorch.org/docs/1.8.1/generated/torch.exp.html)。

## mindspore.ops.exp

```text
mindspore.ops.exp(x) -> Tensor
```

更多内容详见[mindspore.ops.exp](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.exp.html)。

## 差异对比

PyTorch：逐元素计算输入张量`input`的指数。

MindSpore：MindSpore此API实现功能与PyTorch一致，仅参数名不同。

| 分类 | 子类 | PyTorch | MindSpore | 差异 |
| :-: | :-: | :-: | :-: |:-:|
|参数 | 参数1 | input | x | 功能一致，参数名不同 |
| | 参数2 | out | - |不涉及 |

### 代码示例1

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
from torch import tensor

x = tensor([[0, 1, 2], [0, -1, -2]], dtype=torch.float32)
out = torch.exp(x).numpy()
print(out)
# [[1.         2.7182817  7.389056  ]
#  [1.         0.36787945 0.13533528]]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([[0, 1, 2], [0, -1, -2]]), mindspore.float32)
output = mindspore.ops.exp(x)
print(output)
# [[1.         2.718282   7.3890557 ]
#  [1.         0.36787948 0.13533528]]
```

### 代码示例2

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
from torch import tensor
import math

x = tensor([-1, 1, math.log(2.0)], dtype=torch.float32)
out = torch.exp(x).numpy()
print(out)
# [0.36787945 2.7182817  2.        ]

# MindSpore
import mindspore
from mindspore import Tensor
import math

x = Tensor([-1, 1, math.log(2.0)], mindspore.float32)
output = mindspore.ops.exp(x)
print(output)
# [0.36787948 2.718282   2.        ]
```
