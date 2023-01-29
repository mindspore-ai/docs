# 比较与torch.atan的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/atan.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.atan

```text
torch.atan(input, *, out=None) -> Tensor
```

更多内容详见[torch.atan](https://pytorch.org/docs/1.8.1/generated/torch.atan.html)。

## mindspore.ops.atan

```text
mindspore.ops.atan(x) -> Tensor
```

更多内容详见[mindspore.ops.atan](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.atan.html)。

## 差异对比

PyTorch：逐元素计算输入Tensor的反正切值。

MindSpore：MindSpore此API实现功能与PyTorch一致，仅参数名不同。

| 分类 | 子类  | PyTorch | MindSpore | 差异                 |
| ---- | ----- | ------- | --------- | -------------------- |
| 参数 | 参数1 | input   | x         | 功能一致，参数名不同 |
|      | 参数2 | out     | -         | 不涉及               |

### 代码示例1

两API实现功能一致，用法相同。

```python
# PyTorch
import numpy as np
import torch
from torch import tensor

input = torch.tensor(np.array([0.2341, 1.0, 0.0, -0.6448]), dtype=torch.float32)
output = torch.atan(input).numpy()
print(output)
# [ 0.22995889  0.7853982   0.         -0.572711  ]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.array([0.2341, 1.0, 0.0, -0.6448]), mindspore.float32)
output = ops.atan(x)
print(output)
# [ 0.22995889  0.7853982   0.         -0.572711  ]
```
