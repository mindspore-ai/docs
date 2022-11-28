# 比较与torch.abs的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/abs.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.abs

```text
torch.abs(input, *, out=None) -> Tensor
```

更多内容详见 [torch.abs](https://pytorch.org/docs/1.8.1/generated/torch.abs.html)。

## mindspore.ops.abs

```text
mindspore.ops.abs(x) -> Tensor
```

更多内容详见 [mindspore.ops.abs](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.abs.html)。

## 差异对比

PyTorch：计算输入的绝对值。

MindSpore: MindSpore此API实现功能与PyTorch一致，仅参数名不同。

| 分类 | 子类  | PyTorch | MindSpore | 差异                  |
| ---- | ----- | ------- | --------- | --------------------- |
| 参数 | 参数1 | input   | x | 功能一致，参数名不同 |
|  | 参数2 | out | - | 不涉及 |

### 代码示例1

两API实现功能一致， 用法相同。

```python
# PyTorch
import torch
from torch import tensor

input = torch.tensor([-1, 1, 0], dtype=torch.float32)
output = torch.abs(input).numpy()
print(output)
# [1. 1. 0.]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.array([-1, 1, 0]), mindspore.float32)
output = ops.abs(x).asnumpy()
print(output)
# [1. 1. 0.]
```

