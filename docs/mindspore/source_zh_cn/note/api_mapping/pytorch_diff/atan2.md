# 比较与torch.atan2的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/atan2.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.atan2

```text
torch.atan2(input, other, *, out=None) -> Tensor
```

更多内容详见[torch.atan2](https://pytorch.org/docs/1.8.1/generated/torch.atan2.html)。

## mindspore.ops.atan2

```text
mindspore.ops.atan2(input, other) -> Tensor
```

更多内容详见[mindspore.ops.atan2](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.atan2.html)。

## 差异对比

PyTorch：逐元素计算考虑象限的input/other的反正切值，其中第二个参数other是x坐标，第一个参数input是y坐标。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，不过也支持`input`或者`other`为Scalar的输入。

| 分类 | 子类  | PyTorch | MindSpore | 差异                 |
| ---- | ----- | ------- | --------- | -------------------- |
| 参数 | 参数1 | input   | input         | 无差异 |
|      | 参数2 | other   | other         | 无差异 |
|      | 参数3 | out     | -         | 不涉及               |

### 代码示例1

当输入的`input`和`other`均为Tensor的时候，两API实现相同的功能。

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

input = Tensor(np.array([0, 1]), mindspore.float32)
other = Tensor(np.array([1, 1]), mindspore.float32)

output = ops.atan2(input, other)
print(output)
# [0.        0.7853982]
```

### 代码示例2

说明：当输入的`input`或`other`是Scalar的时候，MindSpore能实现对应功能，pytorch不支持。

```python
# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

input = 1
other = Tensor(np.array([1, 1]), mindspore.float32)

output = ops.atan2(input, other)
print(output)
# [0.7853982 0.7853982]
```

