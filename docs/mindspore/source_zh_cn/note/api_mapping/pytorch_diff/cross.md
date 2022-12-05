# 比较与torch.cross的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/cross.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.cross

``` text
torch.cross(input, other, dim=None, *, out=None) -> Tensor
```

更多内容详见[torch.cross](https://pytorch.org/docs/1.8.1/generated/torch.cross.html)。

## mindspore.ops.cross

``` text
mindspore.ops.cross(input, other, dim=None) -> Tensor
```

更多内容详见[mindspore.ops.cross](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cross.html)。

## 差异对比

PyTorch：返回 input 和 other 两个向量组的叉积。

MindSpore：MindSpore此API实现功能与PyTorch一致。

| 分类 | 子类  | PyTorch | MindSpore | 差异                 |
| ---- | ----- | ------- | --------- | -------------------- |
| 参数 | 参数1 | input   | input        | 功能一致，参数名不同 |
|      | 参数2 | other   | other        | 功能一致，参数名不同 |
|      | 参数3 | dim     | dim       | -                    |
|      | 参数4 | out     | -       | 不涉及                  |

### 代码示例

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
from torch import tensor

a = tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
b = tensor([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
c = torch.cross(a, b).detach().numpy()
print(c)
# [[-1 -1 -1]
#  [-1 -2 -3]
#  [ 1  2  3]]

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype
import mindspore.ops as ops

a = Tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]], mstype.int8)
b = Tensor([[1, 2, 3], [3, 4, 5], [4, 5, 6]], mstype.int8)
c = ops.cross(a, b)
print(c)
# [[-1 -1 -1]
#  [-1 -2 -3]
#  [ 1  2  3]]
```