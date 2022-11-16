# 比较与torch.eq的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/equal.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.eq

```text
torch.eq(input, other) -> Tensor
```

更多内容详见 [torch.eq](https://pytorch.org/docs/1.8.1/generated/torch.eq.html)。

## mindspore.ops.equal

```text
mindspore.ops.equal(x, y) -> Tensor
```

更多内容详见 [mindspore.ops.equal](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.equal.html)。

## 差异对比

PyTorch：逐元素比较两个输入Tensor是否相等。

MindSpore：MindSpore此API实现功能与PyTorch一致，仅参数名不同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| :-: | :-: | :-: | :-: |:-:|
|参数 | 参数1 | input | x |功能一致，参数名不同 |
| | 参数2 | other | y |功能一致，参数名不同 |

### 代码示例1

> 实现功能一致，用法相同。

```python
# PyTorch
import torch
from torch import tensor

input1 = tensor([1, 2], dtype=torch.float32)
other = tensor([[1, 2], [0, 2], [1, 3]], dtype=torch.int64)
out = torch.eq(input1, other).numpy()
print(out)
# [[ True  True]
#  [False  True]
#  [ True False]]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([1, 2]), mindspore.float32)
y = Tensor(np.array([[1, 2], [0, 2], [1, 3]]), mindspore.int64)
output = mindspore.ops.equal(x, y)
print(output)
# [[ True  True]
#  [False  True]
#  [ True False]]
```

### 代码示例2

> 实现功能一致，用法相同。

```python
# PyTorch
import torch
from torch import tensor

input1 = tensor([1, 3, 1, 4], dtype=torch.int32)
out = torch.eq(input1, 1).numpy()
print(out)
# [ True False  True False]

# MindSpore
import mindspore
from mindspore import Tensor

x = Tensor([1, 3, 1, 4], mindspore.int32)
output = mindspore.ops.equal(x, 1)
print(output)
# [ True False  True False]
```
