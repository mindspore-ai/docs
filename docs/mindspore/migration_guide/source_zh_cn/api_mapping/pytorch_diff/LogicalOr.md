# 比较与torch.logical_or的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/LogicalOr.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## torch.logical_or

```python
class torch.logical_or(input, other, out=None)
```

更多内容详见 [torch.logical_or](https://pytorch.org/docs/1.5.0/torch.html#torch.logical_or)。

## mindspore.ops.LogicalOr

```python
class class mindspore.ops.LogicalOr(x, y)
```

更多内容详见 [mindspore.ops.LogicalOr](https://mindspore.cn/docs/api/zh-CN/r1.5/api_python/ops/mindspore.ops.LogicalOr.html#mindspore.ops.LogicalOr)。

## 使用方式

PyTorch: 计算给定输入张量的逐元素逻辑或。 零被视为“False”，非零被视为“True”。

MindSpore: 输入可以是bool值或数据类型为bool的张量。

## 代码示例

```python
import numpy as np
import torch
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype

# MindSpore
x = Tensor(np.array([True, False, True]), mstype.bool_)
y = Tensor(np.array([True, True, False]), mstype.bool_)
logical_or = ops.LogicalOr()
logical_or(x, y)
# [ True  True  True]
x = Tensor(np.array([True, False, True]), mstype.bool_)
y = Tensor(np.array([True, True, False]), mstype.int32)
logical_or = ops.LogicalOr()
logical_or(x, y)
# TypeError: For 'LogicalOr', the type of `x` should be subclass of Tensor[Bool], but got Tensor[Int32] .

# PyTorch
torch.logical_or(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
# tensor([ True, False,  True])
a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
torch.logical_or(a, b)
# tensor([ True,  True,  True, False])
torch.logical_or(a.double(), b.double())
# tensor([ True,  True,  True, False])
torch.logical_or(a.double(), b)
# tensor([ True,  True,  True, False])
torch.logical_or(a, b, out=torch.empty(4, dtype=torch.bool))
# tensor([ True,  True,  True, False])
```
