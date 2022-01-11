# 比较与torch.logical_and的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/LogicalAnd.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## torch.logical_and

```python
class torch.logical_and(input, other, out=None)
```

更多内容详见 [torch.logical_and](https://pytorch.org/docs/1.5.0/torch.html#torch.logical_and)。

## mindspore.ops.LogicalAnd

```python
class class mindspore.ops.LogicalAnd()(x, y)
```

更多内容详见 [mindspore.ops.LogicalAnd](https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.LogicalAnd.html#mindspore.ops.LogicalAnd)。

## 使用方式

PyTorch: 计算给定输入张量的逐元素逻辑与。零被视为“False”，非零被视为“True”。

MindSpore: 按元素计算两个输入张量的逻辑与。输入可以是bool值或数据类型为bool的张量。

## 代码示例

```python
import numpy as np
import torch
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype

# MindSpore
x = Tensor(np.array([True, False, True]), mstype.bool_)
y = Tensor(np.array([True, True, False]), mstype.bool_)
logical_and = ops.LogicalAnd()
print(logical_and(x, y))
# [ True False False]
x = Tensor(np.array([True, False, True]), mstype.bool_)
y = Tensor(np.array([True, True, False]), mstype.int32)
logical_and = ops.LogicalAnd()
print(logical_and(x, y))
# TypeError: For primitive[LogicalAnd], the input argument[x, y, ] must be a type of {Tensor[Bool],}, but got Int32.

# Pytorch
print(torch.logical_and(torch.tensor([True, False, True]), torch.tensor([True, False, False])))
# tensor([ True, False, False])
a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
print(torch.logical_and(a, b))
# tensor([False, False,  True, False])
print(torch.logical_and(a.double(), b.double()))
# tensor([False, False,  True, False])
print(torch.logical_and(a.double(), b))
# tensor([False, False,  True, False])
print(torch.logical_and(a, b, out=torch.empty(4, dtype=torch.bool)))
# tensor([False, False,  True, False])
```
