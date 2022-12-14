# 比较与torch.logical_xor的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/logical_xor.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source.png"></a>

## torch.logical_xor

```python
class torch.logical_xor(input, other, out=None)
```

更多内容详见 [torch.logical_xor](https://pytorch.org/docs/1.5.0/torch.html#torch.logical_xor)。

## mindspore.numpy.logical_xor

```python
class mindspore.numpy.logical_xor(x1, x2, dtype=None)
```

更多内容详见 [mindspore.numpy.logical_xor](https://mindspore.cn/docs/zh-CN/r1.10/api_python/numpy/mindspore.numpy.logical_xor.html#mindspore.numpy.logical_xor)。

## 使用方式

PyTorch: 计算给定输入张量的逐元素逻辑异或。零被视为“False”，非零被视为“True”

MindSpore: 按元素计算输入张量的逻辑异或。输入应该是bool或数据类型为bool的张量。

## 代码示例

```python
import mindspore.numpy as np
import torch

# MindSpore
x1 = np.array([True, False])
x2 = np.array([False, False])
print(np.logical_xor(x1, x2))
# [True False]
x1 = np.array([0, 1, 10, 0])
x2 = np.array([4, 0, 1, 0], dtype=bool)
print(np.logical_xor(x1, x2))
# TypeError: For primitive[LogicalOr], the input argument[x] must be a type of {Tensor[Bool],}, but got Int32.

# PyTorch
print(torch.logical_xor(torch.tensor([True, False, True]), torch.tensor([True, False, False])))
# tensor([False, False,  True])
a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
print(torch.logical_xor(a, b))
# tensor([ True,  True, False, False])
print(torch.logical_xor(a.double(), b.double()))
# tensor([ True,  True, False, False])
print(torch.logical_xor(a.double(), b))
# tensor([ True,  True, False, False])
print(torch.logical_xor(a, b, out=torch.empty(4, dtype=torch.bool)))
# tensor([ True,  True, False, False])
```
