# 比较与torch.logical_not的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/logical_not.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.logical_not

```python
class torch.logical_not(input, out=None)
```

更多内容详见 [torch.logical_not](https://pytorch.org/docs/1.5.0/torch.html#torch.logical_not)。

## mindspore.numpy.logical_not

```python
class mindspore.numpy.logical_not(a, dtype=None)
```

更多内容详见 [mindspore.numpy.logical_not](https://mindspore.cn/docs/api/zh-CN/master/api_python/numpy/mindspore.numpy.logical_not.html#mindspore.numpy.logical_not)。

## 使用方式

PyTorch: 计算给定输入张量的逐元素逻辑非。零被视为“False”，非零被视为“True”。

MindSpore: 按元素计算输入张量的逻辑非。输入应该是一个dtype为bool的张量。

## 代码示例

```python
import mindspore.numpy as np
import torch

# MindSpore
print(np.logical_not(np.array([True, False])))
# Tensor(shape=[2], dtype=Bool, value= [False,  True])
print(np.logical_not(np.array([0, 1, -10])))
# TypeError: For primitive[LogicalNot], the input argument[x] must be a type of {Tensor[Bool],}, but got Int32.

# PyTorch
print(torch.logical_not(torch.tensor([True, False])))
# tensor([False,  True])
print(torch.logical_not(torch.tensor([0, 1, -10], dtype=torch.int8)))
# tensor([ True, False, False])
print(torch.logical_not(torch.tensor([0., 1.5, -10.], dtype=torch.double)))
# tensor([ True, False, False])
print(torch.logical_not(torch.tensor([0., 1., -10.], dtype=torch.double), out=torch.empty(3, dtype=torch.int16)))
# tensor([1, 0, 0], dtype=torch.int16)
```
