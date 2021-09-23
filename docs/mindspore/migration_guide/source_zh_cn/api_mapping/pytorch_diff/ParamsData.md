# 比较与torch.nn.Parameter.data的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/ParamsData.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## torch.nn.Parameter.data

```python
torch.nn.Parameter.data
```

更多内容详见[torch.nn.Parameter.data](https://pytorch.org/docs/1.5.0/nn.Parameter.html#torch.nn.Parameter.data)。

## mindspore.Parameter.data

```python
mindspore.Parameter.data
```

更多内容详见[mindspore.Parameter.data](https://mindspore.cn/docs/api/zh-CN/r1.5/api_python/mindspore/mindspore.Parameter.html#mindspore.Parameter.data)。

## 使用方式

PyTorch: 返回Tensor。可以直接使用`torch.nn.Parameter.data`对Parameter进行赋值。

MindSpore：返回parameter实例，包括它的四个属性，`name`、`shape`、`dtype`、`requires_grad`。使用`set_data`接口对Parameter进行赋值。

## 代码示例

```python
from mindspore import Parameter, Tensor
import mindspore
import numpy as np

x = Parameter(Tensor(np.ones([2, 2]), mindspore.float32), name="x", requires_grad=True)
print(x.data)
print(Tensor(x), "\n")

x.set_data(data=Tensor(np.zeros([2, 2]), mindspore.float32))
print(x.data)
print(Tensor(x))

# out
# Parameter (name=x, shape=(2, 2), dtype=Float32, requires_grad=True)
# [[1. 1.]
#  [1. 1.]]
# Parameter (name=x, shape=(2, 2), dtype=Float32, requires_grad=True)
# [[0. 0.]
#  [0. 0.]]
```

```python
from torch import nn
import torch

x = nn.Parameter(torch.ones(2, 2))
print(x.data)

x.data = torch.zeros(2, 2)
print(x.data)

# out
# tensor([[1., 1.],
#         [1., 1.]])
# tensor([[0., 0.],
#         [0., 0.]])
```
