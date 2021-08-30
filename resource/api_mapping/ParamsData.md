# 比较与torch.nn.Parameter.data的功能差异

## torch.nn.Parameter.data

```python
torch.nn.Parameter.data
```

## mindspore.Parameter.data

```python
mindspore.Parameter.data
```

## 使用方式

PyTorch: 返回Tensor。可以直接使用`torch.nn.Parameter.data`对Parameter进行赋值。

MindSpore：返回parameter实例，包括它的四个属性，`name`、`shape`、`dtype`、`requires_grad`。使用`set_data`接口对Parameter进行赋值。

## 代码示例

```python
from mindspore import Parameter, Tensor
import mindspore
import numpy as np

x = Parameter(Tensor(np.random.rand(3, 4), mindspore.float32), name="x", requires_grad=True)
print(x.data)
print(Tensor(x), "\n")

x.set_data(data=Tensor(np.random.rand(3, 4), mindspore.float32))
print(x.data)
print(Tensor(x))

# out
Parameter (name=x, shape=(3, 4), dtype=Float32, requires_grad=True)
[[0.31407008 0.48725787 0.9409604  0.95889574]
 [0.03603205 0.870293   0.421732   0.02004708]
 [0.91354364 0.4765727  0.7693372  0.19445552]]

Parameter (name=x, shape=(3, 4), dtype=Float32, requires_grad=True)
[[0.4505818  0.15254559 0.75569516 0.94943196]
 [0.26312852 0.39158103 0.92813534 0.8305332 ]
 [0.6042876  0.6962918  0.22445041 0.33978763]]
```

```python
from torch import nn
import torch

x = nn.Parameter(torch.randn(3, 4))
print(x.data)

x.data = torch.randn(3, 4)
print(x.data)

# out:
tensor([[-0.6071,  0.1344, -1.7702,  2.6501],
        [-0.1849, -1.1737,  0.4720, -1.2044],
        [ 1.6881,  0.7720, -0.5907, -0.1857]])
tensor([[ 0.3496,  0.1992, -0.2829,  0.8311],
        [-0.0958,  0.2017, -1.6202,  0.8625],
        [-0.3115, -0.3021,  0.1432, -0.8853]])
```
