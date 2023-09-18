# 比较与torch.nn.Linear的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Dense.md)

## torch.nn.Linear

```text
class torch.nn.Linear(
    in_features,
    out_features,
    bias=True
)(input) -> Tensor
```

更多内容详见[torch.nn.Linear](https://pytorch.org/docs/1.8.1/generated/torch.nn.Linear.html)。

## mindspore.nn.Dense

```text
class mindspore.nn.Dense(
    in_channels,
    out_channels,
    weight_init=None,
    bias_init=None,
    has_bias=True,
    activation=None
)(x) -> Tensor
```

更多内容详见[mindspore.nn.Dense](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.Dense.html)。

## 差异对比

PyTorch：全连接层，实现矩阵相乘的运算。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，而且可以在全连接层后添加激活函数。

### 权重初始化差异

mindspore.nn.Dense的 `weight_init` 是 ``None`` 时，权重使用HeUniform初始化。此时和PyTorch权重初始化方式一致。

mindspore.nn.Dense的 `bias_init` 是 ``None`` 时，偏差使用Uniform初始化。此时和PyTorch偏差初始化方式一致。

| 分类 | 子类  | PyTorch      | MindSpore    | 差异                         |
| ---- | ----- | ------------ | ------------ | ---------------------------- |
| 参数 | 参数1 | in_features  | in_channels  | 功能一致，参数名不同                          |
|      | 参数2 | out_features | out_channels | 功能一致，参数名不同                        |
|      | 参数3 | bias         | has_bias     | 功能一致，参数名不同        |
|      | 参数4 | -             | weight_init  | 权重参数的初始化方法，PyTorch无此参数         |
|      | 参数5 | -             | bias_init    | 偏置参数的初始化方法，PyTorch无此参数           |
|      | 参数6 | -             | activation   | 应用于全连接层输出的激活函数，PyTorch无此参数   |
|  输入   | 单输入 | input | x | 功能一致，参数名不同|

### 代码示例

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
from torch import nn
import numpy as np

net = nn.Linear(3, 4)
x = torch.tensor(np.array([[180, 234, 154], [244, 48, 247]]), dtype=torch.float)
output = net(x)
print(output.detach().numpy().shape)
# (2, 4)

# MindSpore
import mindspore
from mindspore import Tensor, nn
import numpy as np

x = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), mindspore.float32)
net = nn.Dense(3, 4)
output = net(x)
print(output.shape)
# (2, 4)
```