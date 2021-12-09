# 比较与torch.nn.Linear的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/Dense.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.nn.Linear

```python
torch.nn.Linear(
    in_features,
    out_features,
    bias=True
)
```

更多内容详见[torch.nn.Linear](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Linear)。

## mindspore.nn.Dense

```python
class mindspore.nn.Dense(
    in_channels,
    out_channels,
    weight_init='normal',
    bias_init='zeros',
    has_bias=True,
    activation=None
)(input)
```

更多内容详见[mindspore.nn.Dense](https://mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.Dense.html#mindspore.nn.Dense)。

## 使用方式

Pytorch：对传入数据应用线性变换，默认权重矩阵和偏移矩阵都由均匀分布初始化。

MindSpore：对传入数据应用线性变换，在输出数据之前可以选择应用激活函数`activation`，默认权重矩阵由标准正态分布初始化，偏移矩阵初始化为0。

## 代码示例

```python
import mindspore
from mindspore import Tensor, nn
import torch
import numpy as np

# In MindSpore, default weight will be initialized through standard normal distribution.
# Default bias will be initialized by zero.
# Default none activation used.
input_net = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), mindspore.float32)
net = nn.Dense(3, 4)
output = net(input_net)
print(output.shape)
# Out：
# (2, 4)

# In torch, default weight and bias will be initialized through uniform distribution.
# No parameter to set the activation.
input_net = torch.Tensor(np.array([[180, 234, 154], [244, 48, 247]]))
net = torch.nn.Linear(3, 4)
output = net(input_net)
print(output.shape)
# Out：
# torch.Size([2, 4])
```