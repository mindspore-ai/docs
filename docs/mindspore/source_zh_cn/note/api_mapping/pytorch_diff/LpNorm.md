# 比较与torch.norm/Tensor.norm的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/LpNorm.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.norm

```python
torch.norm(
    input,
    p='fro',
    dim=None,
    keepdim=False,
    out=None,
    dtype=None
)
```

```python
torch.Tensor.norm(
    p='fro',
    dim=None,
    keepdim=False,
    dtype=None
)
```

更多内容详见[torch.norm](https://pytorch.org/docs/1.5.0/torch.html#torch.norm)。

## mindspore.ops.LpNorm

```python
class mindspore.ops.LpNorm(
    axis,
    p=2,
    keep_dims=False,
    epsilon=1e-12
)(input)
```

更多内容详见[mindspore.nn.LpNorm](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.LpNorm.html#mindspore.ops.LpNorm)。

## 使用方式

PyTorch：p参数可以支持int，float，inf，-inf，'fro'，'nuc'等类型或值，以实现不同类型的归一化。

MindSpore：目前仅支持整数p范式的归一化。

## 代码示例

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, only Lp norm is supported.
net = ops.LpNorm(axis=0, p=2)
input_x = Tensor(np.array([[4, 4, 9, 1], [2, 1, 3, 6]]), mindspore.float32)
output = net(input_x)
print(output)
# Out：
# [4.472136 4.1231055 9.486833 6.0827627]

# In torch, p=2
input_x = torch.tensor(np.array([[4, 4, 9, 1], [2, 1, 3, 6]]), dtype=torch.float)
output1 = torch.norm(input_x, dim=0, p=2)
print(output1)
# Out：
# tensor([4.4721, 4.1231, 9.4868, 6.0828])

# In torch, p='nuc'
input_x = torch.tensor(np.array([[4, 4, 9, 1], [2, 1, 3, 6]]), dtype=torch.float)
output2 = torch.norm(input_x, dim=(0, 1), p='nuc')
print(output2)
# Out：
# tensor([16.8892])
```
