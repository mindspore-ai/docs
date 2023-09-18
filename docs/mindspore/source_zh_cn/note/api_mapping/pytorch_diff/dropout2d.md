# 比较与torch.nn.functional.dropout2d的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/dropout2d.md)

## torch.nn.functional.dropout2d

```python
torch.nn.functional.dropout2d(input, p=0.5, training=True, inplace=False) -> Tensor
```

更多内容详见[torch.nn.functional.dropout2d](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.dropout2d)。

## mindspore.ops.dropout2d

```python
mindspore.ops.dropout2d(input, p=0.5, training=True) -> Tensor
```

更多内容详见[mindspore.ops.dropout2d](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.dropout2d.html)。

## 差异对比

MindSpore此API功能与PyTorchy一致，参数支持的数据类型有差异。

PyTorch：在训练期间，dropout2d以服从伯努利分布的概率p随机将输入Tensor的某些通道归零，每个通道将会独立依据伯努利分布概率p来确定是否被清零。对输入Tensor的某些通道清零，已被证明能有效地减少过度拟合，防止神经元共适应。

MindSpore：MindSpore只支持秩为4的Tensor作为输入。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                                         |
| ---- | ----- | ------- | --------- | -----------------------------------------------------------|
| 参数 | 参数1 | input | input | MindSpore只支持秩为4的Tensor作为输入 |
|      | 参数2 | p | p | - |
|      | 参数3 | training  | training | - |
|      | 参数4 | inplace   | - | - |

### 代码示例1

```python
# PyTorch
import torch

input = torch.ones(3, 2, 4)
output = torch.nn.functional.dropout2d(input)
print(output.shape)
# torch.Size([3, 2, 4])

# MindSpore
import mindspore as ms
from mindspore import ops
from mindspore import Tensor
import numpy as np

input = Tensor(np.ones([3, 2, 4]), ms.float32)
input = input.expand_dims(0)
output = ops.dropout2d(input)
output = output.squeeze(0)
print(output.shape)
# (3, 2, 4)
```

### 代码示例2

```python
# PyTorch
import torch

input = torch.ones(1, 2, 3, 2, 4)
output = torch.nn.functional.dropout2d(input)
print(output.shape)
# torch.Size([1, 2, 3, 2, 4])

# MindSpore
import mindspore as ms
from mindspore import ops
from mindspore import Tensor
import numpy as np

input = Tensor(np.ones([1, 2, 3, 2, 4]), ms.float32)
input = input.squeeze(0)
output = ops.dropout2d(input)
output = output.expand_dims(0)
print(output.shape)
# (1, 2, 3, 2, 4)
```
