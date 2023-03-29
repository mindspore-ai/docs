# 比较与torch.nn.functional.dropout3d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/dropout3d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

## torch.nn.functional.dropout3d

```python
torch.nn.functional.dropout3d(input, p=0.5, training=True, inplace=False) -> Tensor
```

更多内容详见[torch.nn.functional.dropout3d](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.dropout3d)。

## mindspore.ops.dropout3d

```python
mindspore.ops.dropout3d(input, p=0.5, training=True) -> Tensor
```

更多内容详见[mindspore.ops.dropout3d](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.dropout3d.html)。

## 差异对比

PyTorch：在训练期间，dropout3d以服从伯努利分布的概率p随机将输入Tensor的某些通道归零，每个通道将会独立依据伯努利分布概率p来确定是否被清零。对输入Tensor的某些通道清零，已被证明能有效地减少过度拟合，防止神经元共适应。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                                         |
| ---- | ----- | ------- | --------- | -----------------------------------------------------------|
| 参数 | 参数1 | input | input | MindSpore只支持秩为5的Tensor作为输入 |
|      | 参数2 | p | p | - |
|      | 参数3 | training  | training | - |
|      | 参数4 | inplace   | - | - |

### 代码示例1

```python
# PyTorch
import torch

input = torch.ones(2, 3, 2, 4)
output = torch.nn.functional.dropout3d(input)
print(output.shape)
# torch.Size([2, 3, 2, 4])

# MindSpore
import mindspore as ms
from mindspore import ops
from mindspore import Tensor
import numpy as np

input = Tensor(np.ones([2, 3, 2, 4]), ms.float32)
input = input.expand_dims(0)
output = ops.dropout3d(input)
output = output.squeeze(0)
print(output.shape)
# (2, 3, 2, 4)
```

### 代码示例2

```python
# PyTorch
import torch

input = torch.ones(1, 1, 2, 3, 2, 4)
output = torch.nn.functional.dropout3d(input)
print(output.shape)
# torch.Size([1, 1, 2, 3, 2, 4])

# MindSpore
import mindspore as ms
from mindspore import ops
from mindspore import Tensor
import numpy as np

input = Tensor(np.ones([1, 1, 2, 3, 2, 4]), ms.float32)
input = input.squeeze(0)
output = ops.dropout3d(input)
output = output.expand_dims(0)
print(output.shape)
# (1, 1, 2, 3, 2, 4)
```
