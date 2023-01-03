# 比较与torch.broadcast_to的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/broadcast_to.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.broadcast_to

```text
torch.broadcast_to(input, shape) -> Tensor
```

更多内容详见[torch.broadcast_to](https://pytorch.org/docs/1.8.1/generated/torch.broadcast_to.html)。

## mindspore.ops.broadcast_to

```text
mindspore.ops.broadcast_to(x, shape) -> Tensor
```

更多内容详见[mindspore.ops.broadcast_to](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.broadcast_to.html)。

## 差异对比

PyTorch：将输入shape广播到目标shape。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，额外支持shape中存在-1维度的情况。如果目标shape中有-1维度，它被该维度中的输入shape的值替换。如果目标shape中有-1维度，则-1维度不能位于一个不存在的维度中。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 输入 | 单输入 | input | x | 功能一致，参数名不同 |
|参数 | 参数1 | shape | shape |功能一致 |

### 代码示例1

```python
# PyTorch
import torch

shape = (2, 3)
x = torch.tensor([[1], [2]]).float()
torch_output = torch.broadcast_to(x, shape)
print(torch_output.numpy())
# [[1. 1. 1.]
#  [2. 2. 2.]]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

shape = (2, 3)
x = Tensor(np.array([[1], [2]]).astype(np.float32))
output = mindspore.ops.function.broadcast_to(x, shape)
print(output)
# [[1. 1. 1.]
#  [2. 2. 2.]]
```
