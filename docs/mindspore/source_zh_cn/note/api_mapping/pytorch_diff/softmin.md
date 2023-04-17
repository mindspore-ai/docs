# 比较与torch.nn.Softmin的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/softmin.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

## torch.nn.Softmin

```python
torch.nn.Softmin(
    dim=None
)
```

更多内容详见[torch.nn.Softmin](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softmin.html)。

## mindspore.nn.Softmin

```python
class mindspore.nn.Softmin(
    axis=-1
)
```

更多内容详见[mindspore.nn.Softmin](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.Softmin.html)。

## 差异对比

PyTorch：支持使用`dim`参数实例化，将指定维度元素缩放到[0, 1]之间并且总和为1，默认值：None。

MindSpore：支持使用 `axis`参数实例化，将指定维度元素缩放到[0, 1]之间并且总和为1，默认值：-1。

| 分类 | 子类  | PyTorch | MindSpore | 差异                    |
| ---- | ----- |---------|-----------| ----------------------- |
| 参数 | 参数1 | dim     | axis      | 功能一致，参数名不同 |

## 代码示例

```python
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

# MindSpore
x = ms.Tensor(np.array([1, 2, 3, 4, 5]), ms.float32)
softmin = nn.Softmin()
output1 = softmin(x)
print(output1)
# Out:
# [0.6364086 0.23412167 0.08612854 0.03168492 0.01165623]
x = ms.Tensor(np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]), ms.float32)
output2 = softmin(x, axis=0)
print(output2)
# out:
# [[0.98201376 0.880797   0.5        0.11920292 0.01798621]
#  [0.01798621 0.11920292 0.5        0.880797   0.98201376]]

# PyTorch
input = torch.tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
output3 = F.softmin(input, dim=0)
print(output3)
# Out:
# tensor([0.6364, 0.2341, 0.0861, 0.0317, 0.0117], dtype=torch.float64)
```
