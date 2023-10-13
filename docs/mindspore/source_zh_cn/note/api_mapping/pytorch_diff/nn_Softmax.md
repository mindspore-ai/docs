# 比较与torch.nn.Softmax的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/nn_Softmax.md)

## torch.nn.Softmax

```text
class torch.nn.Softmax(dim=None)(input) -> Tensor
```

更多内容详见[torch.nn.Softmax](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softmax.html)。

## mindspore.nn.Softmax

```text
class mindspore.nn.Softmax(axis=-1)(x) -> Tensor
```

更多内容详见[mindspore.nn.Softmax](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/nn/mindspore.nn.Softmax.html)。

## 差异对比

PyTorch：它是二分类函数在多分类上的推广，目的是将多分类的结果以概率的形式展现出来。

MindSpore：MindSpore此API实现功能与PyTorch一致，仅参数名不同。

| 分类 | 子类  | PyTorch | MindSpore | 差异                  |
| ---- | ----- | ------- | --------- | --------------------- |
| 参数 | 参数1 | dim     | axis      | 功能一致，参数名不同，默认值不同 |
| 输入  | 单输入 | input  | x   | 功能一致，参数名不同 |

### 代码示例

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
import numpy
from torch import tensor
import torch.nn as nn

x = torch.FloatTensor([1, 1])
softmax = nn.Softmax(dim=-1)(x)
print(softmax.numpy())
# [0.5 0.5]

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

x = Tensor(np.array([1, 1]), mindspore.float16)
softmax = mindspore.nn.Softmax()
output = softmax(x)
print(output)
# [0.5 0.5]
```
