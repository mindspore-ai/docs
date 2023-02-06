# 比较与torch.nn.Hardshrink的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/HShrink.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.nn.Hardshrink

```text
torch.nn.Hardshrink(lambd=0.5)(input) -> Tensor
```

更多内容详见[torch.nn.Hardshrink](https://pytorch.org/docs/1.8.1/generated/torch.nn.Hardshrink.html)。

## mindspore.nn.HShrink

```text
mindspore.nn.HShrink(lambd=0.5)(input_x) -> Tensor
```

更多内容详见[mindspore.nn.HShrink](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.HShrink.html)。

## 差异对比

PyTorch：激活函数，按输入元素计算输出。

MindSpore：MindSpore此API实现功能与PyTorch一致，仅参数名不同。

| 分类 | 子类  | PyTorch | MindSpore | 差异 |
| ---- | ----- | ------- | --------- | ---- |
| 参数 | 参数1 | lambd   | lambd     | -    |
| 输入 | 单输入 | input   | input_x     | 功能一致，参数名不同 |

### 代码示例

> 两API功能一致，用法相同。

```python
# PyTorch
import torch
import torch.nn as nn

m = nn.Hardshrink()
input = torch.tensor([[0.5, 1, 2.0], [0.0533, 0.0776, -2.1233]], dtype=torch.float32)
output = m(input)
output = output.detach().numpy()
print(output)
# [[ 0.      1.      2.    ]
#  [ 0.      0.     -2.1233]]

# MindSpore
import mindspore
from mindspore import Tensor, nn
import numpy as np

input_x = Tensor(np.array([[0.5, 1, 2.0], [0.0533, 0.0776, -2.1233]]), mindspore.float32)
hshrink = nn.HShrink()
output = hshrink(input_x)
print(output)
# [[ 0.      1.      2.    ]
#  [ 0.      0.     -2.1233]]
```

