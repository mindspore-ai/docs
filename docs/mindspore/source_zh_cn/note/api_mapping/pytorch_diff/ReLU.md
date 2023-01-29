# 比较与torch.nn.ReLU的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/ReLU.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.nn.ReLU

```text
class torch.nn.ReLU(inplace=False)(input) -> Tensor
```

更多内容详见[torch.nn.ReLU](https://pytorch.org/docs/1.8.1/generated/torch.nn.ReLU.html)。

## mindspore.nn.ReLU

```text
class mindspore.nn.ReLU()(x) -> Tensor
```

更多内容详见[mindspore.nn.ReLU](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.ReLU.html)。

## 差异对比

PyTorch：ReLU激活函数。

MindSpore：MindSpore此算子实现功能与PyTorch一致，仅参数设置不同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | inplace | - | 是否就地执行，默认值：False。MindSpore无此参数 |
| | 参数2 | input | x | 功能一致，参数名不同 |

### 代码示例1

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
from torch import tensor
from torch import nn
import numpy as np

x = tensor(np.array([[0.1, -0.6], [-0.9, 0.8]]), dtype=torch.float32)
m = nn.ReLU()
out = m(x)
output = out.detach().numpy()
print(output)
# [[0.1 0. ]
#  [0.  0.8]]

# MindSpore
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([[0.1, -0.6], [-0.9, 0.8]]), dtype=mindspore.float32)
relu = nn.ReLU()
output = relu(x)
print(output)
# [[0.1 0. ]
#  [0.  0.8]]
```