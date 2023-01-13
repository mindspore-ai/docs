# 比较与torch.nn.PReLU的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/PReLU.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.PReLU

```text
class torch.nn.PReLU(num_parameters=1, init=0.25)(input) -> Tensor
```

更多内容详见[torch.nn.PReLU](https://pytorch.org/docs/1.8.1/generated/torch.nn.PReLU.html)。

## mindspore.nn.PReLU

```text
class mindspore.nn.PReLU(channel=1, w=0.25)(x) -> Tensor
```

更多内容详见[mindspore.nn.PReLU](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.PReLU.html)。

## 差异对比

PyTorch：PReLU激活函数。

MindSpore：MindSpore此算子功能与PyTorch一致，仅参数名不同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | num_parameters | channel | 功能一致，参数名不同 |
| | 参数2 | init | w | 功能一致，参数名不同 |
| 输入 | 单输入 | input | x | 功能一致，参数名不同 |

### 代码示例1

> 两API此功能一致，用法相同，默认值相同，仅参数名不同。

```python
# PyTorch
import torch
from torch import tensor
from torch import nn
import numpy as np

x = tensor(np.array([[0.1, -0.6], [-0.9, 0.9]]), dtype=torch.float32)
m = nn.PReLU()
out = m(x)
output = out.detach().numpy()
print(output)
# [[ 0.1   -0.15 ]
#  [-0.225  0.9  ]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x = Tensor(np.array([[0.1, -0.6], [-0.9, 0.9]]), mindspore.float32)
prelu = nn.PReLU()
output = prelu(x)
print(output)
# [[ 0.1   -0.15 ]
#  [-0.225  0.9  ]]
```

### 代码示例2

> 若不使用默认值，使用MindSpore只需将对应参数设置为相等的数即可实现相同功能。

```python
# PyTorch
import torch
from torch import tensor
from torch import nn
import numpy as np

x = tensor(np.array([[0.1, -0.6], [-0.5, 0.9]]), dtype=torch.float32)
m = nn.PReLU(num_parameters=1, init=0.5)
out = m(x)
output = out.detach().numpy()
print(output)
# [[ 0.1  -0.3 ]
#  [-0.25  0.9 ]]

# MindSpore
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([[0.1, -0.6], [-0.5, 0.9]]), mindspore.float32)
prelu = nn.PReLU(channel=1, w=0.5)
output = prelu(x)
print(output)
# [[ 0.1  -0.3 ]
#  [-0.25  0.9 ]]
```
