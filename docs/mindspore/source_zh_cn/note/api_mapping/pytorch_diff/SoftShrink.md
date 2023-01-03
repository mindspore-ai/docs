# 比较与torch.nn.Softshrink的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SoftShrink.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.Softshrink

```text
class torch.nn.Softshrink(lambd=0.5)(input) -> Tensor
```

更多内容详见[torch.nn.Softshrink](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softshrink.html)。

## mindspore.nn.SoftShrink

```text
class mindspore.nn.SoftShrink(lambd=0.5)(input_x) -> Tensor
```

更多内容详见[mindspore.nn.SoftShrink](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.SoftShrink.html)。

## 差异对比

PyTorch：用于计算Softshrink激活函数。

MindSpore：接口名称与PyTorch有差异，MindSpore为SoftShrink，PyTorch为Softshrink，功能一致。

| 分类 | 子类  | PyTorch | MindSpore | 差异                    |
| ---- | ----- | ------ | --------- | ----------------------- |
| 参数 | 参数1 | lambd  | lambd     | - |
|      | 参数2 | input  | input_x   | 功能一致，参数名不同 |

### 代码示例1

> 计算lambd=0.3的SoftShrink激活函数。

```python
# PyTorch
import numpy as np
import torch
from torch import tensor, nn

m = nn.Softshrink(lambd=0.3)
input = np.array([[0.5297, 0.7871, 1.1754], [0.7836, 0.6218, -1.1542]])
input_t = tensor(input)
output = m(input_t)
print(output.numpy())
# [[ 0.2297  0.4871  0.8754]
#  [ 0.4836  0.3218 -0.8542]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor, nn

m = nn.SoftShrink(lambd=0.3)
input = np.array([[0.5297, 0.7871, 1.1754], [0.7836, 0.6218, -1.1542]])
input_t = Tensor(input, mindspore.float32)
output = m(input_t)
print(output)
# [[ 0.22969997  0.4871      0.8754    ]
#  [ 0.48359996  0.3218     -0.85419995]]
```

### 代码示例2

> SoftShrink默认`lambd=0.5`。

```python
# PyTorch
import numpy as np
import torch
from torch import tensor, nn

m = nn.Softshrink()
input = np.array([[0.5297, 0.7871, 1.1754], [0.7836, 0.6218, -1.1542]])
input_t = tensor(input)
output = m(input_t)
print(output.numpy())
# [[ 0.0297  0.2871  0.6754]
#  [ 0.2836  0.1218 -0.6542]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor, nn

m = nn.SoftShrink()
input = np.array([[0.5297, 0.7871, 1.1754], [0.7836, 0.6218, -1.1542]])
input_t = Tensor(input, mindspore.float32)
output = m(input_t)
print(output)
# [[ 0.02969998  0.28710002  0.6754    ]
#  [ 0.28359997  0.12180001 -0.65419996]]
```
