# 比较与torch.nn.functional.elu的功能差异

## torch.nn.functional.elu

```text
torch.nn.functional.elu(input, alpha=1.0, inplace=False)
```

更多内容详见[torch.nn.functional.elu](https://pytorch.org/docs/1.8.1/nn.functional.html#elu)。

## mindspore.ops.Elu

``` text
class mindspore.ops.Elu(x, alpha=1.0)(x)
```

更多内容详见 [mindspore.ops.Elu](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Elu.html)。

## 差异对比

PyTorch：计算输入x的指数线性值，返回结果为 $$ \text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1)) $$，inplace参数可选择是否就地操作，默认为False。

MindSpore：MindSpore此API实现功能与PyTorch基本一致， 不过α目前只支持1.0。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input | x |功能一致， 参数名不同 |
| | 参数2 | alpha | alpha | α系数，MindSpore目前只支持alpha等于1.0 |
| | 参数3 | inplace | - | 功能一致，MindsSpore无此参数 |

### 代码示例1

> 两API都是实现指数线性单元功能，但PyTorch可以自定义α系数，MindSpore只支持系数为1.0，使用时需先进行实例化。

```python
# PyTorch
import torch
from torch import tensor
from torch.nn.functional import elu
import numpy as np

x_ = np.array([[np.arange(-6,0).reshape(2, 3),np.arange(0,6).reshape(2, 3)]])
x = tensor(x_, dtype=torch.float32)
output = elu(x, alpha = 1).detach().numpy()
print(output)
##[[[[-0.9975212  -0.99326205 -0.9816844 ]
#   [-0.95021296 -0.86466473 -0.63212055]]
#
#  [[ 0.          1.          2.        ]
#   [ 3.          4.          5.        ]]]]

# MindSpore
import mindspore as ms
from mindspore import ops
import numpy as np

x_ = np.array([[np.arange(-6,0).reshape(2, 3),np.arange(0,6).reshape(2, 3)]])
x = ms.Tensor(x_, ms.float32)
elu = ops.Elu()
output = elu(x)
print(output)
##[[[[-0.9975212  -0.99326205 -0.9816844 ]
# [-0.95021296 -0.86466473 -0.6321205 ]]
#
# [[ 0.          1.          2.        ]
#  [ 3.          4.          5.        ]]]]

```