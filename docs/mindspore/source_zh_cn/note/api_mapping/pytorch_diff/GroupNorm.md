# 比较与torch.nn.GroupNorm的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3.q1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3.q1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/GroupNorm.md)

## torch.nn.GroupNorm

```text
class torch.nn.GroupNorm(
    num_groups,
    num_channels,
    eps=1e-05,
    affine=True
)(input) -> Tensor
```

更多内容详见[torch.nn.GroupNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.GroupNorm.html)。

## mindspore.nn.GroupNorm

```text
class mindspore.nn.GroupNorm(
    num_groups,
    num_channels,
    eps=1e-05,
    affine=True,
    gamma_init='ones',
    beta_init='zeros'
)(x) -> Tensor
```

更多内容详见[mindspore.nn.GroupNorm](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/nn/mindspore.nn.GroupNorm.html)。

## 差异对比

PyTorch：在mini-batch输入上进行组归一化，把通道划分为组，然后计算每一组之内的均值和方差，以进行归一化。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，MindSpore还可以对需要学习的放射参数进行额外的初始化。

| 分类 | 子类  | PyTorch      | MindSpore    | 差异                                                         |
| ---- | ----- | ------------ | ------------ | ------------------------------------------------------------ |
| 参数 | 参数1 | num_groups   | num_groups   | -                                                            |
|      | 参数2 | num_channels | num_channels | -                                                            |
|      | 参数3 | eps          | eps          | -                                                            |
|      | 参数4 | affine       | affine       | -                                                            |
|      | 参数5 | -            | gamma_init   | 给公式中用于学习的放射变换参数gamma初始化，默认是'ones'，而PyTorch不能额外设置，只能是'ones' |
|      | 参数6 | -           | beta_init    | 给公式中用于学习的放射变换参数beta初始化，默认是'zeros'，而PyTorch不能额外设置，只能是'zeros' |
| 输入 | 单输入 | input        | x            | 接口输入，功能一致，参数名不同                                         |

## 代码示例1

> 两API功能基本一致，MindSpore还可以对两个学习的参数进行额外初始化。

```python
# PyTorch
import torch
import numpy as np
from torch import tensor, nn

x = tensor(np.ones([1, 2, 4, 4], np.float32))
net = nn.GroupNorm(2, 2)
output = net(x).detach().numpy()
print(output)
# [[[[0. 0. 0. 0.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]]
#
#   [[0. 0. 0. 0.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]]]]

# MindSpore
import mindspore as ms
import numpy as np
from mindspore import Tensor, nn

x = Tensor(np.ones([1, 2, 4, 4], np.float32))
net = nn.GroupNorm(2, 2)
output = net(x)
print(output)
# [[[[0. 0. 0. 0.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]]
#
#   [[0. 0. 0. 0.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]]]]
```
