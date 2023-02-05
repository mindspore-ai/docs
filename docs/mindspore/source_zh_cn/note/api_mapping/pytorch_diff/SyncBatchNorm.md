# 比较与torch.nn.SyncBatchNorm的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SyncBatchNorm.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.SyncBatchNorm

```text
class torch.nn.SyncBatchNorm(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=True,
    track_running_stats=True,
    process_group=None
)(input) -> Tensor
```

更多内容详见[torch.nn.SyncBatchNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.SyncBatchNorm.html)。

## mindspore.nn.SyncBatchNorm

```text
class mindspore.nn.SyncBatchNorm(
    num_groups,
    eps=1e-05,
    momentum=0.9,
    affine=True,
    gamma_init='ones',
    beta_init='zeros'
    moving_mean_init='zeros',
    moving_var_init='ones',
    use_batch_statistics=None,
    process_groups=None
)(x) -> Tensor
```

更多内容详见[mindspore.nn.SyncBatchNorm](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.SyncBatchNorm.html)。

## 差异对比

PyTorch：。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，MindSpore还可以对需要学习的放射参数进行额外的初始化。

| 分类 | 子类  | PyTorch      | MindSpore    | 差异                                                         |
| ---- | ----- | ------------ | ------------ | ------------------------------------------------------------ |
| 输入 | 单输入 | input        | x            | 接口输入，功能一致，参数名不同                                         |
| 参数 | 参数1 | num_groups   | num_groups   | -                                                            |
|      | 参数2 | num_channels | num_channels | -                                                            |
|      | 参数3 | eps          | eps          | -                                                            |
|      | 参数4 | affine       | affine       | -                                                            |
|      | 参数5 | -            | gamma_init   | 给公式中用于学习的放射变换参数gamma初始化，默认是'ones'，而PyTorch不能额外设置，只能是'ones' |
|      | 参数6 | -           | beta_init    | 给公式中用于学习的放射变换参数beta初始化，默认是'zeros'，而PyTorch不能额外设置，只能是'zeros' |

## 代码示例1

> 两API功能基本一致，MindSpore还可以对两个学习的参数进行额外初始化。

```python
# PyTorch
import torch
import numpy as np
from torch import tensor, nn

x = tensor(np.ones([1, 2, 4, 4], np.float32))
net = nn.SyncBatchNorm(2, 2)
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
net = nn.SyncBatchNorm(2, 2)
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
