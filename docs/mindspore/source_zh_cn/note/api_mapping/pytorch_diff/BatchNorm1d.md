# 比较与torch.nn.BatchNorm1d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/BatchNorm1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.BatchNorm1d

```text
class torch.nn.BatchNorm1d(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=True,
    track_running_stats=True
)(input) -> Tensor
```

更多内容详见[torch.nn.BatchNorm1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.BatchNorm1d.html)。

## mindspore.nn.BatchNorm1d

```text
class mindspore.nn.BatchNorm1d(
    num_features,
    eps=1e-5,
    momentum=0.9,
    affine=True,
    gamma_init='ones',
    beta_init='zeros',
    moving_mean_init='zeros',
    moving_var_init='ones',
    use_batch_statistics=None,
    data_format='NCHW'
)(x) -> Tensor
```

更多内容详见[mindspore.nn.BatchNorm1d](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.BatchNorm1d.html)。

## 差异对比

PyTorch：对输入的二维或三维数据进行批归一化。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。MindSpore中momentum参数默认值为0.9，与PyTorch的momentum转换关系为1-momentum，默认值行为与PyTorch相同；训练以及推理时的参数更新策略和PyTorch有所不同，详细区别请参考[与PyTorch典型区别-BatchNorm](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/typical_api_comparision.html#nn.BatchNorm2d)。

| 分类 | 子类   | PyTorch             | MindSpore            | 差异                                                         |
| ---- | ------ | ------------------- | -------------------- | ------------------------------------------------------------ |
| 参数 | 参数1  | num_features        | num_features         | -                                                            |
|      | 参数2  | eps                 | eps                  | -                                                            |
|      | 参数3  | momentum            | momentum             | 功能一致，但PyTorch中的默认值是0.1，MindSpore中是0.9，与PyTorch的momentum转换关系为1-momentum，默认值行为与PyTorch相同        |
|      | 参数4  | affine              | affine               | -                                                            |
|      | 参数5  | track_running_stats              | use_batch_statistics               | 功能一致，不同值对应的默认方式不同，详细区别请参考[与PyTorch典型区别-nn.BatchNorm2d](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/typical_api_comparision.html#nn.BatchNorm2d)                                |
|      | 参数6  | -                   | gamma_init           |    PyTorch无此参数，MindSpore可以初始化参数gamma的值    |
|      | 参数7  | -                   | beta_init            |    PyTorch无此参数，MindSpore可以初始化参数beta的值     |
|      | 参数8  | -                   | moving_mean_init     |    PyTorch无此参数，MindSpore可以初始化参数moving_mean的值    |
|      | 参数9  | -                   | moving_var_init      |    PyTorch无此参数，MindSpore可以初始化参数moving_var的值     |
|      | 参数10  | -                   | data_format      |    PyTorch无此参数    |
| 输入 | 单输入 | input               | x                    | 接口输入，功能一致，仅参数名不同 |

### 代码示例

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
import numpy as np
from torch import nn, tensor

net = nn.BatchNorm1d(4, affine=False, momentum=0.1)
x = tensor(np.array([[0.7, 0.5, 0.5, 0.6], [0.5, 0.4, 0.6, 0.9]]).astype(np.float32))
output = net(x)
print(output.detach().numpy())
# [[ 0.9995001   0.9980063  -0.998006   -0.99977785]
#  [-0.9995007  -0.9980057   0.998006    0.99977785]]

# MindSpore
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor

net = nn.BatchNorm1d(num_features=4, affine=False, momentum=0.9)
net.set_train()
x = Tensor(np.array([[0.7, 0.5, 0.5, 0.6], [0.5, 0.4, 0.6, 0.9]]).astype(np.float32))
output = net(x)
print(output.asnumpy())
# [[ 0.9995001  0.9980063 -0.998006  -0.9997778]
#  [-0.9995007 -0.9980057  0.998006   0.9997778]]
```
