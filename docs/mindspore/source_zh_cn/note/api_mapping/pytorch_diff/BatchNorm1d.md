# 比较与torch.nn.BatchNorm1d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/BatchNorm1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

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

更多内容详见[mindspore.nn.BatchNorm1d](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.BatchNorm1d.html)。

## 差异对比

PyTorch：对输入的二维或三维数据进行批归一化。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，momentum参数默认值为0.9，与PyTorch的momentum转换关系为1-momentum。

| 分类 | 子类   | PyTorch             | MindSpore            | 差异                                                         |
| ---- | ------ | ------------------- | -------------------- | ------------------------------------------------------------ |
| 参数 | 参数1  | num_features        | num_features         | -                                                            |
|      | 参数2  | eps                 | eps                  | -                                                            |
|      | 参数3  | momentum            | momentum             | 功能一致，但PyTorch里的默认值是0.1，MindSpore里是0.9         |
|      | 参数4  | affine              | affine               | -                                                            |
|      | 参数5  | track_running_stats              | -               | 不涉及                                |
|      | 参数6  | input               | x                    | 功能基本一致，但PyTorch里允许输入是二维或三维的，而MindSpore里的输入只能是二维的 |
|      | 参数7  | -                   | gamma_init           |    PyTorch无此参数，MindSpore可以初始化参数gamma的值    |
|      | 参数8  | -                   | beta_init            |    PyTorch无此参数，MindSpore可以初始化参数beta的值     |
|      | 参数9  | -                   | moving_mean_init     |    PyTorch无此参数，MindSpore可以初始化参数moving_mean的值    |
|      | 参数10  | -                   | moving_var_init      |    PyTorch无此参数，MindSpore可以初始化参数moving_var的值     |
|      | 参数11 | -                   | use_batch_statistics |    PyTorch无此参数，MindSpore里如果为True，则使用当前批次数据的平均值和方差值      |
|      | 参数12  | -                   | data_format      |    PyTorch无此参数    |

### 代码示例

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
import numpy as np
from torch import nn, tensor

m = nn.BatchNorm1d(4, affine=False, momentum=0.1)
input = tensor(np.array([[0.7, 0.5, 0.5, 0.6], [0.5, 0.4, 0.6, 0.9]]).astype(np.float32))
output = m(input)
print(output.detach().numpy())
# [[ 0.9995001   0.9980063  -0.998006   -0.99977785]
#  [-0.9995007  -0.9980057   0.998006    0.99977785]]

# MindSpore
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor

net = nn.BatchNorm1d(num_features=4, affine=False, momentum=0.9)
net.set_train()
input = Tensor(np.array([[0.7, 0.5, 0.5, 0.6], [0.5, 0.4, 0.6, 0.9]]).astype(np.float32))
output = net(input)
print(output.asnumpy())
# [[ 0.9995001  0.9980063 -0.998006  -0.9997778]
#  [-0.9995007 -0.9980057  0.998006   0.9997778]]
```
