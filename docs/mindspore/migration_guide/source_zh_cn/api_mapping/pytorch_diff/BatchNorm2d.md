# 比较与torch.nn.BatchNorm2d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/BatchNorm2d.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.nn.BatchNorm2d

```python
class torch.nn.BatchNorm2d(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=True,
    track_running_stats=True
)
```

更多内容详见[torch.nn.BatchNorm2d](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.BatchNorm2d)。

## mindspore.nn.BatchNorm2d

```python
class mindspore.nn.BatchNorm2d(
    num_features,
    eps=1e-05,
    momentum=0.9,
    affine=True,
    gamma_init="ones",
    beta_init="zeros",
    moving_mean_init="zeros",
    moving_var_init="ones",
    use_batch_statistics=None,
    data_format="NCHW")
)
```

更多内容详见[mindspore.nn.BatchNorm2d](https://mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.BatchNorm2d.html#mindspore.nn.BatchNorm2d)。

## 使用方式

PyTorch：用于running_mean和running_var计算的momentum参数的默认值为0.1。

MindSpore：momentum参数的默认值为0.9，与Pytorch的momentum关系为1-momentum，即当Pytorch的momentum值为0.2时，MindSpore的momemtum应为0.8。

## 代码示例

```python
# The following implements BatchNorm2d with MindSpore.
import numpy as np
import torch
import mindspore.nn as nn
from mindspore import Tensor

net = nn.BatchNorm2d(num_features=3, momentum=0.8)
x = Tensor(np.ones([1, 3, 2, 2]).astype(np.float32))
output = net(x)
print(output)
# Out:
# [[[[0.999995   0.999995]
#    [0.999995   0.999995]]
#
#   [[0.999995   0.999995]
#    [0.999995   0.999995]]
#
#   [[0.999995   0.999995]
#    [0.999995   0.999995]]]]


# The following implements BatchNorm2d with torch.
input_x = torch.randn(1, 3, 2, 2)
m = torch.nn.BatchNorm2d(3, momentum=0.2)
output = m(input_x)
print(output)
# Out:
# tensor([[[[ 0.0054,  1.6285],
#           [-0.8927, -0.7412]],
#
#          [[-0.2833, -0.1956],
#           [ 1.6118, -1.1329]],
#
#          [[-1.3467,  1.4556],
#           [-0.2303,  0.1214]]]], grad_fn=<NativeBatchNormBackward>)
```
