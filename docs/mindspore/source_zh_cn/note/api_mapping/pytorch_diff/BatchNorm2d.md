# 比较与torch.nn.BatchNorm2d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/BatchNorm2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

MindSpore：momentum参数的默认值为0.9，与Pytorch的momentum关系为1-momentum，即当Pytorch的momentum值为0.2时，MindSpore的momemtum应为0.8。其中，beta、gamma、moving_mean和moving_variance参数分别对应Pytorch的bias、weight、running_mean和running_var参数。

## 代码示例

```python
# The following implements BatchNorm2d with MindSpore.
import numpy as np
import torch
import mindspore.nn as nn
from mindspore import Tensor

net = nn.BatchNorm2d(num_features=2, momentum=0.8)
x = Tensor(np.array([[[[1, 2], [1, 2]], [[3, 4], [3, 4]]]]).astype(np.float32))
output = net(x)
print(output)
# Out:
# [[[[0.999995   1.99999]
#    [0.999995   1.99999]]
#
#   [[2.999985   3.99998]
#    [2.999985   3.99998]]]]


# The following implements BatchNorm2d with torch.
input_x = torch.tensor(np.array([[[[1, 2], [1, 2]], [[3, 4], [3, 4]]]]).astype(np.float32))
m = torch.nn.BatchNorm2d(2, momentum=0.2)
output = m(input_x)
print(output)
# Out:
# tensor([[[[-1.0000,  1.0000],
#           [-1.0000,  1.0000]],
#
#          [[-1.0000,  1.0000],
#           [-1.0000,  1.0000]]]], grad_fn=<NativeBatchNormBackward>)
```
