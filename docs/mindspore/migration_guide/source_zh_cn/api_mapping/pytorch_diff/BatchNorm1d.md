# 比较与torch.nn.BatchNorm1d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/BatchNorm1d.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.nn.BatchNorm1d

```python
class torch.nn.BatchNorm1d(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=True,
    track_running_stats=True
)
```

更多内容详见[torch.nn.BatchNorm1d](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.BatchNorm1d)。

## mindspore.nn.BatchNorm1d

```python
class mindspore.nn.BatchNorm1d(
    num_features,
    eps=1e-05,
    momentum=0.9,
    affine=True,
    gamma_init="ones",
    beta_init="zeros",
    moving_mean_init="zeros",
    moving_var_init="ones",
    use_batch_statistics=None)
)
```

更多内容详见[mindspore.nn.BatchNorm1d](https://mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.BatchNorm1d.html#mindspore.nn.BatchNorm1d)。

## 使用方式

PyTorch：用于running_mean和running_var计算的momentum参数的默认值为0.1。

MindSpore：momentum参数的默认值为0.9，与Pytorch的momentum关系为1-momentum，即当Pytorch的momentum值为0.2时，MindSpore的momemtum应为0.8。

## 代码示例

```python
# The following implements BatchNorm1d with MindSpore.
import numpy as np
import torch
import mindspore.nn as nn
from mindspore import Tensor

net = nn.BatchNorm1d(num_features=4, momentum=0.8)
x = Tensor(np.array([[0.7, 0.5, 0.5, 0.6],
                     [0.5, 0.4, 0.6, 0.9]]).astype(np.float32))
output = net(x)
print(output)
# Out:
# [[ 0.6999965   0.4999975  0.4999975  0.59999704 ]
#  [ 0.4999975   0.399998   0.59999704 0.89999545 ]]


# The following implements BatchNorm1d with torch.
input_x = torch.randn(2, 4)
m = torch.nn.BatchNorm1d(4, momentum=0.2)
output = m(input_x)
print(output)
# Out:
# tensor([[-0.9991, -1.0000, -1.0000,  1.0000],
#         [ 0.9991,  1.0000,  1.0000, -1.0000]],
#        grad_fn=<NativeBatchNormBackward>)
```
