# 比较与torch.nn.MaxPool3d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/MaxPool3d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

## torch.nn.MaxPool3d

```text
torch.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)(input) -> Tensor
```

更多内容详见[torch.nn.MaxPool3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxPool3d.html)。

## mindspore.nn.MaxPool3d

```text
mindspore.nn.MaxPool3d(kernel_size=1, stride=1, pad_mode="valid", padding=0, dilation=1, return_indices=False, ceil_mode=False)(x) -> Tensor
```

更多内容详见[mindspore.nn.MaxPool3d](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.MaxPool3d.html)。

## 使用方式

PyTorch：对输入的多维数据进行三维的最大池化运算。

MindSpore：MindSpore此API实现功能同时兼容TensorFlow和PyTorch，`pad_mode` 为 "valid" 或者 "same" 时，功能与TensorFlow一致，`pad_mode` 为 "pad" 时，功能与PyTorch一致，MindSpore相比PyTorch1.8.1额外支持了维度为2的输入，与PyTorch1.12一致。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | kernel_size | kernel_size |功能一致，PyTorch无默认值 |
| | 参数2 | stride | stride |功能一致，默认值不同 |
| | 参数3 | padding | padding | 功能一致 |
| | 参数4 | dilation | dilation | 功能一致 |
| | 参数5 | return_indices | return_indices | 功能一致|
| | 参数6 | ceil_mode | ceil_mode | 功能一致 |
| | 参数7 | input | x | 功能一致，参数名不同 |
| | 参数8 | - | pad_mode | 控制填充模式，PyTorch无此参数 |

## 代码示例

> mindspore为 `pad` 模式时，行为一致。

```python
import mindspore as ms
from mindspore import Tensor
import mindspore.nn as nn
import torch
import numpy as np

np_x = np.random.randint(0, 10, [1, 2, 4, 4, 5])

x = Tensor(np_x, ms.float32)
max_pool = nn.MaxPool3d(kernel_size=2, stride=1, pad_mode='pad', padding=1, dilation=1, return_indices=False)
output = max_pool(x)
result = output.shape
print(result)
# (1, 2, 5, 5, 6)
x = torch.tensor(np_x, dtype=torch.float32)
max_pool = torch.nn.MaxPool3d(kernel_size=2, stride=1, padding=1, dilation=1, return_indices=False)
output = max_pool(x)
result = output.shape
print(result)
# torch.Size([1, 2, 5, 5, 6])
```
