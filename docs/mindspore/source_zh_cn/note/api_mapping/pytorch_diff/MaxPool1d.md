# 比较与torch.nn.MaxPool1d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/MaxPool1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

## torch.nn.MaxPool1d

```text
torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)(input) -> Tensor
```

更多内容详见[torch.nn.MaxPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxPool1d.html)。

## mindspore.nn.MaxPool1d

```text
mindspore.nn.MaxPool1d(kernel_size=1, stride=1, pad_mode="valid", padding=0, dilation=1, return_indices=False, ceil_mode=False)(x) -> Tensor
```

更多内容详见[mindspore.nn.MaxPool1d](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.MaxPool1d.html)。

## 差异对比

PyTorch：对时间数据进行最大池化运算。

MindSpore：MindSpore此API实现功能同时兼容TensorFlow和PyTorch，`pad_mode` 为 "valid" 或者 "same" 时，功能与TensorFlow一致，`pad_mode` 为 "pad" 时，功能与PyTorch一致，MindSpore相比PyTorch1.8.1额外支持了维度为2的输入，与PyTorch1.12一致。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | kernel_size | kernel_size | 功能一致，PyTorch无默认值 |
| | 参数2 | stride | stride | 功能一致，默认值不同  |
| | 参数3 | padding |padding| 功能一致 |
| | 参数4 | dilation | dilation | 功能一致 |
| | 参数5 | return_indices | return_indices | 功能一致 |
| | 参数6 | ceil_mode | ceil_mode | 功能一致 |
| | 参数7 | input | x | 功能一致，参数名不同 |
| | 参数8 | - | pad_mode | 控制填充模式，PyTorch无此参数 |

### 代码示例1

> 构建一个卷积核大小为1x3，步长为1的池化层，padding默认为0，不进行元素填充。dilation的默认值为1，窗口中的元素是连续的。池化填充模式的默认值为valid，在不填充的前提下返回有效计算所得的输出，不满足计算的多余像素会被丢弃。在相同的参数设置下，两API实现相同的功能，对数据进行了最大池化运算。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

max_pool = torch.nn.MaxPool1d(kernel_size=3, stride=1)
x = tensor(np.random.randint(0, 10, [1, 2, 4]), dtype=torch.float32)
output = max_pool(x)
result = output.shape
print(tuple(result))
# (1, 2, 2)

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

max_pool = mindspore.nn.MaxPool1d(kernel_size=3, stride=1)
x = Tensor(np.random.randint(0, 10, [1, 2, 4]), mindspore.float32)
output = max_pool(x)
result = output.shape
print(result)
# (1, 2, 2)
```

### 代码示例2

> mindspore为 `pad` 模式时，行为一致。

```python
import mindspore as ms
from mindspore import Tensor
import mindspore.nn as nn
import torch
import numpy as np

np_x = np.random.randint(0, 10, [1, 2, 4])

x = Tensor(np_x, ms.float32)
max_pool = nn.MaxPool1d(kernel_size=2, stride=1, pad_mode='pad', padding=1, dilation=1, return_indices=False)
output = max_pool(x)
result = output.shape
print(result)
# (1, 2, 5)
x = torch.tensor(np_x, dtype=torch.float32)
max_pool = torch.nn.MaxPool1d(kernel_size=2, stride=1, padding=1, dilation=1, return_indices=False)
output = max_pool(x)
result = output.shape
print(result)
# torch.Size([1, 2, 5])
```

