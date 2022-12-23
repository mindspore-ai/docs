# 比较与torch.nn.MaxPool1d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/MaxPool1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.MaxPool1d

```text
class torch.nn.MaxPool1d(
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    return_indices=False,
    ceil_mode=False
)(input) -> Tensor
```

更多内容详见[torch.nn.MaxPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxPool1d.html)。

## mindspore.nn.MaxPool1d

```text
class mindspore.nn.MaxPool1d(
    kernel_size=1,
    stride=1,
    pad_mode='valid'
)(x) -> Tensor
```

更多内容详见[mindspore.nn.MaxPool1d](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.MaxPool1d.html)。

## 差异对比

PyTorch：对时间数据进行最大池化运算。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，但参数设定上缺少padding，dilation，return_indices的功能。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | kernel_size | kernel_size | 功能一致，PyTorch无默认值 |
| | 参数2 | stride | stride | 功能一致，默认值不同  |
| | 参数3 | padding | - | 填充元素个数。默认值为0（不填充），值不能超过kernel_size/2（向下取值），更多内容详见[Conv 和 Pooling](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/typical_api_comparision.html#conv-%E5%92%8C-pooling) |
| | 参数4 | dilation | - | 窗口内元素间跨步长度：默认值为1，此时窗口内的元素是连续的。若值>1，窗口中的元素是间隔的 |
| | 参数5 | return_indices | - | 返回索引：若值为True，会在返回最大池化结果的同时返回对应元素的索引。对于后续调用torch.nn.MaxUnpool1d的时候很有用|
| | 参数6 | ceil_mode | - | 控制输出shape(N, C, L_{out})中L_{out}向上取整还是向下取整，MindSpore默认向下取整 |
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

> ceil_mode=True和pad_mode='same'时，两API实现相同的功能。

```python
# PyTorch
import torch

max_pool = torch.nn.MaxPool1d(kernel_size=3, stride=2, ceil_mode=True)
x = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], dtype=torch.float32)
output = max_pool(x)
print(output.numpy())
# [[[ 3.  5.  7.  9. 10.]
#   [ 3.  5.  7.  9. 10.]]]

# MindSpore
import mindspore
from mindspore import Tensor

max_pool = mindspore.nn.MaxPool1d(kernel_size=3, stride=2, pad_mode='same')
x = Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], mindspore.float32)
output = max_pool(x)
print(output)
# [[[ 3.  5.  7.  9. 10.]
#   [ 3.  5.  7.  9. 10.]]]
```

### 代码示例3

> 在PyTorch中，当ceil_mode=False时，设置padding=1，在MindSpore中pad_mode='valid'，先通过ops.pad()对x进行填充，再计算最大池化的结果，使两API实现相同的功能。

```python
# PyTorch
import torch

max_pool = torch.nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
x = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], dtype=torch.float32)
output = max_pool(x)
result = output.shape
print(output.numpy())
# [[[ 3.  5.  7.  9. 10.]
#   [ 3.  5.  7.  9. 10.]]]
print(tuple(result))
# (1, 2, 5)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

max_pool = mindspore.nn.MaxPool1d(kernel_size=4, stride=2)
x = Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], mindspore.float32)
data = ops.pad(x, ((0, 0), (0, 0), (1, 1)))
output = max_pool(data)
result = output.shape
print(output)
# [[[ 3.  5.  7.  9. 10.]
#   [ 3.  5.  7.  9. 10.]]]
print(result)
# (1, 2, 5)
```

