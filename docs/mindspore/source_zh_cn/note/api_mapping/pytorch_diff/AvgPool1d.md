# 比较与torch.nn.AvgPool1d的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/AvgPool1d.md)

## torch.nn.AvgPool1d

```text
torch.nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)(input) -> Tensor
```

更多内容详见[torch.nn.AvgPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AvgPool1d.html)。

## mindspore.nn.AvgPool1d

```text
mindspore.nn.AvgPool1d(kernel_size=1, stride=1, pad_mode="valid", padding=0, ceil_mode=False, count_include_pad=True)(x) -> Tensor
```

更多内容详见[mindspore.nn.AvgPool1d](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/nn/mindspore.nn.AvgPool1d.html)。

## 差异对比

PyTorch：对输入的多维数据进行一维平面上的平均池化运算。

MindSpore：MindSpore此API实现功能同时兼容TensorFlow和PyTorch，`pad_mode` 为 "valid" 或者 "same" 时，功能与TensorFlow一致，`pad_mode` 为 "pad" 时，功能与PyTorch一致，MindSpore相比PyTorch1.8.1额外支持了维度为2的输入，与PyTorch1.12一致。

| 分类 | 子类   | PyTorch           | MindSpore   | 差异                                                         |
| ---- | ------ | ----------------- | ----------- | ------------------------------------------------------------ |
| 参数 | 参数1  | kernel_size       | kernel_size | 功能一致，PyTorch无默认值                                    |
|      | 参数2  | stride            | stride      | 功能一致，参数默认值不同                                     |
|      | 参数3  | padding           | padding  | 功能一致 |
|      | 参数4  | ceil_mode         | ceil_mode   | 功能一致 |
|      | 参数5  | count_include_pad | count_include_pad   | 功能一致 |
|      | 参数6  | -                 | pad_mode    | MindSpore指定池化的填充方式，可选值为"same"，"valid" 或者 "pad"，PyTorch无此参数 |
| 输入 | 单输入 | input             | x           | 接口输入，功能一致，参数名不同                               |

### 代码示例1

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
import torch.nn as nn

m = nn.AvgPool1d(kernel_size=6, stride=1)
input_x = torch.tensor([[[1,2,3,4,5,6,7]]], dtype=torch.float32)
print(m(input_x).numpy())
# [[[3.5 4.5]]]

# MindSpore
import mindspore
import mindspore.nn as nn
from mindspore import Tensor

pool = nn.AvgPool1d(kernel_size=6, stride=1)
x = Tensor([[[1,2,3,4,5,6,7]]], dtype=mindspore.float32)
output = pool(x)
print(output)
# [[[3.5 4.5]]]
```

### 代码示例2

> 使用pad模式保证功能一致。

```python
import torch
import mindspore.nn as nn
import mindspore.ops as ops

pool = nn.AvgPool1d(4, stride=1, ceil_mode=True, pad_mode='pad', padding=2)
x1 = ops.randn(6, 6, 8)
output = pool(x1)
print(output.shape)
# (6, 6, 9)

pool = torch.nn.AvgPool1d(4, stride=1, ceil_mode=True, padding=2)
x1 = torch.randn(6, 6, 8)
output = pool(x1)
print(output.shape)
# torch.Size([6, 6, 9])
```