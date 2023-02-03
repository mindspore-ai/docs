# 比较与torch.nn.AvgPool2d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/AvgPool2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.nn.AvgPool2d

```text
torch.nn.AvgPool2d(
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None
)(input) -> Tensor
```

更多内容详见[torch.nn.AvgPool2d](https://PyTorch.org/docs/1.8.1/generated/torch.nn.AvgPool2d.html)。

## mindspore.nn.AvgPool2d

```text
mindspore.nn.AvgPool2d(
    kernel_size=1,
    stride=1,
    pad_mode='valid',
    data_format='NCHW'
)(x) -> Tensor
```

更多内容详见[mindspore.nn.AvgPool2d](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.AvgPool2d.html)。

## 差异对比

PyTorch：对由多个输入平面组成的输入信号应用二维平均池化。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。

| 分类 | 子类   | PyTorch               | MindSpore   | 差异                                                         |
| ---- | ------ | --------------------- | ----------- | ------------------------------------------------------------ |
| 参数 | 参数1  | kernel_size           | kernel_size | 功能一致，PyTorch无默认值                                    |
|      | 参数2  | stride                | stride      | 功能一致，参数默认值不同                                     |
|      | 参数3  | padding               | -           | PyTorch中此参数用于添加隐式零填充，MindSpore无此参数         |
|      | 参数4  | ceil_mode             | -           | PyTorch中此参数用于决定输出shape: ($N$,$C$,$H_{out}$,$W_{out}$)中$H_{out}$,$W_{out}$为小数时，是取上界ceil值还是舍弃小数部分取floor值；MindSpore无此参数，默认取floor值 |
|      | 参数5  | count_include_pad     | -           | PyTorch中此参数用于决定是否在平均计算中包括零填充，MindSpore无此参数 |
|      | 参数6  | divisor_override=None | -           | PyTorch中如果指定，它将被用作除数，否则将使用kernel_size，MindSpore无此参数 |
|      | 参数7  | -                     | pad_mode    | MindSpore中指定池化填充模式，可选值为"same"或"valid"，PyTorch无此参数 |
|      | 参数8  | -                     | data_format | MindSpore中指定输入数据格式，值可为"NHWC"或"NCHW"，PyTorch无此参数 |
| 输入 | 单输入 | input                 | x           | 功能一致，参数名不同                               |

### 代码示例

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
import torch.nn as nn

m = nn.AvgPool2d(kernel_size=1, stride=1)
input_x = torch.tensor([[[[1, 0, 1], [0, 1, 1]]]],dtype=torch.float32)
output = m(input_x)
print(output.numpy())
# [[[[1. 0. 1.]
#    [0. 1. 1.]]]]

# MindSpore
import mindspore
import mindspore.nn as nn
from mindspore import Tensor

pool = nn.AvgPool2d(kernel_size=1, stride=1)
x = Tensor([[[[1, 0, 1], [0, 1, 1]]]], dtype=mindspore.float32)
output = pool(x)
print(output)
# [[[[1. 0. 1.]
#    [0. 1. 1.]]]]
```
