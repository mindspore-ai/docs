# 比较与torch.nn.AvgPool1d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/AvgPool1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.nn.AvgPool1d

```text
torch.nn.AvgPool1d(
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True
)(input) -> Tensor
```

更多内容详见[torch.nn.AvgPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AvgPool1d.html)。

## mindspore.nn.AvgPool1d

```text
mindspore.nn.AvgPool1d(
    kernel_size=1,
    stride=1,
    pad_mode='valid'
)(x) -> Tensor
```

更多内容详见[mindspore.nn.AvgPool1d](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.AvgPool1d.html)。

## 差异对比

PyTorch：对输入的多维数据进行一维平面上的平均池化运算。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，MindSpore不存在padding、ceil_mode、count_include_pad参数，而PyTorch中无pad_mode参数。

| 分类 | 子类   | PyTorch           | MindSpore   | 差异                                                         |
| ---- | ------ | ----------------- | ----------- | ------------------------------------------------------------ |
| 参数 | 参数1  | kernel_size       | kernel_size | 功能一致，PyTorch无默认值                                    |
|      | 参数2  | stride            | stride      | 功能一致，参数默认值不同                                     |
|      | 参数3  | padding           | -           | PyTorch中此参数用于说明输入的每一条边补充0的层数，MindSpore无此参数 |
|      | 参数4  | ceil_mode         | -           | PyTorch中此参数用于决定输出shape: ($N$, $C$, $L_{out}$)中$L_{out}$为小数时，是取上界ceil值还是舍弃小数部分取floor值；MindSpore无此参数，默认取floor值 |
|      | 参数5  | count_include_pad | -           | PyTorch中此参数用于决定是否在平均计算中包括padding，MindSpore无此参数 |
|      | 参数6  | -                 | pad_mode    | MindSpore指定池化的填充方式，可选值为"same"或"valid"，PyTorch无此参数 |
| 输入 | 单输入 | input             | x           | 接口输入，功能一致，参数名不同                               |

### 代码示例

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
