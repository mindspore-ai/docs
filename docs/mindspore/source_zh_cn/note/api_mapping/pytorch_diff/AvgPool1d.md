# 比较与torch.nn.AvgPool1d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/AvgPool1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

更多内容详见[mindspore.nn.AvgPool1d](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.AvgPool1d.html)。

## 差异对比

PyTorch：对输入的多维数据进行一维平面上的平均池化运算。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，MindSpore不存在padding、ceil_mode、count_include_pad参数，而PyTorch中无pad_mode参数。

| 分类 | 子类  | PyTorch           | MindSpore   | 差异                                                         |
| ---- | ----- | ----------------- | ----------- | ------------------------------------------------------------ |
| 参数 | 参数1 | kernel_size       | kernel_size | 功能一致，PyTorch无默认值                                                            |
|      | 参数2 | stride            | stride      | 功能一致，参数无默认值不同                                                            |
|      | 参数3 | padding           | -           | PyTorch中此参数用于说明输入的每一条边补充0的层数，MindSpore无此参数 |
|      | 参数4 | ceil_mode         | -           | PyTorch中此参数用于决定输出shape: (N, C, L{out})中L{out}为小数时，是取上界ceil值还是舍弃小数部分取floor值；MindSpore无此参数，默认取floor值 |
|      | 参数5 | count_include_pad | -           | PyTorch中此参数用于决定是否在平均计算中包括padding，MindSpore无此参数 |
|      | 参数6 | input             | x           | 功能一致，参数名不同                                         |
|      | 参数7 | -                 | pad_mode    | MindSpore指定池化的填充方式，可选值为"same"或"valid"，PyTorch无此参数 |

### 代码示例1

> 当不涉及到padding、count_include_pad、pad_mode参数时，两API实现功能一致，用法相同。

```python
# PyTorch
import torch
import torch.nn as nn

m = nn.AvgPool1d(kernel_size=6, stride=1)
input = torch.tensor([[[1.,2,3,4,5,6,7]]], dtype=torch.float32)
print(input.numpy())
# [[[1. 2. 3. 4. 5. 6. 7.]]]
print(m(input).numpy())
# [[[3.5 4.5]]]

# MindSpore
import mindspore
import mindspore.nn as nn
from mindspore import Tensor

pool = nn.AvgPool1d(kernel_size=6, stride=1)
x = Tensor([[[1.,2,3,4,5,6,7]]], dtype=mindspore.float32)
print(x)
# [[[1. 2. 3. 4. 5. 6. 7.]]]
output = pool(x)
print(output)
# [[[3.5 4.5]]]
```

### 代码示例2

> torch.nn.AvgPool1d可以通过参数ceil_mode来决定输出形状Output: (N, C, L{out})中L{out}为小数时，是取上界ceil值还是舍弃小数部分取floor值，而mindspore.nn.AvgPool1d会默认取floor值，与PyTorch存在差异。

```python
#PyTorch
import torch
import torch.nn as nn

m = nn.AvgPool1d(kernel_size=4, stride=2, padding=0, ceil_mode=False)
input = torch.tensor([[[1.,2,3,4,5,6,7]]], dtype=torch.float32)
print(input.numpy())
# [[[1. 2. 3. 4. 5. 6. 7.]]]
print(m(input).numpy())
# [[[2.5 4.5]]]

#MindSpore
import mindspore
import mindspore.nn as nn
from mindspore import Tensor

pool = nn.AvgPool1d(kernel_size=4, stride=2, pad_mode='valid')
x = Tensor([[[1.,2,3,4,5,6,7]]], dtype=mindspore.float32)
print(x)
# [[[1. 2. 3. 4. 5. 6. 7.]]]
output = pool(x)
print(output)
# [[[2.5 4.5]]]
```
