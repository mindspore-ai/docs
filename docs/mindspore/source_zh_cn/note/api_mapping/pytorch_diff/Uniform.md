# 比较与torch.nn.init.uniform_的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Uniform.md)

## torch.nn.init.uniform_

```text
torch.nn.init.uniform_(
    tensor,
    a=0.0,
    b=1.0
) -> Tensor
```

更多内容详见[torch.nn.init.uniform_](https://pytorch.org/docs/1.8.1/nn.init.html#torch.nn.init.uniform_)。

## mindspore.ops.uniform

```text
mindspore.ops.uniform(shape, minval, maxval, seed=None, dtype=mstype.float32) -> Tensor
```

更多内容详见[mindspore.ops.uniform](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.uniform.html)。

## 差异对比

PyTorch：通过入参`a`和`b`分别指定均匀分布的上下界，即U(-a, b)。

MindSpore：通过入参`minval`和`maxval`分别指定均匀分布的上下界，即U(minval, maxval)，通过seed指定随机种子。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | tensor | shape         | Pytorch是一个n维Tensor，MindSpore则为shape或包裹shape的Tensor   |
|  | 参数2 | a       | minval          | 参数名不同，功能相似，指定生成随机值最小值   |
|  | 参数3 | b       | maxval         | 参数名不同，功能相似，指定生成随机值最大值 |
|  | 参数4 | -       | seed          | 指定随机种子 |
|  | 参数5 | -       | dtype         | 指定输入数据的类型，根据数据类型确定均匀分布生成数据是离散型或是连续型 |

## 代码示例

```python
# PyTorch
import torch
from torch import nn

w = torch.empty(3, 2)
output = nn.init.uniform_(w, a=1, b=4)
print(tuple(output.shape))
# (3, 2)

# MindSpore
import numpy as np
import mindspore
from mindspore import ops
from mindspore import Tensor

shape = (3,2)
minval = Tensor(1, mindspore.float32)
maxval = Tensor(4, mindspore.float32)
output = ops.uniform(shape, minval, maxval, dtype=mindspore.float32)
print(output.shape)
# Out：
# (3, 2)
```
