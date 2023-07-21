# 比较与torch.nn.AdaptiveAvgPool1d的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/AdaptiveAvgPool1d.md)

以下映射关系均可参考本文。

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
| torch.nn.AdaptiveAvgPool1d | mindspore.nn.AdaptiveAvgPool1d |
| torch.nn.functional.adaptive_avg_pool1d | mindspore.ops.adaptive_avg_pool1d |

## torch.nn.AdaptiveAvgPool1d

```text
torch.nn.AdaptiveAvgPool1d(output_size)(input) -> Tensor
```

更多内容详见[torch.nn.AdaptiveAvgPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveAvgPool1d.html)。

## mindspore.nn.AdaptiveAvgPool1d

```text
mindspore.nn.AdaptiveAvgPool1d(output_size)(input) -> Tensor
```

更多内容详见[mindspore.nn.AdaptiveAvgPool1d](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.AdaptiveAvgPool1d.html)。

## 差异对比

PyTorch：对时序数据进行自适应平均池化运算，支持2D和3D数据。

MindSpore：MindSpore此API目前只支持3D数据，要求输入数据的最后一个维度长度要大于输出大小，并且必须整除output_size。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | output_size | output_size | 要求输入数据的最后一个维度长度要大于输出大小，并且必须整除output_size |
|输入 | 单输入 | input | input | MindSpore目前只支持3D数据 |

### 代码示例1

> 对三维数据，在输入长度可以整除输出长度时对数据进行自适应平均池化运算。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

avg_pool = torch.nn.AdaptiveAvgPool1d(output_size=4)
x = tensor(np.arange(16).reshape(1, 2, 8), dtype=torch.float32)
output = avg_pool(x)
print(output)
# tensor([[[ 0.5000,  2.5000,  4.5000,  6.5000],
#          [ 8.5000, 10.5000, 12.5000, 14.5000]]])

# MindSpore
import mindspore
from mindspore import Tensor, nn
import numpy as np
pool = nn.AdaptiveAvgPool1d(output_size=4)
x = Tensor(np.arange(16).reshape(1, 2, 8), mindspore.float32)
output = pool(x)
print(output)
# [[[ 0.5  2.5  4.5  6.5]
#   [ 8.5 10.5 12.5 14.5]]]
```

