# 比较与torch.nn.AdaptiveMaxPool1d的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/AdaptiveMaxPool1d.md)

以下映射关系均可参考本文。

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
| torch.nn.AdaptiveMaxPool1d | mindspore.nn.AdaptiveMaxPool1d |
| torch.nn.functional.adaptive_max_pool1d | mindspore.ops.adaptive_max_pool1d |

## torch.nn.AdaptiveMaxPool1d

```text
torch.nn.AdaptiveMaxPool1d(output_size, return_indices=False)(input) -> Tensor
```

更多内容详见[torch.nn.AdaptiveMaxPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveMaxPool1d.html)。

## mindspore.nn.AdaptiveMaxPool1d

```text
mindspore.nn.AdaptiveMaxPool1d(output_size)(x) -> Tensor
```

更多内容详见[mindspore.nn.AdaptiveMaxPool1d](https://www.mindspore.cn/docs/zh-CN/r1.11/api_python/nn/mindspore.nn.AdaptiveMaxPool1d.html)。

## 差异对比

PyTorch：对时间数据进行自适应最大池化运算，支持2D和3D数据。

MindSpore：MindSpore此API目前只支持3D数据，要求输入数据的最后一个维度长度要大于输出大小，并且必须整除output_size；目前不支持返回最大值的索引下标。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | output_size | output_size | MindSpore目前只支持3D数据，并且要求输入数据的最后一个维度的长度必须整除output_size |
| | 参数2 | return_indices | - | MindSpore无此参数，暂不支持返回最大值的索引下标  |
|输入 | 单输入 | input | x | 功能一致，参数名不同 |

### 代码示例1

> 对三维数据，在输入长度可以整除输出长度时对数据进行自适应最大池化运算。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

max_pool = torch.nn.AdaptiveMaxPool1d(output_size=4)
x = tensor(np.arange(16).reshape(1, 2, 8), dtype=torch.float32)
output = max_pool(x)
print(output)
# tensor([[[ 1.,  3.,  5.,  7.],
#          [ 9., 11., 13., 15.]]])

# MindSpore
import mindspore
from mindspore import Tensor, nn
import numpy as np
pool = nn.AdaptiveMaxPool1d(output_size=4)
x = Tensor(np.arange(16).reshape(1, 2, 8), mindspore.float32)
output = pool(x)
print(output)
# [[[ 1.  3.  5.  7.]
#   [ 9. 11. 13. 15.]]]
```

