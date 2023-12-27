# 比较与torch.nn.functional.avg_pool1d的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/avg_pool1d.md)

以下映射关系均可参考本文。

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
| torch.nn.functional.avg_pool1d | mindspore.ops.avg_pool1d |
| torch.nn.functional.avg_pool2d | mindspore.ops.avg_pool2d |
| torch.nn.functional.avg_pool3d | mindspore.ops.avg_pool3d |

## torch.nn.functional.avg_pool1d

```text
torch.nn.functional.avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
```

更多内容详见[torch.nn.functional.avg_pool1d](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.avg_pool1d)。

## mindspore.ops.avg_pool1d

```text
mindspore.ops.avg_pool1d(input_x, kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
```

更多内容详见[mindspore.ops.avg_pool1d](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.avg_pool1d.html)。

## 差异对比

PyTorch：对时序数据进行平均池化运算。

MindSpore：MindSpore此API功能与pytorch基本一致，部分输入默认值不同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 | input             | input_x           | 参数名不同 |
|  | 参数2 | kernel_size       | kernel_size       | pytorch参数无默认值，MindSpore参数默认值为1 |
|  | 参数3 | stride            | stride            | pytorch参数默认值为None，默认与kernel_size一致，MindSpore参数默认值为1 |
|  | 参数4 | padding           | padding           |  |
|  | 参数5 | ceil_mode         | ceil_mode         |  |
|  | 参数6 | count_include_pad | count_include_pad |  |

### 代码示例1

```python
# PyTorch
import torch
import numpy as np

input = torch.tensor([[[1, 2, 3, 4, 5, 6, 7]]], dtype=torch.float32)
output = torch.nn.functional.avg_pool1d(input, kernel_size=3, stride=2)
print(output)
# tensor([[[ 2.,  4.,  6.]]])

# MindSpore
import mindspore
from mindspore import Tensor, ops

input_x = Tensor([[[1, 2, 3, 4, 5, 6, 7]]], mindspore.float32)
output = ops.avg_pool1d(input_x, kernel_size=3, stride=2)
print(output)
# tensor([[[ 2. 4. 6.]]])
```
