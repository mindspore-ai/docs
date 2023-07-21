# 比较与torch.nn.functional.grid_sample的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/grid_sample.md)

## torch.nn.functional.grid_sample

```text
torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zero', align_corners=None) -> Tensor
```

更多内容详见[torch.nn.functional.grid_sample](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.grid_sample)。

## mindspore.ops.grid_sample

```text
mindspore.ops.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
```

更多内容详见[mindspore.ops.grid_sample](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.grid_sample.html)。

## 差异对比

PyTorch：给定一个输入和一个网格，使用网格中的输入值和像素位置计算输出。input 只支持4-D（GridSampler2D）和5-D（GridSampler3D）。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，不过暂不支持mode为“bicubic”。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                    |
| ---- | ----- | ------- | --------- | --------------------------------------- |
| 参数 | 参数1 | input   | input     | 功能一致            |
|      | 参数2 | grid   | grid      | 功能一致 |
|      | 参数3 | mode   | mode | 功能一致，MindSpore暂不支持“bicubic” |
|      | 参数4 | padding_mode | padding_mode | 功能一致 |
|      | 参数5 | align_corners | align_corners | 功能一致 |

### 代码示例1

```python
# PyTorch
import torch
from torch import tensor
import numpy as np
input_x = tensor(np.arange(16).reshape((2, 2, 2, 2)).astype(np.float32))
grid = tensor(np.arange(0.2, 1, 0.1).reshape((2, 2, 1, 2)).astype(np.float32))
output = torch.nn.functional.grid_sample(input_x, grid)
print(output)
#tensor([[[[ 2.3000],
#          [ 2.9000]],
#
#         [[ 6.3000],
#          [ 6.9000]]],
#
#
#        [[[ 7.9200],
#          [ 4.6200]],
#
#         [[10.8000],
#          [ 6.3000]]]])

# MindSpore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
input_x = Tensor(np.arange(16).reshape((2, 2, 2, 2)).astype(np.float32))
grid = Tensor(np.arange(0.2, 1, 0.1).reshape((2, 2, 1, 2)).astype(np.float32))
output = ops.grid_sample(input_x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
print(output)
#[[[[ 2.3      ]
#   [ 2.8999999]]
#
#  [[ 6.3      ]
#   [ 6.8999996]]]
#
#
# [[[ 7.919999 ]
#   [ 4.6200004]]
#
#  [[10.799998 ]
#   [ 6.3000007]]]]
```
