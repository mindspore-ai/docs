# Differences with torch.nn.functional.grid_sample

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/grid_sample.md)

## torch.nn.functional.grid_sample

```text
torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zero', align_corners=None) -> Tensor
```

For more information, see [torch.nn.functional.grid_sample](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.grid_sample).

## mindspore.ops.grid_sample

```text
mindspore.ops.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
```

For more information, see [mindspore.ops.grid_sample](https://www.mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.grid_sample.html).

## Differences

PyTorch: Given an input and a flow-field grid, computes the output using input values and pixel locations from grid. Only spatial (4-D) and volumetric (5-D) input is supported.

MindSpore: MindSpore API implements functions basically same as PyTorch, but the mode of "bicubic" is not supported yet in MindSpore.

| Categories | Subcategories| PyTorch | MindSpore |Differences |
| ---- | ----- | ------- | --------- |------------------ |
| Parameters | Parameter 1 | input   | input     | Same function                   |
|      | Parameter 2 | grid   | grid | Same function |
|      | Parameter 3 | mode   | mode | Same function, MindSpore does not support "bicubic" mode yet  |
|      | Parameter 4 | padding_mode  | padding_mode   | Same function  |
|      | Parameter 5 | align_corners | align_corners  | Same function  |

### Code Example 1

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
