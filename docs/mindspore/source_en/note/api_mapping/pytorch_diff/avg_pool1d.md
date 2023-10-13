# Differences with torch.nn.functional.avg_pool1d

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_en/note/api_mapping/pytorch_diff/avg_pool1d.md)

The following mapping relationships can be found in this file.

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
| torch.nn.functional.avg_pool1d | mindspore.ops.avg_pool1d |
| torch.nn.functional.avg_pool2d | mindspore.ops.avg_pool2d |
| torch.nn.functional.avg_pool3d | mindspore.ops.avg_pool3d |

## torch.nn.functional.avg_pool1d

```text
torch.nn.functional.avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
```

For more information, see [torch.nn.functional.avg_pool1d](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.avg_pool1d).

## mindspore.ops.avg_pool1d

```text
mindspore.ops.avg_pool1d(input_x, kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
```

For more information, see [mindspore.ops.avg_pool1d](https://mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.avg_pool1d.html).

## Differences

PyTorch: Perform average pooling operations on time series data.

MindSpore: MindSpore API function is basically the same as pytorch, with different default values for some inputs.

| Categories | Subcategories| PyTorch | MindSpore |Differences |
| ---- | ----- | ------- | --------- |------------------ |
| Parameters | Parameter 1 | input             | input_x           | Different parameter names |
|  | Parameter 2 | kernel_size       | kernel_size       | The pytorch parameter has no default value and the MindSpore parameter has a default value of 1. |
|  | Parameter 3 | stride            | stride            | The default value of pytorch parameter is None, which is consistent with kernel_size by default, and the default value of MindSpore Parameter is 1. |
|  | Parameter 4 | padding           | padding           |  |
|  | Parameter 5 | ceil_mode         | ceil_mode         |  |
|  | Parameter 6 | count_include_pad | count_include_pad |  |

### Code Example 1

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

input_x = Tensor([1, 2, 3, 4, 5, 6, 7], mindspore.float32)
output = ops.avg_pool1d(input_x, kernel_size=3, stride=2)
print(output)
# ValueError: For avg_pool1d, input must have 3 dim, but got 1.
```
