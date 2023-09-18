# Differences between torch.nn.AdaptiveAvgPool1d

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/AdaptiveAvgPool1d.md)

The following mapping relationships can be found in this file.

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
| torch.nn.AdaptiveAvgPool1d | mindspore.nn.AdaptiveAvgPool1d |
| torch.nn.functional.adaptive_avg_pool1d | mindspore.ops.adaptive_avg_pool1d |

## torch.nn.AdaptiveAvgPool1d

```text
torch.nn.AdaptiveAvgPool1d(output_size)(input) -> Tensor
```

For more information, see [torch.nn.AdaptiveAvgPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveAvgPool1d.html).

## mindspore.nn.AdaptiveAvgPool1d

```text
mindspore.nn.AdaptiveAvgPool1d(output_size)(input) -> Tensor
```

For more information, see [mindspore.nn.AdaptiveAvgPool1d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.AdaptiveAvgPool1d.html).

## Differences

PyTorch: Adaptive average pooling operation for temporal data, supporting 2D and 3D data.

MindSpore: This API in MindSpore currently only supports 3D data and requires the last dimension of the input data to be larger than that of the output data, and must divide the output_size.

| Categories | Subcategories| PyTorch | MindSpore |Differences |
| ---- | ----- | ------- | --------- |------------------ |
|Parameters | Parameter 1 | output_size | output_size | MindSpore requires the last dimension of the input data to be larger than that of the output data, and must divide the output_size. |
|Input | Single input | input | input | MindSpore currently only supports 3D data |

### Code Example 1

> For 3D data, perform adaptive average pooling operation on the data when the output length can be divisiable by the input length.

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
