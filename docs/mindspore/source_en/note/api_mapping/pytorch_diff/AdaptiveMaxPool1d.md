# Function Differences between torch.nn.AdaptiveMaxPool1d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/AdaptiveMaxPool1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The following mapping relationships can be found in this file.

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
| torch.nn.AdaptiveMaxPool1d | mindspore.nn.AdaptiveMaxPool1d |
| torch.nn.functional.adaptive_max_pool1d | mindspore.ops.adaptive_max_pool1d |

## torch.nn.AdaptiveMaxPool1d

```text
torch.nn.AdaptiveMaxPool1d(output_size, return_indices=False)(input) -> Tensor
```

For more information, see [torch.nn.AdaptiveMaxPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveMaxPool1d.html).

## mindspore.nn.AdaptiveMaxPool1d

```text
mindspore.nn.AdaptiveMaxPool1d(output_size)(x) -> Tensor
```

For more information, see [mindspore.nn.AdaptiveMaxPool1d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.AdaptiveMaxPool1d.html).

## Differences

PyTorch: Adaptive max pooling operation for temporal data, supporting 2D and 3D data.

MindSpore: This API in MindSpore currently only supports 3D data and requires the last dimension of the input data to be larger than that of the output data, and must divide the output_size. Currently not support index subscripts that return the maximum value.

| Categories | Subcategories| PyTorch | MindSpore |Differences |
| ---- | ----- | ------- | --------- |------------------ |
|Parameters | Parameter 1 | output_size | output_size | MindSpore currently only supports 3D data and requires the length of the last dimension of the input data to be divisible by output_size |
| | Parameter 2 | return_indices | - | MindSpore does not have this parameter and does not support index subscripts that return the maximum value currently.  |
|Input | Single input | input | x | Same function, different parameter names |

### Code Example 1

> For 3D data, perform adaptive maximum pooling operation on the data when the output length can be divisiable by the input length.

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
