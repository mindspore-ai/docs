# Differences with torch.nn.MaxPool3d

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/MaxPool3d.md)

## torch.nn.MaxPool3d

```text
torch.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)(input) -> Tensor
```

For more information, see [torch.nn.MaxPool3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxPool3d.html).

## mindspore.nn.MaxPool3d

```text
mindspore.nn.MaxPool3d(kernel_size=1, stride=1, pad_mode="valid", padding=0, dilation=1, return_indices=False, ceil_mode=False)(x) -> Tensor
```

For more information, see [mindspore.nn.MaxPool3d](https://www.mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.MaxPool3d.html).

## Differences

PyTorch: Perform three-dimensional maximum pooling operations on the input multidimensional data.

MindSpore: This API implementation function of MindSpore is compatible with TensorFlow and PyTorch, When `pad_mode` is "valid" or "same", the function is consistent with TensorFlow, and when `pad_mode` is "pad", the function is consistent with PyTorch, MindSpore additionally supports 2D input, which is consistent with PyTorch 1.12.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | kernel_size | kernel_size |Consistent function, no default values for PyTorch |
| | Parameter 2 | stride | stride |Consistent function, different default value |
| | Parameter 3 | padding |padding| Consistent |
| | Parameter 4 | dilation | dilation | Consistent |
| | Parameter 5 | return_indices | return_indices | Consistent|
| | Parameter 6 | ceil_mode | ceil_mode | Consistent |
| | Parameter 7 | input | x | Consistent function, different parameter names |
| | Parameter 8 | - | pad_mode | Control the padding mode, and PyTorch does not have this parameter |

## Code Example

> Use pad mode to ensure functional consistency.

```python
import mindspore as ms
from mindspore import Tensor
import mindspore.nn as nn
import torch
import numpy as np

np_x = np.random.randint(0, 10, [1, 2, 4, 4, 5])

x = Tensor(np_x, ms.float32)
max_pool = nn.MaxPool3d(kernel_size=2, stride=1, pad_mode='pad', padding=1, dilation=1, return_indices=False)
output = max_pool(x)
result = output.shape
print(result)
# (1, 2, 5, 5, 6)
x = torch.tensor(np_x, dtype=torch.float32)
max_pool = torch.nn.MaxPool3d(kernel_size=2, stride=1, padding=1, dilation=1, return_indices=False)
output = max_pool(x)
result = output.shape
print(result)
# torch.Size([1, 2, 5, 5, 6])
```