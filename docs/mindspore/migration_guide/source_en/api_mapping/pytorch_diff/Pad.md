# # Function Differences with torch.nn.functional.pad

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/Pad.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## torch.nn.functional.pad

```python
class torch.nn.functional.pad(
    input
    pad,
    mode='constant',
    value=0.0
)
```

For more information, see [torch.nn.functional.pad](https://pytorch.org/docs/1.5.0/nn.functional.html#torch.nn.functional.pad).

## mindspore.nn.Pad

```python
class mindspore.nn.Pad(
    paddings,
    mode="CONSTANT"
)(x)
```

For more information, see [mindspore.nn.Pad](https://mindspore.cn/docs/api/en/r1.5/api_python/nn/mindspore.nn.Pad.html#mindspore.nn.Pad).

## Differences

PyTorch：The pad parameter is a tuple with m values, m/2 is less than or equal to the dimension of the input data, and m is even. Negative dimensions are supported.

MindSpore：The paddings parameter is a tuple whose shape is (n, 2), n is the dimension of the input data. Negative dimensions are not supported currently, and can be cut into smaller slice by ops.Slice.

## Code Example

```python
# In MindSpore.
import numpy as np
import torch
import mindspore.nn as nn
from mindspore import Tensor

x = Tensor(np.ones(3, 3).astype(np.float32))
pad_op = nn.Pad(paddings=((0, 0), (1, 1)))
output = pad_op(x)
print(output.shape)
# Out:
# (3, 5)

# In Pytorch.
x = torch.empty(3, 3)
pad = (1, 1)
output = torch.nn.functional.pad(x, pad)
print(output.size())
# Out:
# (3, 5)
```
