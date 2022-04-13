# # Function Differences with torch.nn.functional.pad

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Pad.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [mindspore.nn.Pad](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Pad.html#mindspore.nn.Pad).

## Differences

PyTorch：The pad parameter is a tuple with m values, m/2 is less than or equal to the dimension of the input data, and m is even. Negative dimensions are supported. Assuming pad=(k1, k2, ..., kl, km), the shape of the input x is (d1, d2..., dg), then the two sides of the dg dimension are filled with the values of lengths k1 and k2 respectively. Similarly, the two sides of the d1 dimension are filled with the values of length kl and km respectively.

MindSpore：The paddings parameter is a tuple whose shape is (n, 2), n is the dimension of the input data. For the D dimension of the input x, the size of the corresponding output D dimension is equal to paddings[D, 0] + x.dim_size(D) + paddings[D, 1]. Negative dimensions are not supported currently, and can be cut into smaller slice by ops.Slice. Assuming that the shape of the input x is (1, 2, 2, 3), and the pad parameter of Pytorch is (1, 1, 2, 2), to make the output shape of MindSpore consistent with that of Pytorch, the paddings parameter should be ((0, 0), (0, 0), (2, 2), (1, 1)), the output shape is (1, 2, 6, 5).

## Code Example

```python
# In MindSpore.
import numpy as np
import torch
import mindspore.nn as nn
from mindspore import Tensor

x = Tensor(np.ones([1, 2, 2, 3]).astype(np.float32))
pad_op = nn.Pad(paddings=((0, 0), (0, 0), (2, 2), (1, 1)))
output = pad_op(x)
print(output.shape)
# Out:
# (1, 2, 6, 5)

# In Pytorch.
x = torch.empty(1, 2, 2, 3)
pad = (1, 1, 2, 2)
output = torch.nn.functional.pad(x, pad)
print(output.size())
# Out:
# torch.Size([1, 2, 6, 5])
```
