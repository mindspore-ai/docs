# Differences with torch.nn.functional.dropout2d

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/dropout2d.md)

## torch.nn.functional.dropout2d

```python
torch.nn.functional.dropout2d(input, p=0.5, training=True, inplace=False) -> Tensor
```

For more information, see [torch.nn.functional.dropout2d](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.dropout2d).

## mindspore.ops.dropout2d

```python
mindspore.ops.dropout2d(input, p=0.5, training=True) -> Tensor
```

For more information, see [mindspore.ops.dropout2d](https://www.mindspore.cn/docs/en/r2.3/api_python/ops/mindspore.ops.dropout2d.html).

## Differences

API function of MindSpore is consistent with that of PyTorch, with differences in the supported data types for parameters.

PyTorch: During training, dropout2d randomly zeroes some channels of the input tensor with probability p from a Bernoulli distribution, each channel will be zeroed out independently on every forward call which based on Bernoulli distribution probability p. Zeroing some channels of the input tensor is proved that it can effectively reduce over fitting and prevent neuronal coadaptation.

MindSpore: Mindspore only supports a tensor with a rank of 4 as input.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ----|
| Parameters | Parameter 1 | input | input | Mindspore only supports a tensor with a rank of 4 as input |
|      | Parameter 2 | p | p | - |
|      | Parameter 3 | training | training | - |
|      | Parameter 4 | inplace| - | - |

### Code Example 1

```python
# PyTorch
import torch

input = torch.ones(3, 2, 4)
output = torch.nn.functional.dropout2d(input)
print(output.shape)
# torch.Size([3, 2, 4])

# MindSpore
import mindspore as ms
from mindspore import ops
from mindspore import Tensor
import numpy as np

input = Tensor(np.ones([3, 2, 4]), ms.float32)
input = input.expand_dims(0)
output = ops.dropout2d(input)
output = output.squeeze(0)
print(output.shape)
# (3, 2, 4)
```

### Code Example 2

```python
# PyTorch
import torch

input = torch.ones(1, 2, 3, 2, 4)
output = torch.nn.functional.dropout2d(input)
print(output.shape)
# torch.Size([1, 2, 3, 2, 4])

# MindSpore
import mindspore as ms
from mindspore import ops
from mindspore import Tensor
import numpy as np

input = Tensor(np.ones([1, 2, 3, 2, 4]), ms.float32)
input = input.squeeze(0)
output = ops.dropout2d(input)
output = output.expand_dims(0)
print(output.shape)
# (1, 2, 3, 2, 4)
```
