# Function Differences with torch.nn.functional.dropout3d

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/dropout3d.md)

## torch.nn.functional.dropout3d

```python
torch.nn.functional.dropout3d(input, p=0.5, training=True, inplace=False) -> Tensor
```

For more information, see [torch.nn.functional.dropout3d](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.dropout3d).

## mindspore.ops.dropout3d

```python
mindspore.ops.dropout3d(input, p=0.5, training=True) -> Tensor
```

For more information, see [mindspore.ops.dropout3d](https://www.mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.dropout3d.html).

## Differences

PyTorch: During training, dropout3d randomly zeroes some channels of the input tensor with probability p from a Bernoulli distribution, each channel will be zeroed out independently on every forward call which based on Bernoulli distribution probability p. Zeroing some channels of the input tensor is proved that it can effectively reduce over fitting and prevent neuronal coadaptation.

MindSpore: MindSpore API Basically achieves the same function as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ----|
| Parameters | Parameter 1 | input | input | Mindspore only supports a tensor with a rank of 5 as input |
|      | Parameter 2 | p | p | - |
|      | Parameter 3 | training | training | - |
|      | Parameter 4 | inplace| - | - |

### Code Example 1

```python
# PyTorch
import torch

input = torch.ones(2, 3, 2, 4)
output = torch.nn.functional.dropout3d(input)
print(output.shape)
# torch.Size([2, 3, 2, 4])

# MindSpore
import mindspore as ms
from mindspore import ops
from mindspore import Tensor
import numpy as np

input = Tensor(np.ones([2, 3, 2, 4]), ms.float32)
input = input.expand_dims(0)
output = ops.dropout3d(input)
output = output.squeeze(0)
print(output.shape)
# (2, 3, 2, 4)
```

### Code Example 2

```python
# PyTorch
import torch

input = torch.ones(1, 1, 2, 3, 2, 4)
output = torch.nn.functional.dropout3d(input)
print(output.shape)
# torch.Size([1, 1, 2, 3, 2, 4])

# MindSpore
import mindspore as ms
from mindspore import ops
from mindspore import Tensor
import numpy as np

input = Tensor(np.ones([1, 1, 2, 3, 2, 4]), ms.float32)
input = input.squeeze(0)
output = ops.dropout3d(input)
output = output.expand_dims(0)
print(output.shape)
# (1, 1, 2, 3, 2, 4)
```