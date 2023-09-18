# Differences with torch.broadcast_to

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/broadcast_to.md)

The following mapping relationships can be found in this file.

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|    torch.broadcast_to     |  mindspore.ops.broadcast_to   |
|   torch.Tensor.broadcast_to    |   mindspore.Tensor.broadcast_to    |

## torch.broadcast_to

```text
torch.broadcast_to(input, shape) -> Tensor
```

For more information, see [torch.broadcast_to](https://pytorch.org/docs/1.8.1/generated/torch.broadcast_to.html).

## mindspore.ops.broadcast_to

```text
mindspore.ops.broadcast_to(input, shape) -> Tensor
```

For more information, see [mindspore.ops.broadcast_to](https://mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.broadcast_to.html).

## Differences

PyTorch: Broadcast the input shape to the target shape.

MindSpore: MindSpore API basically implements the same function as PyTorch, with additional support for the -1 dimension in the shape. If there is a -1 dimension in the target shape, it is replaced by the value of the input shape in that dimension. If there is a -1 dimension in the target shape, the -1 dimension cannot be located in a dimension that does not exist.

| Categories | Subcategories | PyTorch | MindSpore | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Input | Single input | input | input | Same function, different parameter names |
|Parameter | Parameter 1 | shape | shape |Same function |

### Code Example 1

```python
# PyTorch
import torch

shape = (2, 3)
x = torch.tensor([[1], [2]]).float()
torch_output = torch.broadcast_to(x, shape)
print(torch_output.numpy())
# [[1. 1. 1.]
#  [2. 2. 2.]]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

shape = (2, 3)
x = Tensor(np.array([[1], [2]]).astype(np.float32))
output = mindspore.ops.function.broadcast_to(x, shape)
print(output)
# [[1. 1. 1.]
#  [2. 2. 2.]]
```
