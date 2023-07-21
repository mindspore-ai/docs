# Function Differences with torch.ones

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ones.md)

## torch.ones

```text
torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
```

For more information, see [torch.ones](https://pytorch.org/docs/1.8.1/generated/torch.ones.html).

## mindspore.ops.ones

```text
mindspore.ops.ones(shape, dtype=dtype) -> Tensor
```

For more information,
see [mindspore.ops.ones](https://mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.ones.html).

## Differences

PyTorch: Generate a Tensor of size `*size` with a padding value of 1.

MindSpore: MindSpore API implements the same function as TensorFlow, and only the parameter names are different.

| Categories | Subcategories | PyTorch       | MindSpore | Difference                                         |
|------------|---------------|---------------|-----------|----------------------------------------------------|
| Parameters | Parameter 1   | size          | shape     | MindSpore supports input of int, tuple or Tensor type |
|            | Parameter 2   | out           | -         | Not involved                                       |
|            | Parameter 3   | dtype         | dtype     | The parameter is consistent                        |
|            | Parameter 4   | layout        | -         | Not involved                                       |
|            | Parameter 5   | device        | -         | Not involved                                       |
|            | Parameter 6   | requires_grad | -         | Not involved                                       |

### Code Example 1

The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
from torch import tensor

output = torch.ones(2, 2, dtype=torch.float32)
print(output.numpy())
# [[1. 1.]
#  [1. 1.]]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
import mindspore as ms
from mindspore import Tensor

output = ops.ones((2, 2), dtype=ms.float32).asnumpy()
print(output)
# [[1. 1.]
#  [1. 1.]]
```
