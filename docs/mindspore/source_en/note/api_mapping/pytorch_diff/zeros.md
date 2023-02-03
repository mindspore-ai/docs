# Function Differences with torch.zeros

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/zeros.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.zeros

```text
torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
```

For more information, see [torch.zeros](https://pytorch.org/docs/1.8.1/generated/torch.zeros.html).

## mindspore.ops.zeros

```text
mindspore.ops.zeros(size, dtype=dtype) -> Tensor
```

For more information, see [mindspore.ops.zeros](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.zeros.html).

## Differences

PyTorch: Generate a Tensor of size `*size` with a padding value of 0.

MindSpore: MindSpore API implements the same function as TensorFlow, and only the parameter names are different.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters  | Parameter 1 | size          | size      | MindSpore only supports input of int or tuple type |
|     | Parameter 2 | out           | -         | Not involved                        |
|     | Parameter 3 | layout        | -         | Not involved                        |
|     | Parameter 4 | device        | -         | Not involved                        |
|     | Parameter 5 | requires_grad | -         | Not involved                        |

### Code Example 1

The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
from torch import tensor

output = torch.zeros(2, 2, dtype=torch.float32)
print(output.numpy())
# [[0. 0.]
#  [0. 0.]]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
import mindspore as ms
from mindspore import Tensor

output = ops.zeros((2, 2), dtype=ms.float32).asnumpy()
print(output)
# [[0. 0.]
#  [0. 0.]]
```
