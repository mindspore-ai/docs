# Function Differences with torch.cdist

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/cdist.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

## torch.cdist

```text
torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
```

For more information, see [torch.cdist](https://pytorch.org/docs/1.8.1/generated/torch.cdist.html).

## mindspore.ops.cdist

```text
mindspore.ops.cdist(x1, x2, p=2.0)
```

For more information, see [mindspore.ops.cdist](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.cdist.html).

## Differences

PyTorch: Compute the p-norm distance between each pair of column vectors of the two Tensors.

MindSpore: MindSpore API basically implements the same functionality as PyTorch, with a slight difference in accuracy.

| Categories | Subcategories | PyTorch | MindSpore | Differences  |
| --- |---------------|---------| --- |-------------|
| Parameters | Parameter 1 |x1 | x1 | -  |
| | Parameter 2 | x2 | x2 | - |
|  | Parameter 3 | p | p | - |
| | Parameter 4 | compute_mode | - | torch specifies whether to calculate the Euclidean distance by using matrix multiplication, which is not available in MindSpore |

### Code Example 1

```python
# PyTorch
import torch
import numpy as np

x =  torch.tensor(np.array([[1.0, 1.0], [2.0, 2.0]]).astype(np.float32))
y =  torch.tensor(np.array([[3.0, 3.0], [3.0, 3.0]]).astype(np.float32))
output = torch.cdist(x, y, 2.0)
print(output)
# tensor([[2.8284, 2.8284],
#         [1.4142, 1.4142]])

# MindSpore
import mindspore.numpy as np
from mindspore import Tensor
from mindspore import ops

x = Tensor(np.array([[1.0, 1.0], [2.0, 2.0]]).astype(np.float32))
y = Tensor(np.array([[3.0, 3.0], [3.0, 3.0]]).astype(np.float32))
output = ops.cdist(x, y, 2.0)
print(output)
# [[2.828427  2.828427 ]
#  [1.4142135 1.4142135]]

```
