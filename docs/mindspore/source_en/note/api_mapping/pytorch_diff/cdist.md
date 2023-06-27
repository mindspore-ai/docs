# Differences with torch.cdist

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/cdist.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

MindSpore is basically the same as PyTorch, but MindSpore cannot specify whether to compute the Euclidean distance between vector pairs using matrix multiplication.

PyTorch: When the parameter `compute_mode` is ``use_mm_for_euclid_dist_if_necessary`` and the number of row vectors in a batch of `x1` or `x2` exceeds 25, the Euclidean distance between vector pairs is calculated using matrix multiplication. When `compute_mode` is ``use_mm_for_euclid_dist``, the Euclidean distance between vector pairs is calculated using matrix multiplication. When `compute_mode` is ``donot_use_mm_for_euclid_dist``, the Euclidean distances between vector pairs are not computed using matrix multiplication.

MindSpore: No parameter `compute_mode` to specify whether to use matrix multiplication to compute the Euclidean distance between vector pairs. Euclidean distances between vector pairs are not computed using matrix multiplication on ``GPU`` and ``CPU``, while on ``Ascend``, Euclidean distances between vector pairs are computed using matrix multiplication.

| Categories | Subcategories | PyTorch | MindSpore | Differences  |
| --- |---------------|---------| --- |-------------|
| Parameters | Parameter 1 |x1 | x1 | -  |
| | Parameter 2 | x2 | x2 | - |
|  | Parameter 3 | p | p | - |
| | Parameter 4 | compute_mode | - | A parameter in PyTorch specifying whether to calculate Euclidean distances by matrix multiplication, which is not available in MindSpore |

### Code Example

```python
# PyTorch
import torch
import numpy as np

torch.set_printoptions(precision=7)
x =  torch.tensor(np.array([[1.0, 1.0], [2.0, 2.0]]).astype(np.float32))
y =  torch.tensor(np.array([[3.0, 3.0], [3.0, 3.0]]).astype(np.float32))
output = torch.cdist(x, y, 2.0)
print(output)
# tensor([[2.8284271, 2.8284271],
#         [1.4142135, 1.4142135]])

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
