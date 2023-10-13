# Differences with torch.meshgrid

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_en/note/api_mapping/pytorch_diff/meshgrid.md)

## torch.meshgrid

```text
torch.meshgrid(
    *tensors)
```

For more information, see [torch.meshgrid](https://pytorch.org/docs/1.8.1/generated/torch.meshgrid.html).

## mindspore.ops.meshgrid

```text
mindspore.ops.meshgrid(*inputs, indexing='xy')
```

For more information, see [mindspore.ops.meshgrid](https://mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.meshgrid.html).

## Differences

PyTorch: Generate the grid matrix from the given tensors. If it is a list of scalars, the scalar will automatically be considered as a tensor of size (1,).

MindSpore: MindSpore API basically implements the same function as TensorFlow. The inputs parameter is only supported for Tensor, not for scalar.

| Categories | Subcategories |PyTorch | MindSpore | Difference                                                                                                                                                                                                         |
| --- |---------------| --- | --- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Parameters | Parameter 1   | tensors  | inputs | Same function                                                                                                                                                                                                      |
| | Parameter 2   | -        | indexing | torch.meshgrid v1.8.1 has no parameter `indexing`, but has the same function as mindspore.ops.meshgrid when `indexing` parameter is set to 'ij'. From v1.10, torch.meshgrid supports `indexing` parameter. |

### Code Example 1

```python
# PyTorch
import numpy as np
import torch

x = torch.tensor(np.array([1, 2, 3, 4]).astype(np.int32))
y = torch.tensor(np.array([5, 6, 7]).astype(np.int32))
z = torch.tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
output = torch.meshgrid(x, y, z)
print(output)
# (tensor([[[1, 1, 1, 1, 1],
#          [1, 1, 1, 1, 1],
#          [1, 1, 1, 1, 1]],
#         [[2, 2, 2, 2, 2],
#          [2, 2, 2, 2, 2],
#          [2, 2, 2, 2, 2]],
#         [[3, 3, 3, 3, 3],
#          [3, 3, 3, 3, 3],
#          [3, 3, 3, 3, 3]],
#         [[4, 4, 4, 4, 4],
#          [4, 4, 4, 4, 4],
#          [4, 4, 4, 4, 4]]], dtype=torch.int32), tensor([[[5, 5, 5, 5, 5],
#          [6, 6, 6, 6, 6],
#          [7, 7, 7, 7, 7]],
#         [[5, 5, 5, 5, 5],
#          [6, 6, 6, 6, 6],
#          [7, 7, 7, 7, 7]],
#         [[5, 5, 5, 5, 5],
#          [6, 6, 6, 6, 6],
#          [7, 7, 7, 7, 7]],
#         [[5, 5, 5, 5, 5],
#          [6, 6, 6, 6, 6],
#          [7, 7, 7, 7, 7]]], dtype=torch.int32), tensor([[[8, 9, 0, 1, 2],
#          [8, 9, 0, 1, 2],
#          [8, 9, 0, 1, 2]],
#         [[8, 9, 0, 1, 2],
#          [8, 9, 0, 1, 2],
#          [8, 9, 0, 1, 2]],
#         [[8, 9, 0, 1, 2],
#          [8, 9, 0, 1, 2],
#          [8, 9, 0, 1, 2]],
#         [[8, 9, 0, 1, 2],
#          [8, 9, 0, 1, 2],
#          [8, 9, 0, 1, 2]]], dtype=torch.int32))


# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
y = Tensor(np.array([5, 6, 7]).astype(np.int32))
z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
output = mindspore.ops.meshgrid(x, y, z, indexing='ij')
print(output)
# (Tensor(shape=[4, 3, 5], dtype=Int32, value=
#     [[[1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1]],
#      [[2, 2, 2, 2, 2],
#       [2, 2, 2, 2, 2],
#       [2, 2, 2, 2, 2]],
#      [[3, 3, 3, 3, 3],
#       [3, 3, 3, 3, 3],
#       [3, 3, 3, 3, 3]],
#      [[4, 4, 4, 4, 4],
#       [4, 4, 4, 4, 4],
#       [4, 4, 4, 4, 4]]]), Tensor(shape=[4, 3, 5], dtype=Int32, value=
#     [[[5, 5, 5, 5, 5],
#       [6, 6, 6, 6, 6],
#       [7, 7, 7, 7, 7]],
#      [[5, 5, 5, 5, 5],
#       [6, 6, 6, 6, 6],
#       [7, 7, 7, 7, 7]],
#      [[5, 5, 5, 5, 5],
#       [6, 6, 6, 6, 6],
#       [7, 7, 7, 7, 7]],
#      [[5, 5, 5, 5, 5],
#       [6, 6, 6, 6, 6],
#       [7, 7, 7, 7, 7]]]), Tensor(shape=[4, 3, 5], dtype=Int32, value=
#     [[[8, 9, 0, 1, 2],
#       [8, 9, 0, 1, 2],
#       [8, 9, 0, 1, 2]],
#      [[8, 9, 0, 1, 2],
#       [8, 9, 0, 1, 2],
#       [8, 9, 0, 1, 2]],
#      [[8, 9, 0, 1, 2],
#       [8, 9, 0, 1, 2],
#       [8, 9, 0, 1, 2]],
#      [[8, 9, 0, 1, 2],
#       [8, 9, 0, 1, 2],
#       [8, 9, 0, 1, 2]]]))
```
