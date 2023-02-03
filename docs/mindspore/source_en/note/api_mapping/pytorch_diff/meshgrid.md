# Function Differences with torch.meshgrid

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/meshgrid.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.meshgrid

```text
torch.meshgrid(
    *tensors,
    indexing=None)
```

For more information, see [torch.meshgrid](https://pytorch.org/docs/1.8.1/generated/torch.meshgrid.html).

## mindspore.ops.meshgrid

```text
mindspore.ops.meshgrid(*inputs, indexing='xy')
```

For more information, see [mindspore.ops.meshgrid](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.meshgrid.html).

## Differences

PyTorch: Generate the grid matrix from the given tensors. If it is a list of scalars, the scalar will automatically be considered as a tensor of size (1,).

MindSpore: MindSpore API basically implements the same function as TensorFlow. The inputs parameter is only supported for Tensor, not for scalar.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | tensors  | inputs | Same function |
| | Parameter 2 | indexing | indexing |The function is the same, and the default value is different. The default value of torch is 'ij', while the default value of mindspore is 'xy' |

### Code Example 1

```python
# PyTorch
import torch

x = torch.tensor(np.array([1, 2, 3, 4]).astype(np.int32))
y = torch.tensor(np.array([5, 6, 7]).astype(np.int32))
z = torch.tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
inputs = (x, y, z)
output = torch.meshgrid(inputs, indexing='xy')
print(output)
#(tensor([[[1, 1, 1, 1, 1],
#         [2, 2, 2, 2, 2],
#         [3, 3, 3, 3, 3],
#         [4, 4, 4, 4, 4]],

#        [[1, 1, 1, 1, 1],
#         [2, 2, 2, 2, 2],
#         [3, 3, 3, 3, 3],
#         [4, 4, 4, 4, 4]],

#        [[1, 1, 1, 1, 1],
#         [2, 2, 2, 2, 2],
#         [3, 3, 3, 3, 3],
#         [4, 4, 4, 4, 4]]], dtype=torch.int32), tensor([[[5, 5, 5, 5, 5],
#         [5, 5, 5, 5, 5],
#         [5, 5, 5, 5, 5],
#         [5, 5, 5, 5, 5]],

#        [[6, 6, 6, 6, 6],
#         [6, 6, 6, 6, 6],
#         [6, 6, 6, 6, 6],
#         [6, 6, 6, 6, 6]],

#        [[7, 7, 7, 7, 7],
#         [7, 7, 7, 7, 7],
#         [7, 7, 7, 7, 7],
#         [7, 7, 7, 7, 7]]], dtype=torch.int32), tensor([[[8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2]],

#        [[8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2]],

#        [[8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2]]], dtype=torch.int32))


# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
y = Tensor(np.array([5, 6, 7]).astype(np.int32))
z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
inputs = (x, y, z)
output = mindspore.ops.meshgrid(inputs, indexing='xy')
print(output)
#(Tensor(shape=[3, 4, 5], dtype=Int32, value=
#    [[[1, 1, 1, 1, 1],
#    [2, 2, 2, 2, 2],
#    [3, 3, 3, 3, 3],
#    [4, 4, 4, 4, 4]],
#    [[1, 1, 1, 1, 1],
#    [2, 2, 2, 2, 2],
#    [3, 3, 3, 3, 3],
#    [4, 4, 4, 4, 4]],
#    [[1, 1, 1, 1, 1],
#    [2, 2, 2, 2, 2],
#    [3, 3, 3, 3, 3],
#    [4, 4, 4, 4, 4]]]),
#    Tensor(shape=[3, 4, 5], dtype=Int32, value=
#    [[[5, 5, 5, 5, 5],
#    [5, 5, 5, 5, 5],
#    [5, 5, 5, 5, 5],
#    [5, 5, 5, 5, 5]],
#    [[6, 6, 6, 6, 6],
#    [6, 6, 6, 6, 6],
#    [6, 6, 6, 6, 6],
#    [6, 6, 6, 6, 6]],
#    [[7, 7, 7, 7, 7],
#    [7, 7, 7, 7, 7],
#    [7, 7, 7, 7, 7],
#    [7, 7, 7, 7, 7]]]),
#    Tensor(shape=[3, 4, 5], dtype=Int32, value=
#    [[[8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2]],
#    [[8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2]],
#    [[8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2]]]))
```
