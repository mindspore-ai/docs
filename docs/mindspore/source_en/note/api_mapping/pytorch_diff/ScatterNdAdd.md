# Function Differences with torch.Tensor.scatter_add_

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ScatterNdAdd.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.Tensor.scatter_add_

```python
torch.Tensor.scatter_add_(
    dim,
    index,
    src
)
```

For more information, see [torch.Tensor.scatter_add_](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.scatter_add_).

## mindspore.ops.ScatterNdAdd

```python
class mindspore.ops.ScatterNdAdd(use_locking=False)(
    input_x,
    indices,
    update
)
```

For more information, see [mindspore.ops.ScatterNdAdd](https://mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.ScatterNdAdd.html#mindspore.ops.ScatterNdAdd).

## Differences

PyTorch: Given an input tensor, updates the tensor and index tensor; adds the updated tensor to the input tensor based on the index tensor along the given axis.

MindSpore: Given an input tensor, updates the tensor and index tensor; adds the updated tensor to the input tensor based on the index tensor. Setting axis is not supported.

## Code Example

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, no parameter for specifying dimension.
input_x = mindspore.Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
scatter_nd_add = ops.ScatterNdAdd()
output = scatter_nd_add(input_x, indices, updates)
print(output)
# Out：
# [1. 10. 9. 4. 12. 6. 7. 17.]

# In torch, parameter dim can be set to specify dimension.
input_x = torch.tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.float32))
indices = torch.tensor(np.array([2, 4, 1, 7]).astype(np.int64))
updates = torch.tensor(np.array([6, 7, 8, 9]).astype(np.float32))
output = input_x.scatter_add_(dim=0, index=indices, src=updates)
print(output)
# Out:
# tensor([1., 10., 9., 4., 12., 6., 7., 17.])
```
