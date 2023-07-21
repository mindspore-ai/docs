# Function Differences with torch.Tensor.scatter_add_

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ScatterNdAdd.md)

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

For more information, see [mindspore.ops.ScatterNdAdd](https://mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.ScatterNdAdd.html#mindspore.ops.ScatterNdAdd).

## Differences

PyTorch: Given an input tensor, updates the tensor and index tensor; adds the updated tensor to the input tensor based on the index tensor along the given axis.

MindSpore: Given an input tensor, updates the tensor and index tensor; adds the updated tensor to the input tensor based on the index tensor.
Customizing the axes by parameters is not supported, but the axes can be specified by adjusting the shape of the index tensor.

## Code Example

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, no parameter for specifying dimension.
input_x = ms.Parameter(ms.Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), ms.float32), name="x")
indices = ms.Tensor(np.array([[2], [4], [1], [7]]), ms.int32)
updates = ms.Tensor(np.array([6, 7, 8, 9]), ms.float32)
scatter_nd_add = ops.ScatterNdAdd()
output = scatter_nd_add(input_x, indices, updates)
print(output)
# Out：
# [ 1. 10.  9.  4. 12.  6.  7. 17.]

# In torch, parameter dim can be set to specify dimension.
input_x = torch.tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.float32))
indices = torch.tensor(np.array([2, 4, 1, 7]).astype(np.int64))
updates = torch.tensor(np.array([6, 7, 8, 9]).astype(np.float32))
output = input_x.scatter_add_(dim=0, index=indices, src=updates)
print(output)
# Out:
# tensor([ 1., 10.,  9.,  4., 12.,  6.,  7., 17.])
```
