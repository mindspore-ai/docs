# Function Differences with torch.unique

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.8/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Unique.md)

## torch.unique

```python
torch.unique(
    input,
    sorted=True,
    return_inverse=False,
    return_counts=False,
    dim=None
)
```

For more information, see [torch.unique](https://pytorch.org/docs/1.5.0/torch.html#torch.unique).

## mindspore.ops.Unique

```python
class mindspore.ops.Unique(*args, **kwargs)(x)
```

For more information, see [mindspore.ops.Unique](https://mindspore.cn/docs/en/r1.8/api_python/ops/mindspore.ops.Unique.html#mindspore.ops.Unique).

## Differences

PyTorch: By setting relevant parameters, determines whether to sort the output, to return indices of elements in the input corresponding to the output tensor, to return counts for each unique element.

MindSpore: Outputs all unique elements in ascending order, and returns indices of elements in the input corresponding to the output tensor.

## Code Example

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, the tensor containing unique elements in ascending order.
# As well as another tensor containing the corresponding indices will be directly returned.
x = ms.Tensor(np.array([1, 2, 5, 2]), ms.int32)
unique = ops.Unique()
output, indices = unique(x)
print(output)
print(indices)
# Out：
# [1 2 5]
# [0 1 2 1]

# In torch, parameters can be set to determine whether to output tensor containing unique elements in ascending order.
# As well as whether to output tensor containing corresponding indices.
x = torch.tensor([1, 2, 5, 2])
output, indices = torch.unique(x, sorted=True, return_inverse=True)
print(output)
print(indices)
# Out：
# tensor([1, 2, 5])
# tensor([0, 1, 2, 1])
```