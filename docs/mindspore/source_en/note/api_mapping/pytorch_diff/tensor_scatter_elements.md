# Differences with torch.Tensor.scatter_

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/tensor_scatter_elements.md)

## torch.Tensor.scatter_

```text
torch.Tensor.scatter_(dim, index, src, reduce) -> Tensor
```

For more information, see [torch.Tensor.scatter_](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.scatter_).

## mindspore.ops.tensor_scatter_elements

```text
mindspore.ops.tensor_scatter_elements(
    input_x,
    indices,
    updates,
    axis=0,
    reduction='none'
) -> Tensor
```

For more information, see [mindspore.ops.tensor_scatter_elements](https://www.mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.tensor_scatter_elements.html).

## Differences

PyTorch: Replaces the element at the specified index position in the Tensor with the given value.

MindSpore: MindSpore API implements the same function as PyTorch, which is a Tensor interface with a slightly different invocation method in PyTorch.

| Categories | Subcategories | PyTorch | MindSpore | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
|Parameters | Parameter 1 | dim | axis | Same function, different parameter names |
|    | Parameter 2 | index | indices | Same function, different parameter names |
|    | Parameter 3 | src | updates | Same function, different parameter names |
|    | Parameter 4 | reduce | reduction | Specification computation method. MindSpore only supports "none" and "add" modes. |
|    | Parameter 5 | - | input_x | This interface is the Tensor interface in PyTorch |

### Code Example

> The two APIs achieve the same function.

```python
# PyTorch
import torch

t = torch.zeros((3, 4), dtype=torch.float32)
indices = torch.tensor([[1, 2], [0, 1]])
values = torch.tensor([[3, 4], [5, 6]], dtype=torch.float32)
t.scatter_(0, indices, values)
print(t)
# tensor([[5., 0., 0., 0.],
#         [3., 6., 0., 0.],
#         [0., 4., 0., 0.]])

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor, Parameter
from mindspore import ops

input_x = Parameter(Tensor(np.zeros((3, 4)), mindspore.float32), name="x")
indices = Tensor(np.array([[1, 2], [0, 1]]), mindspore.int32)
updates = Tensor(np.array([[3, 4], [5, 6]]), mindspore.float32)
axis = 0
reduction = "none"
output = ops.tensor_scatter_elements(input_x, indices, updates, axis, reduction)
print(output)
# [[5. 0. 0. 0.]
#  [3. 6. 0. 0.]
#  [0. 4. 0. 0.]]
```
