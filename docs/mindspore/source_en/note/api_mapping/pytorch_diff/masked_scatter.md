# Function Differences with torch.Tensor.masked_scatter

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/masked_scatter.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.Tensor.masked_scatter

```python
torch.Tensor.masked_scatter(mask, tensor) -> Tensor
```

For more information, see [torch.Tensor.masked_scatter](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.Tensor.masked_scatter).

## mindspore.Tensor.masked_scatter

```python
mindspore.Tensor.masked_scatter(mask, tensor) -> Tensor
```

For more information, see [mindspore.Tensor.masked_scatter](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.Tensor.masked_scatter.html).

## Differences

PyTorch: Returns a Tensor. Updates the value in the "self Tensor" with the `tensor` value according to the mask.

MindSpore: MindSpore API Basically achieves the same function as PyTorch. But PyTorch supports bidirectional broadcast of `mask` and "self Tensor", MindSpore only supports `mask` broadcasting to "self Tensor".

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ----|
| Parameters | Parameter 1 | mask | mask | PyTorch supports bidirectional broadcast of `mask` and "self Tensor", MindSpore only supports `mask` broadcasting to "self Tensor". |
|      | Parameter 2 | tensor | tensor | - |

### Code Example 1

```python
# PyTorch
import torch

self = torch.tensor([0, 0, 0, 0, 0])
mask = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]])
source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
output = self.masked_scatter(mask, source)
print(output)
# tensor([[0, 0, 0, 0, 1],
#         [2, 3, 0, 4, 5]])

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

self = Tensor(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]), mindspore.int32)
mask = Tensor(np.array([[False, False, False, True, True], [True, True, False, True, True]]), mindspore.bool_)
source = Tensor(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), mindspore.int32)
output = self.masked_scatter(mask, source)
print(output)
# [[0, 0, 0, 0, 1],
#  [2, 3, 0, 4, 5]]
```

### Code Example 2

```python
import torch

self = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
mask = torch.tensor([0, 0, 0, 1, 1])
source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
output = self.masked_scatter(mask, source)
print(output)
# tensor([[0, 0, 0, 0, 1],
#         [0, 0, 0, 2, 3]])

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

self = Tensor(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]), mindspore.int32)
mask = Tensor(np.array([False, False, False, True, True]), mindspore.bool_)
source = Tensor(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), mindspore.int32)
output = self.masked_scatter(mask, source)
print(output)
# [[0, 0, 0, 0, 1],
#  [0, 0, 0, 2, 3]]
```
