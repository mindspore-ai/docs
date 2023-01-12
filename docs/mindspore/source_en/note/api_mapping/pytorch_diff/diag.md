# Function Differences with torch.diag

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/diag.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.diag

```text
torch.diag(input, diagonal=0, *, out=None) -> Tensor
```

For more information, see [torch.diag](https://pytorch.org/docs/1.8.1/generated/torch.diag.html).

## mindspore.ops.diag

```text
mindspore.ops.diag(input_x) -> Tensor
```

For more information, see [mindspore.ops.diag](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.diag.html).

## Differences

PyTorch: If the input is a one-dimensional tensor, the diagonal tensor is constructed by using the one-dimensional tensor composed of the diagonal values of the input. If the input is a matrix, the one-dimensional tensor composed of the diagonal elements of the input is returned.

MindSpore: If the input is a one-dimensional tensor, MindSpore API achieves the same function as PyTorch. If the input is a matrix, it does not achieve the same function as PyTorch, and there is no `diagonal` parameter to control the position of the diagonals to be considered.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | input | input_x |Same function, different parameter names |
| | Parameter 2 | diagonal | - | The value of `diagonal` in PyTorch is used to control the position of the diagonal to be considered. MindSpore does not have this parameter. |
| | Parameter 3 | out | - | Not involved |

### Code Example 1

> The PyTorch API parameter `x` supports both multidimensional and one-dimensional tensors, and there is a `diagonal` parameter to control the position of the diagonal to be considered, while the MindSpore API does not have a `diagonal` parameter. When the input parameter x is a one-dimensional tensor and `diagonal` is 0, the two APIs achieve the same function.

```python
# PyTorch
import torch
x = torch.tensor([1,2,3,4],dtype=int)
out = torch.diag(x)
out = out.detach().numpy()
print(out)
# [[1 0 0 0]
#  [0 2 0 0]
#  [0 0 3 0]
#  [0 0 0 4]]

# MindSpore
from mindspore import Tensor
import mindspore.ops as ops
input_x = Tensor([1, 2, 3, 4]).astype('int32')
output = ops.diag(input_x)
print(output)
# [[1 0 0 0]
#  [0 2 0 0]
#  [0 0 3 0]
#  [0 0 0 4]]

```

### Code Example 2

> When the input parameter `x` is a one-dimensional tensor and `diagonal` is not 0, this API of PyTorch controls the position of the diagonal to be considered, while this API of MindSpore does not have a `diagonal` parameter, and the output obtained from this API can be processed by mindspore.ops.pad to achieve the same function.

```python
# PyTorch
import torch
x = torch.tensor([1,2,3,4],dtype=int)
# Results for diagonal greater than 0
out = torch.diag(x, diagonal=1)
out = out.detach().numpy()
print(out)
# [[0 1 0 0 0]
#  [0 0 2 0 0]
#  [0 0 0 3 0]
#  [0 0 0 0 4]
#  [0 0 0 0 0]]

# Results for diagonal smaller than 0
out = torch.diag(x, diagonal=-1)
out = out.detach().numpy()
print(out)
# [[0 0 0 0 0]
#  [1 0 0 0 0]
#  [0 2 0 0 0]
#  [0 0 3 0 0]
#  [0 0 0 4 0]]

# MindSpore
from mindspore import Tensor
import mindspore.ops as ops
input_x = Tensor([1, 2, 3, 4]).astype('int32')
output = ops.diag(input_x)
# MindSpore implements this API function when the diagonal is greater than 0.
padding = ((0, 1), (1, 0))
a = ops.pad(output, padding)
print(a)
# [[0 1 0 0 0]
#  [0 0 2 0 0]
#  [0 0 0 3 0]
#  [0 0 0 0 4]
#  [0 0 0 0 0]]

# MindSpore implements this API function when the diagonal is smaller than 0.
padding = ((1, 0), (0, 1))
a = ops.pad(output, padding)
print(a)
# [[0 0 0 0 0]
#  [1 0 0 0 0]
#  [0 2 0 0 0]
#  [0 0 3 0 0]
#  [0 0 0 4 0]]
```

### Code Example 3

> This API of PyTorch is used to extract a one-dimensional tensor composed of diagonals when the input is a matrix and `diagonal` is used, MindSpore does not support this function. Using mindspore.numpy.diag can implement this function.

```python
# PyTorch
import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],dtype=int)
# Results for diagonal greater than 0
out = torch.diag(x, diagonal=1)
out = out.detach().numpy()
print(out)
# [2 6]

# Result when diagonal is the default value 0
out = torch.diag(x)
out = out.detach().numpy()
print(out)
# [1 5 9]

# Results for diagonal smaller than 0
out = torch.diag(x, diagonal=-1)
out = out.detach().numpy()
print(out)
# [4 8]

# MindSpore
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.numpy as np
input_x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype('int32')
# This function of mindspore.numpy.diag when diagonal is greater than 0
output = np.diag(input_x, k=1)
print(output)
# [2 6]

# This function of mindspore.numpy.diag when diagonal defaults to 0
output = np.diag(input_x)
print(output)
# [1 5 9]

# This function of mindspore.numpy.diag when diagonal is smaller than 0
output = np.diag(input_x, k=-1)
print(output)
# [4 8]
```
