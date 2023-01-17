# Function Differences with torch.transpose

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/mindspore.Tensor.transpose.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.transpose

```text
torch.transpose(input, dim0, dim1) -> Tensor
```

For more information, see [torch.transpose](https://pytorch.org/docs/1.8.1/generated/torch.transpose).

## mindspore.Tensor.transpose

```text
mindspore.Tensor.transpose(*axes) -> Tensor
```

For more information, see [mindspore.Tensor.transpose](https://www.mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.transpose.html).

## Differences

PyTorch: Transpose between the specified two dimensions of the input Tensor.

MindSpore: Not only can you transpose between two dimensions on MindSpore, but you can also transpose between multiple dimensions by modifying the parameter *axes.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
|Parameters | Parameter 1 | input |  |The Tensor interface itself is the object being manipulated, with the same function |
| | Parameter 2 | dim0 | - | In PyTorch, transpose between these two dimensions in conjunction with dim1. MindSpore does not have this parameter and can use axes to achieve the same function |
| | Parameter 3 | dim1 | - | In PyTorch, transpose between these two dimensions in conjunction with dim0. MindSpore does not have this parameter and can use axes to achieve the same function|
| | Parameter 4 | - | *axes | PyTorch does not have this parameter, but dim0 and dim1 in PyTorch can implement some of the functions of this parameter |

### Code Example 1

Description: When using torch.transpose(input, dim0, dim1), transpose between the two dimensions of input by setting dim0 and dim1. Although the two dimensions to be transposed cannot be specified directly in MindSpore, the axes parameter can be adjusted to achieve the same purpose. Suppose the input has a shape of (3, 2, 1, 4) and dim0, dim1 are 0, 2 respectively, it will be transposed between the first and third dimensions, and the shape after the operation is (1, 2, 3, 4). To implement this operation on MindSpore, only axes need to be set to (2, 1, 0, 3), i.e., the positions of 0 and 2 are switched based on the default dimensions (0, 1, 2, 3).
In general, for an arbitrary n-dimensional input and valid dim0 and dim1, setting axes simply involves swapping the position of the corresponding values of dim0, dim1 on the basis of (0, ... , n-1).

```python
#PyTorch
import torch
import numpy as np

input = torch.tensor(np.arange(2*3*4).reshape(1, 2, 3, 4))
dim0 = 0
dim1 = 2
output =  torch.transpose(input, dim0, dim2)
print(output.numpy())
#[[[[ 0  1  2  3]]
#  [[12 13 14 15]]]
# [[[ 4  5  6  7]]
#  [[16 17 18 19]]]
# [[[ 8  9 10 11]]
#  [[20 21 22 23]]]]

#MindSpore
import mindspore as ms
from mindspore import Tensor
import numpy as np

input_x = Tensor(np.arange(2*3*4).reshape(1, 2, 3, 4))
axes = (2, 1, 0, 3)
output = input_x.transpose(axes)
print(output.asnumpy())
#[[[[ 0  1  2  3]]
#  [[12 13 14 15]]]
# [[[ 4  5  6  7]]
#  [[16 17 18 19]]]
# [[[ 8  9 10 11]]
#  [[20 21 22 23]]]]
```
