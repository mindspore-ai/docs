# Function Differences with torch.equal

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/equal_2.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

> For function differences between `mindspore.Tensor.equal` and `torch.Tensor.equal`, refer to the function between `mindspore.ops.equal` and `torch.equal`.

## torch.equal

```text
torch.equal(input, other) -> bool
```

For more information, see [torch.eq](https://pytorch.org/docs/1.8.1/generated/torch.equal.html).

## mindspore.ops.equal

```text
mindspore.ops.equal(x, y) -> Tensor
```

For more information, see [mindspore.ops.equal](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.equal.html).

## Differences

PyTorch: True if two tensors have the same size and elements, False otherwise.

MindSpore: Compares two input Tensors element-wise to see if they are equal.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| :-: | :-: | :-: | :-: |:-:|
|Parameters | Parameter 1 | input | x | different parameter names |
| | Parameter 2 | other | y | different parameter names |

### Code Example

> torch.equal and mindspore.ops.equal implement the different function. torch.equal compares two input tensors to see if they have the same size and elements, returns a bool value; mindspore.ops.equal compares two input tensors element-wise to see if they are equal, returns a bool tensor whose shape is the same as the one after broadcasting.

```python
# PyTorch
import torch
from torch import tensor

input1 = tensor([1, 2], dtype=torch.float32)
other = tensor([[1, 2], [0, 2], [1, 3]], dtype=torch.int64)
out = torch.equal(input1, other)
print(out)
# False

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([1, 2]), mindspore.float32)
y = Tensor(np.array([[1, 2], [0, 2], [1, 3]]), mindspore.int64)
output = mindspore.ops.equal(x, y)
print(output)
# [[ True  True]
#  [False  True]
#  [ True False]]
```
