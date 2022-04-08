# Function Differences with torch.Tensor.t

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Transpose.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.Tensor.t

```python
torch.Tensor.t(input)
```

For more information, see [torch.Tensor.t](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.t).

## mindspore.ops.Transpose

```python
class mindspore.ops.Transpose(*args, **kwargs)(
    input_x,
    input_perm
)
```

For more information, see [mindspore.ops.Transpose](https://mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.Transpose.html#mindspore.ops.Transpose).

## Differences

PyTorch: Only applies to a 1D or 2D input.

MindSpore: No limit for dimension of the input, and how to transpose should be set by relevant parameters.

## Code Example

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, the input tensor will be transposed based on the dimension you set.
input_tensor = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
perm = (0, 2, 1)
transpose = ops.Transpose()
output = transpose(input_tensor, perm)
print(output.shape)
# Out：
# (2, 3, 2)

# In torch, only input of 2D dimension or lower will be accepted.
input1 = torch.randn(())
input2 = torch.randn((2, 3))
input3 = torch.randn((2, 3, 4))
for n, x in enumerate([input1, input2, input3]):
    try:
        output = torch.t(x)
        print(output.shape)
    except Exception as e:
        print('ERROR when inputting {}D: '.format(n + 1) + str(e))
# Out：
# torch.Size([])
# torch.Size([3, 2])
# ERROR when inputting 3D: t() expects a tensor with <=2 dimensions, but self is 3D.
```