# Function Differences with torch.argmax

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/argmax.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

> For function differences between `mindspore.Tensor.argmax` and `torch.Tensor.argmax`, refer to the function differences between `mindspore.ops.argmax` and `torch.argmax`.

## torch.argmax

```text
torch.argmax(input, dim, keepdim=False) -> Tensor
```

For more information, see [torch.argmax](https://pytorch.org/docs/1.8.1/generated/torch.argmax.html).

## mindspore.ops.argmax

```text
mindspore.ops.argmax(x, axis=None, keepdims=False) -> Tensor
```

For more information, see [mindspore.ops.argmax](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.argmax.html).

## Differences

PyTorch: Return the subscript along the given dimension where the maximum value of the Tensor is located, and the return value is of type torch.int64.

MindSpore: The implementation function of API in MindSpore is basically the same as that of PyTorch。

| Categories | Subcategories   | PyTorch     | MindSpore   | Differences   |
| ---- | ----- | ------- | --------- | --------------------- |
| Input | Single input | input | x | Input Tensor |
| Parameters | Parameter 1 | dim | axis | Same function, different parameter names |
|  | Parameter 2 | keepdim | keepdims | Same function, different parameter names |

### Code Example 1

> For a zero-dimensional Tensor, PyTorch supports any combination of None/-1/0 for the dim parameter and True/False for the keepdim parameter, and the computation results are all consistent, all being a zero-dimensional Tensor. MindSpore version 1.8.1 does not support handling zero-dimensional Tensor at the moment, and you need to use [mindspore.ops.ExpandDims](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ExpandDims.html) to expand the Tensor to one dimension, and then follow the default parameter computation of the mindspore.ops.argmax operator.

```python
# PyTorch
import torch
import numpy as np

x = np.arange(1).reshape(()).astype(np.float32)
torch_argmax = torch.argmax
torch_output = torch_argmax(torch.tensor(x))
torch_out_np = torch_output.numpy()
print(torch_out_np)
# 0

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

x = np.arange(1).reshape(()).astype(np.float32)
ms_argmax = mindspore.ops.argmax
ms_expanddims = mindspore.ops.ExpandDims()
ms_tensor = Tensor(x)

if not ms_tensor.shape:
    ms_tensor_tmp = ms_expanddims(ms_tensor, 0)
    ms_output = ms_argmax(ms_tensor_tmp)

ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# 0
```

### Code Example 2

> When the dim parameter is not explicitly given, PyTorch argmax operator computes the result of doing an argmax operation on the original array flattened as a one-dimensional tensor, while MindSpore only supports computation on a single dimension. Therefore, to get the same result, pass the mindspore.ops.argmax operator into flatten Tensor before the calculation.

```python
# PyTorch
import torch
import numpy as np

x = np.arange(2*3*4).reshape(2, 3, 4).astype(np.float32)
torch_argmax = torch.argmax
torch_output = torch_argmax(torch.tensor(x))
torch_out_np = torch_output.numpy()
print(torch_out_np)
# 23

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

dim = None
x = np.arange(2*3*4).reshape(2,3,4).astype(np.float32)
ms_argmax = mindspore.ops.argmax
ms_expanddims = mindspore.ops.ExpandDims()
ms_tensor = Tensor(x)

ms_output = ms_argmax(ms_tensor, axis=dim) if dim is not None else ms_argmax(
    ms_tensor.flatten())

ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# 23
```

### Code Example 3

> The PyTorch operator has a keepdim parameter. When set to True, it serves to keep the dimension for which aggregation is performed and is set to 1. MindSpore keepdims parameter is consistent with its function. To achieve the same result, after the calculation is done, use the [mindspore.ops.ExpandDims](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ExpandDims.html) operator to expand the dimensionality.

```python
# PyTorch
import torch
import numpy as np

dim = 1
keepdims = True
x = np.arange(2*4).reshape(2, 4).astype(np.float32)
torch_argmax = torch.argmax
torch_output = torch_argmax(torch.tensor(x), dim=dim, keepdims=keepdims)
torch_out_np = torch_output.numpy()
print(torch_out_np)
# [[3]
#  [3]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

dim = 1
keepdims = True
x = np.arange(2*4).reshape(2, 4).astype(np.float32)
ms_argmax = mindspore.ops.argmax
ms_expanddims = mindspore.ops.ExpandDims()
ms_tensor = Tensor(x)

ms_output = ms_argmax(ms_tensor, axis=dim, keepdims=keepdims)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# [[3]
#  [3]]
```
