# Function Differences with torch.argmin

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/argmin.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

> For function differences between `mindspore.Tensor.argmin` and `torch.Tensor.argmin`, refer to the function differences between `mindspore.ops.argmin` and `torch.argmin`.

## torch.argmin

```text
torch.argmin(input, dim=None, keepdim=False) -> Tensor
```

For more information, see [torch.argmin](https://pytorch.org/docs/1.8.1/generated/torch.argmin.html).

## mindspore.ops.argmin

```text
mindspore.ops.argmin(x, axis=None, keepdims=False) -> Tensor
```

For more information, see [mindspore.ops.argmin](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.argmin.html).

## Differences

PyTorch: Return the index of the minimum value of the Tensor flattened or along the given dimension, and the return value is of type torch.int64. If there is more than one minimum value, the index of the first minimum value is returned.

MindSpore: The implementation function of API in MindSpore is basically the same as that of PyTorch, and the return value type is int32.

To ensure that the two output types are identical, use the [mindspore.ops.Cast](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Cast.html) operator to convert the result of MindSpore to mindspore.int64, which is done in each of the following examples.

| Categories | Subcategories   | PyTorch     | MindSpore   | Differences   |
| ---- | ----- | ------- | --------- | --------------------- |
| Input | Single input | input | x | Input Tensor |
| Parameters | Parameter 1 | dim | axis | Same function, different parameter names |
|  | Parameter 2 | keepdim | keepdims | Same function, different parameter names |

### Code Example 1

> For a zero-dimensional Tensor, PyTorch supports any combination of None/-1/0 for the dim parameter and True/False for the keepdim parameter, and the computation results are all consistent, all being a zero-dimensional Tensor. MindSpore version 1.8.1 does not support handling zero-dimensional Tensor at the moment, and you need to use [mindspore.ops.ExpandDims](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ExpandDims.html) to expand the Tensor to one dimension, and then follow the default parameter computation of the mindspore.ops.argmin operator.

```python
# PyTorch
import torch
import numpy as np

x = np.arange(1).reshape(()).astype(np.float32)
torch_argmin = torch.argmin
torch_output = torch_argmin(torch.tensor(x))
torch_out_np = torch_output.numpy()
print(torch_out_np)
# 0

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

x = np.arange(1).reshape(()).astype(np.float32)
ms_argmin = mindspore.ops.argmin
ms_expanddims = mindspore.ops.ExpandDims()
ms_cast = mindspore.ops.Cast()
ms_tensor = Tensor(x)

if not ms_tensor.shape:
    ms_tensor_tmp = ms_expanddims(ms_tensor, 0)
    ms_output = ms_argmin(ms_tensor_tmp)

ms_output = ms_cast(ms_output, mindspore.int64)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# 0
```

### Code Example 2

> When the dim parameter is not explicitly given, PyTorch argmin operator computes the result of doing an argmin operation on the original array flattened as a one-dimensional tensor, while MindSpore only supports computation on a single dimension. Therefore, to get the same result, pass the mindspore.ops.argmin operator into flatten Tensor before the calculation.

```python
# PyTorch
import torch
import numpy as np

x = np.arange(2*3*4).reshape(2, 3, 4).astype(np.float32)
torch_argmin = torch.argmin
torch_output = torch_argmin(torch.tensor(x))
torch_out_np = torch_output.numpy()
print(torch_out_np)
# 0

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

dim = None
x = np.arange(2*3*4).reshape(2,3,4).astype(np.float32)
ms_argmin = mindspore.ops.argmin
ms_expanddims = mindspore.ops.ExpandDims()
ms_cast = mindspore.ops.Cast()
ms_tensor = Tensor(x)

ms_output = ms_argmin(ms_tensor, axis=dim) if dim is not None else ms_argmin(
    ms_tensor.flatten())

ms_output = ms_cast(ms_output, mindspore.int64)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# 0
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
torch_argmin = torch.argmin
torch_output = torch_argmin(torch.tensor(x), dim=dim, keepdims=keepdims)
torch_out_np = torch_output.numpy()
print(torch_out_np)
# [[0]
#  [0]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

dim = 1
keepdims = True
x = np.arange(2*4).reshape(2, 4).astype(np.float32)
ms_argmin = mindspore.ops.argmin
ms_expanddims = mindspore.ops.ExpandDims()
ms_cast = mindspore.ops.Cast()
ms_tensor = Tensor(x)

ms_output = ms_argmin(ms_tensor, axis=dim, keepdims=keepdims)
ms_output = ms_cast(ms_output, mindspore.int64)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# [[0]
#  [0]]
```
