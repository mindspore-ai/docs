# Differences with torch.dot

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/tensor_dot.md)

## torch.dot

```python
torch.dot(input, other, *, out=None)
```

For more information, see [torch.dot](https://pytorch.org/docs/1.8.1/generated/torch.dot.html).

## mindspore.ops.tensor_dot

```python
mindspore.ops.tensor_dot(x1, x2, axes)
```

For more information, see [mindspore.ops.tensor_dot](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.tensor_dot.html#mindspore.ops.tensor_dot).

## Differences

API function of MindSpore is not consistent with that of PyTorch.

PyTorch: Calculates the dot product (inner product) of two tensors of the same shape, only 1D is supported. The supported input data types include uint8, int8/16/32/64, float32/64.

MindSpore: Calculates the dot product of two tensors on any axis. Support tensor of any dimension, but the shape corresponding to the specified axis should be equal. The function of the PyTorch is the same when the input is 1D and the axis is set to 0. The supported input data types are float16 or float32.

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameter 1   | input        | x1            | Different parameter names |
|            | Parameter 2   | other        | x2            | Different parameter names |
|            | Parameter 3   | out          | -             | For details, see [General Difference Parameter Table](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table) |
|            | Parameter 4   | -            | axes          | The function of the PyTorch is the same when the input is 1D and the axis is set to 0. |

## Code Example 1

> The data type of the input is int, and the data type of the output is also int.

```python
import torch
input_x1 = torch.tensor([2, 3, 4], dtype=torch.int32)
input_x2 = torch.tensor([2, 1, 3], dtype=torch.int32)
output = torch.dot(input_x1, input_x2)
print(output)
# tensor(19, dtype=torch.int32)
print(output.dtype)
# torch.int32

# MindSpore doesn't support this feature currently.
```

## Code Example 2

> The data type of the input is float, and the data type of the output is also float.

```python
import torch
input_x1 = torch.tensor([2, 3, 4], dtype=torch.float32)
input_x2 = torch.tensor([2, 1, 3], dtype=torch.float32)
output = torch.dot(input_x1, input_x2)
print(output)
# tensor(19.)
print(output.dtype)
# torch.float32

import mindspore as ms
import mindspore.ops as ops
import numpy as np
input_x1 = ms.Tensor(np.array([2, 3, 4]), ms.float32)
input_x2 = ms.Tensor(np.array([2, 1, 3]), ms.float32)
output = ops.tensor_dot(input_x1, input_x2, 1)
print(output)
# 19.0
print(output.dtype)
# Float32
```
