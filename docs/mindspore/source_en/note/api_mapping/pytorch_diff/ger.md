# Differences with torch.ger

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ger.md)

The following mapping relationships can be found in this file.

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
| torch.ger             | mindspore.ops.ger         |
| torch.Tensor.ger      | mindspore.Tensor.ger      |

## torch.ger

```python
torch.ger(input, vec2, *, out=None)
```

For more information, see [torch.ger](https://pytorch.org/docs/1.8.1/generated/torch.ger.html).

## mindspore.ops.ger

```python
mindspore.ops.ger(input, other)
```

For more information, see [mindspore.ops.ger](https://www.mindspore.cn/docs/en/r2.3/api_python/ops/mindspore.ops.ger.html).

## Differences

API function of MindSpore is not consistent with that of PyTorch.

PyTorch: The parameters `input` and `vec2` support all data types of uint, int and float, and can be different data types. The data type of the return value selects a larger range of data types in the input parameter.

MindSpore: The data types of parameters `input` and `other` support float16/32/64, and must be the same data type. The data type of the return value is the same as the input.

There is no difference in function.

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameter 1   | input        | input         | PyTorch supports all data types of uint, int and float, and MindSpore only supports float16/32/64 |
|            | Parameter 2   | vec2         | other         | PyTorch supports all data types of uint, int and float, and MindSpore only supports float16/32/64 |
|            | Parameter 3   | out          | -             | For details, see [General Difference Parameter Table](https://www.mindspore.cn/docs/en/r2.3/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table) |

## Code Example 1

> The data type of the input is int, and the data type of the return value is also int.

```python
# PyTorch
import torch
import numpy as np

x1 = np.arange(3)
x2 = np.arange(6)

input = torch.tensor(x1, dtype=torch.int32)
other = torch.tensor(x2, dtype=torch.int32)
output = torch.ger(input, other)
print(output)
# tensor([[ 0,  0,  0,  0,  0,  0],
#         [ 0,  1,  2,  3,  4,  5],
#         [ 0,  2,  4,  6,  8, 10]], dtype=torch.int32)
print(output.dtype)
# torch.int32

# MindSpore doesn't support this feature currently.
```

## Code Example 2

> The data type of the input is float, and the data type of the return value is also float.

```python
# PyTorch
import torch
import numpy as np
x1 = np.arange(3)
x2 = np.arange(6)
input = torch.tensor(x1, dtype=torch.float32)
other = torch.tensor(x2, dtype=torch.float32)
output = torch.ger(input, other)
print(output)
# tensor([[ 0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  1.,  2.,  3.,  4.,  5.],
#         [ 0.,  2.,  4.,  6.,  8., 10.]])
print(output.dtype)
# torch.float32

# MindSpore
import mindspore as ms
import numpy as np
x1 = np.arange(3)
x2 = np.arange(6)
input = ms.Tensor(x1, ms.float32)
other = ms.Tensor(x2, ms.float32)
output = ms.ops.ger(input, other)
print(output)
# [[ 0.  0.  0.  0.  0.  0.]
#  [ 0.  1.  2.  3.  4.  5.]
#  [ 0.  2.  4.  6.  8. 10.]]
print(output.dtype)
# Float32
```
