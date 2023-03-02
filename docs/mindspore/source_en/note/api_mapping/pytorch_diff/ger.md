# Function Differences with torch.ger

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ger.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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
mindspore.ops.ger(x1, x2)
```

For more information, see [mindspore.ops.ger](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ger.html).

## Differences

PyTorch: The parameters `input` and `vec2` support all data types of uint, int and float, and can be different data types. The data type of the return value selects a larger range of data types in the input parameter.

MindSpore: The data types of parameters `x1` and `x2` support float16/32/64, and must be the same data type. The data type of the return value is the same as the input.

There is no difference in function.

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameter 1   | input        | x1            | PyTorch supports all data types of uint, int and float, and MindSpore only supports float16/32/64 |
|            | Parameter 2   | vec2         | x2            | PyTorch supports all data types of uint, int and float, and MindSpore only supports float16/32/64 |
|            | Parameter 3   | out          | -             | Not involved  |

## Code Example

```python
# PyTorch
import torch
import numpy as np

x1 = np.arange(3)
x2 = np.arange(6)

input_x1 = torch.tensor(x1, dtype=torch.int32)
input_x2 = torch.tensor(x2, dtype=torch.float32)
output = torch.ger(input_x1, input_x2)
print(output)
print(output.dtype)
# tensor([[ 0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  1.,  2.,  3.,  4.,  5.],
#         [ 0.,  2.,  4.,  6.,  8., 10.]])
# torch.float32

# MindSpore
import mindspore as ms
import numpy as np

x1 = np.arange(3)
x2 = np.arange(6)

input_x1 = ms.Tensor(x1, ms.float32)
input_x2 = ms.Tensor(x2, ms.float32)
output = ms.ops.ger(input_x1, input_x2)
print(output)
print(output.dtype)
# [[ 0.  0.  0.  0.  0.  0.]
#  [ 0.  1.  2.  3.  4.  5.]
#  [ 0.  2.  4.  6.  8. 10.]]
# Float32
```
