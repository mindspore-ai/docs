# Function Differences with torch.dot

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/tensor_dot.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

PyTorch: Calculates the dot product (inner product) of two tensors of the same shape, only 1D is supported. The supported input data types include uint8, int8/16/32/64, float32/64.

MindSpore: Calculates the dot product of two tensors on any axis. Support tensor of any dimension, but the shape corresponding to the specified axis should be equal. The function of the PyTorch is the same when the input is 1D and the axis is set to 0. The supported input data types are float16 or float32.

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameter 1   | input        | x1            | Same function, different parameter names |
|            | Parameter 2   | other        | x2            | Same function, different parameter names |
|            | Parameter 3   | out          | -             | Not involved |
|            | Parameter 4   | -            | axes          | The function of the PyTorch is the same when the input is 1D and the axis is set to 0. |

## Code Example

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, tensor of any dimension will be supported.
# And parameters will be set to specify how to compute among dimensions.
input_x1 = ms.Tensor(np.array([2, 3, 4]), ms.float32)
input_x2 = ms.Tensor(np.array([2, 1, 3]), ms.float32)
output = ops.tensor_dot(input_x1, input_x2, 1)
print(output)
# Out:
# 19.0

# In torch, only 1D tensor's computation will be supported.
input_x1 = torch.tensor([2, 3, 4])
input_x2 = torch.tensor([2, 1, 3])
output = torch.dot(input_x1, input_x2)
print(output)
# Out:
# tensor(19)
```
