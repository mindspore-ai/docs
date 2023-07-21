# Function Differences with torch.Tensor.float

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Cast.md)

## torch.Tensor.float

```python
torch.Tensor.float(memory_format=torch.preserve_format)
```

For more information, see [torch.Tensor.float](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.float).

## mindspore.ops.Cast

```python
class mindspore.ops.Cast(*args, **kwargs)(
    input_x,
    type
)
```

For more information, see [mindspore.ops.Cast](https://mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.Cast.html#mindspore.ops.Cast).

## Differences

PyTorch: Changes the tensor type to float.

MindSpore：Converts the input type to the specified data type.

## Code Example

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, you can specify the data type to be transformed into.
input_x = ms.Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
cast = ops.Cast()
output = cast(input_x, ms.int32)
print(output.dtype)
# Int32
print(output.shape)
# (2, 3, 4, 5)

# In torch, the input will be transformed into float.
input_x = torch.Tensor(np.random.randn(2, 3, 4, 5).astype(np.int32))
output = input_x.float()
print(output.dtype)
# torch.float32
print(output.shape)
# torch.Size([2, 3, 4, 5])
```