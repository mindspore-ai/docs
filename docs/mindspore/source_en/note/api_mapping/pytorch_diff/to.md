# Differences with torch.ger

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/to.md)

## torch.Tensor.to

```python
torch.Tensor.to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
torch.Tensor.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
torch.Tensor.to(other, non_blocking=False, copy=False) -> Tensor
```

For more information, see [torch.Tensor.to](https://pytorch.org/docs/1.8.1/tensors.html?#torch.Tensor.to).

## mindspore.Tensor.to

```python
mindspore.Tensor.to(dtype)
```

For more information, see [mindspore.Tensor.to](https://mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.to.html).

## Differences

API function of MindSpore is not consistent with that of PyTorch.

PyTorch: Three interface usages are supported.

- When only the `dtype` parameter is supplied, the interface returns Tensor of the specified data type, and the usage is the same as MindSpore.
- When the `device` parameter is provided, the Tensor returned by the interface specifies the device, which MindSpore does not support.
- When `other` is supplied, the interface returns a Tensor of the same data type and device as `other`, which is not supported by MindSpore.

MindSpore：Only the `dtype` parameter is supported, which returns Tensor of the specified data type.

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameter 1   | dtype        | dtype         | Use the data types under the corresponding framework. |
|            | Parameter 2   | device       | -             | PyTorch specifies the device, which MindSpore does not support. |
|            | Parameter 3   | other        | -             | PyTorch specifies the Tensor used, which MindSpore does not support. |
|            | Parameter 4   | non_blocking | -          | PyTorch uses this for asynchronous copying between the CPU and GPU, which MindSpore does not support. |
|            | Parameter 5   | copy         | -             | PyTorch uses this to force the creation of new Tensors, which MindSpore does not support. |
|            | Parameter 6   | memory_format| -             | For details, see [General Difference Parameter Table](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table) |

## Code Example 1

> Specify only the `dtype` parameter.

```python
# PyTorch
import torch
import numpy as np
input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
input_x = torch.tensor(input_np)
dtype = torch.int32
output = input_x.to(dtype)
print(output.dtype)
# torch.int32

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np
input_x = Tensor(input_np)
dtype = mindspore.int32
output = input_x.to(dtype)
print(output.dtype)
# Int32
```

## Code Example 2

> Specify the `device` parameter.

```python
# PyTorch
import torch
import numpy as np
input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
input_x = torch.tensor(input_np)
device = torch.device('cpu')
output = input_x.to(device)
print(output.device)
# cpu

# MindSpore doesn't support this feature currently.
```

## Code Example 3

> Specify another Tensor.

```python
# PyTorch
import torch
import numpy as np
input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
input_x = torch.tensor(input_np)
input_y = torch.tensor(input_np).type(torch.int64)
output = input_x.to(input_y)
print(output.dtype)
# torch.int64

# MindSpore doesn't support this feature currently.
```
