# Differences with torch.ger

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/normal.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

## torch.normal

```python
torch.normal(mean, std, *, generator=None, out=None)
torch.normal(mean=0.0, std, *, out=None)
torch.normal(mean, std=1.0, *, out=None)
torch.normal(mean, std, size, *, out=None)
```

For more information, see [torch.normal](https://pytorch.org/docs/1.8.1/generated/torch.normal.html).

## mindspore.ops.normal

```python
mindspore.ops.normal(shape, mean, stddev, seed=None)
```

For more information, see [mindspore.ops.normal](https://www.mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.normal.html).

## Differences

API function of MindSpore is consistent with that of PyTorch.

PyTorch: Four interface usages are supported.

- `mean` and `std` are both Tensor, requiring the same number of members for `mean` and `std`. The shape of the return value matches the shape of `mean` .
- `mean` is the float type, `std` is Tensor. The shape of the return value matches the shape of `std` .
- `std` is the float type, `mean` is Tensor. The shape of the return value matches the shape of `mean` .
- `mean` and `std` are both float types. The shape of the return value matches the shape of `size` .

MindSpore: The data types supported by `mean` and `std` are Tensor, and the shape of the return value is broadcast by `shape`, `mean`, and `stddev`.

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameter 1   | -            | shape         | This value in MindSpore is used to broadcast the shape of the return value together with `mean` and `stddev` |
|            | Parameter 2   | mean         | mean          | The data type supported in MindSpore is Tensor. Tensor and float are supported in PyTorch, corresponding to different usages |
|            | Parameter 3   | std          | stddev        | The data type supported in MindSpore is Tensor. Tensor and float are supported in PyTorch, corresponding to different usages |
|            | Parameter 4   | generator    | seed          | For details, see [General Difference Parameter Table](https://www.mindspore.cn/docs/en/r2.0/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table) |
|            | Parameter 5   | size         | -             | The shape of the return value in PyTorch, used under the specified interface usage |
|            | Parameter 6   | out          | -             | For details, see [General Difference Parameter Table](https://www.mindspore.cn/docs/en/r2.0/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table) |

## Code Example 1

> In PyTorch, 'mean' and 'std' are both Tensor.

```python
# PyTorch
import torch
import numpy as np

mean = torch.tensor(np.array([[3, 4], [5, 6]]), dtype=torch.float32)
stddev = torch.tensor(np.array([[0.2, 0.3], [0.4, 0.5]]), dtype=torch.float32)
output = torch.normal(mean, stddev)
print(output.shape)
# torch.Size([2, 2])

# MindSpore
import mindspore as ms
import numpy as np

shape = (2, 2)
mean = ms.Tensor(np.array([[3, 4], [5, 6]]), ms.float32)
stddev = ms.Tensor(np.array([[0.2, 0.3], [0.4, 0.5]]), ms.float32)
output = ms.ops.normal(shape, mean, stddev)
print(output.shape)
# (2, 2)
```

## Code Example 2

> In PyTorch, 'mean' is the float and 'std' is the Tensor.

```python
# PyTorch
import torch
import numpy as np

mean = 3.0
stddev = torch.tensor(np.array([[0.2, 0.3], [0.4, 0.5]]), dtype=torch.float32)
output = torch.normal(mean, stddev)
print(output.shape)
# torch.Size([2, 2])

# MindSpore
import mindspore as ms
import numpy as np

shape = (2, 2)
mean = ms.Tensor(3.0, ms.float32)
stddev = ms.Tensor(np.array([[0.2, 0.3], [0.4, 0.5]]), ms.float32)
output = ms.ops.normal(shape, mean, stddev)
print(output.shape)
# (2, 2)
```

## Code Example 3

> In PyTorch, 'mean' is Tensor, and 'std' is the float.

```python
# PyTorch
import torch
import numpy as np

mean = torch.tensor(np.array([[3, 4], [5, 6]]), dtype=torch.float32)
stddev = 1.0
output = torch.normal(mean, stddev)
print(output.shape)
# torch.Size([2, 2])

# MindSpore
import mindspore as ms
import numpy as np

shape = (2, 2)
mean = ms.Tensor(np.array([[3, 4], [5, 6]]), ms.float32)
stddev = ms.Tensor(1.0, ms.float32)
output = ms.ops.normal(shape, mean, stddev)
print(output.shape)
# (2, 2)
```

## Code Example 4

> In PyTorch, 'mean' and 'std' are both float.

```python
# PyTorch
import torch
import numpy as np

mean = 3.0
stddev = 1.0
size = (2, 2)
output = torch.normal(mean, stddev, size)
print(output.shape)
# torch.Size([2, 2])

# MindSpore
import mindspore as ms
import numpy as np

shape = (2, 2)
mean = ms.Tensor(3.0, ms.float32)
stddev = ms.Tensor(1.0, ms.float32)
output = ms.ops.normal(shape, mean, stddev)
print(output.shape)
# (2, 2)
```
