# Function Differences with torch.Tensor.max

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/tensor_max.md)

## torch.Tensor.max

```python
torch.Tensor.max(dim=None, keepdim=False)
```

For more information, see [torch.Tensor.max](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.max).

## mindspore.Tensor.max

```python
mindspore.Tensor.max(axis=None, keepdims=False, *, initial=None, where=True, return_indices=False)
```

For more information, see [mindspore.Tensor.max](https://www.mindspore.cn/docs/en/r2.0/api_python/mindspore/Tensor/mindspore.Tensor.max.html).

## Differences

MindSpore is compatible with Numpy parameters `initial` and `where` based on PyTorch, added parameter return_ Indicators are used to control whether indexes are returned.

| Categories | Subcategories | PyTorch | MindSpore | Differences  |
| --- |---------------|---------| --- |-------------|
| Inputs  | Input 1 | dim     | axis      | Same function, different parameter names |
|     | Input 2 | keepdim | keepdims  | Same function, different parameter names |
|     | Input 3 | - | initial        | Not involved        |
|     | Input 4 |  - | where      | Not involved        |
|     | Input 5 |  -     |return_indices    | Not involved         |

### Code Example 1

When no dimension is specified, the two APIs implement the same functionality.

```python
import mindspore as ms
import torch
import numpy as np

np_x = np.array([[-0.0081, -0.3283, -0.7814, -0.0934],
                 [1.4201, -0.3566, -0.3848, -0.1608],
                 [-0.0446, -0.1843, -1.1348, 0.5722],
                 [-0.6668, -0.2368, 0.2790, 0.0453]]).astype(np.float32)
# mindspore
input_x = ms.Tensor(np_x)
output = input_x.max()
print(output)
# 1.4201

# torch
input_x = torch.tensor(np_x)
output = input_x.max()
print(output)
# tensor(1.4201)
```

### Code Example 2

When specifying dimensions, MindSpore does not return an index by default and needs to be manually specified.

```python
import mindspore as ms
import torch
import numpy as np

np_x = np.array([[-0.0081, -0.3283, -0.7814, -0.0934],
                 [1.4201, -0.3566, -0.3848, -0.1608],
                 [-0.0446, -0.1843, -1.1348, 0.5722],
                 [-0.6668, -0.2368, 0.2790, 0.0453]]).astype(np.float32)
# mindspore
input_x = ms.Tensor(np_x)
values, indices = input_x.max(axis=1, return_indices=True)
print(values)
# [-0.0081  1.4201  0.5722  0.279 ]
print(indices)
# [0 0 3 2]

# torch
input_x = torch.tensor(np_x)
values, indices = input_x.max(dim=1)
print(values)
# tensor([-0.0081,  1.4201,  0.5722,  0.2790])
print(indices)
# tensor([0, 0, 3, 2])
```