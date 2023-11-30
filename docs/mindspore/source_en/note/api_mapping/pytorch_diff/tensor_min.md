# Differences with torch.Tensor.min

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/tensor_min.md)

## torch.Tensor.min

```python
torch.Tensor.min(dim=None, keepdim=False)
```

For more information, see [torch.Tensor.min](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.min).

## mindspore.Tensor.min

```python
mindspore.Tensor.min(axis=None, keepdims=False, *, initial=None, where=True, return_indices=False)
```

For more information, see [mindspore.Tensor.min](https://www.mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.min.html).

## Differences

MindSpore is compatible with Numpy parameters `initial` and `where` based on PyTorch, added parameter return_indices is used to control whether indexes are returned.

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
output = input_x.min()
print(output)
# -1.1348

# torch
input_x = torch.tensor(np_x)
output = input_x.min()
print(output)
# tensor(-1.1348)
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
values, indices = input_x.min(axis=1, return_indices=True)
print(values)
# [-0.7814 -0.3848 -1.1348 -0.6668]
print(indices)
# [2 2 2 0]

# torch
input_x = torch.tensor(np_x)
values, indices = input_x.min(dim=1)
print(values)
# tensor([-0.7814, -0.3848, -1.1348, -0.6668])
print(indices)
# tensor([2, 2, 2, 0])
```