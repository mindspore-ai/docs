# Function Differences with torch.scatter_add

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/scatter_add.md)

## torch.scatter_add

```python
torch.scatter_add(input, dim, index, src)
```

For more information, see [torch.scatter_add](https://pytorch.org/docs/1.8.1/generated/torch.scatter_add.html).

## mindspore.ops.tensor_scatter_elements

```python
mindspore.ops.tensor_scatter_elements(input_x, indices, updates, axis, reduction)
```

For more information, see [mindspore.ops.tensor_scatter_elements](https://www.mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.tensor_scatter_elements.html).

## Differences

PyTorch: For all dimensions `d` , `index.size(d) <= src.size(d)` is required, i.e. `index` can select some or all of the data of `src` to be scattered into `input` .

MindSpore: The shape of `indices` must be the same as the shape of `updates` , i.e. all data of `updates` will be scattered into `input_x` by `indices` .

There is no difference in function.

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameter 1   | input        | input_x       | Same function, different parameter names |
|            | Parameter 2   | dim          | axis          | Same function, different parameter names |
|            | Parameter 3   | index        | indices       | For MindSpore, the shape of `indices` must be the same as the shape of `updates` . For PyTorch, `index.size(d) <= src.size(d)` is required for all dimensions `d` |
|            | Parameter 4   | src          | updates       | Same function  |
|            | Parameter 5   |              | reduction     | `reduction` must be set as "add" |

### Code Example

```python
# PyTorch
import torch
import numpy as np
x = torch.tensor(np.zeros((5, 5)), dtype=torch.float32)
src = torch.tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=torch.float32)
index = torch.tensor(np.array([[0, 1], [0, 1], [0, 1]]), dtype=torch.int64)
out = torch.scatter_add(x=x, dim=1, index=index, src=src)
print(out)
# tensor([[1., 2., 0., 0., 0.],
#         [4., 5., 0., 0., 0.],
#         [7., 8., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.]])

# MindSpore
import mindspore as ms
import numpy as np
x = ms.Tensor(np.zeros((5, 5)), dtype=ms.float32)
src = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
index = ms.Tensor(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), dtype=ms.int64)
out = ms.ops.tensor_scatter_elements(input_x=x, axis=1, indices=index, updates=src, reduction="add")
print(out)
# [[1. 2. 3. 0. 0.]
#  [4. 5. 6. 0. 0.]
#  [7. 8. 9. 0. 0.]
#  [0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]]
```
