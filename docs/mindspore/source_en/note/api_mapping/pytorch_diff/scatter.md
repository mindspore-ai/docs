# Differences with torch.scatter

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_en/note/api_mapping/pytorch_diff/scatter.md)

The following mapping relationships can be found in this file.

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|   torch.scatter    |   mindspore.ops.scatter    |
|    torch.Tensor.scatter   |  mindspore.Tensor.scatter   |

## torch.scatter

```python
torch.scatter(input, dim, index, src)
```

For more information, see [torch.scatter](https://pytorch.org/docs/1.8.1/generated/torch.scatter.html).

## mindspore.ops.scatter

```python
mindspore.ops.scatter(input, axis, index, src)
```

For more information, see [mindspore.ops.scatter](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.scatter.html).

## Differences

API function of MindSpore is not consistent with that of PyTorch.

PyTorch: For all dimensions `d` , `index.size(d) <= src.size(d)` is required, i.e. `index` can select some or all of the data of `src` to be scattered into `input` .

MindSpore: The shape of `index` must be the same as the shape of `src` , i.e. all data of `src` will be scattered into `input` by `index` .

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameter 1   | input        | input         | Consistent    |
|            | Parameter 2   | dim          | axis          | Different parameter names |
|            | Parameter 3   | index        | index         | For MindSpore, the shape of `index` must be the same as the shape of `src` . For PyTorch, `index.size(d) <= src.size(d)` is required for all dimensions `d` |
|            | Parameter 4   | src          | src           | Consistent    |

## Code Example 1

> Scatter operation on part of `src` data.

```python
# PyTorch
import torch
import numpy as np
input = torch.tensor(np.zeros((5, 5)), dtype=torch.float32)
src = torch.tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=torch.float32)
index = torch.tensor(np.array([[0, 1], [0, 1], [0, 1]]), dtype=torch.int64)
out = torch.scatter(input=input, dim=1, index=index, src=src)
print(out)
# tensor([[1., 2., 0., 0., 0.],
#         [4., 5., 0., 0., 0.],
#         [7., 8., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.]])

# MindSpore doesn't support this feature currently.
```

## Code Example 2

> Scatter operation on all of `src` data.
> The output values of singular value decomposition are not unique.

```python
import torch
import numpy as np
input = torch.tensor(np.zeros((5, 5)), dtype=torch.float32)
src = torch.tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=torch.float32)
index = torch.tensor(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), dtype=torch.int64)
out = torch.scatter(input=input, dim=1, index=index, src=src)
print(out)
# tensor([[1., 2., 3., 0., 0.],
#         [4., 5., 6., 0., 0.],
#         [7., 8., 9., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.]])

# MindSpore
import mindspore as ms
import numpy as np
input = ms.Tensor(np.zeros((5, 5)), dtype=ms.float32)
src = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
index = ms.Tensor(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), dtype=ms.int64)
out = ms.ops.scatter(input=input, axis=1, index=index, src=src)
print(out)
# [[1. 2. 3. 0. 0.]
#  [4. 5. 6. 0. 0.]
#  [7. 8. 9. 0. 0.]
#  [0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]]
```
