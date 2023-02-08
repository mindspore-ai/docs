# Function Differences with torch.scatter

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/scatter.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

> For the functional differences between `mindspore.Tensor.scatter` and `torch.Tensor.scatter` , refer to the functional differences between `mindspore.ops.scatter` and `torch.scatter` .

## torch.scatter

```python
torch.scatter(input, dim, index, src)
```

For more information, see [torch.scatter](https://pytorch.org/docs/1.8.1/generated/torch.scatter.html).

## mindspore.ops.scatter

```python
mindspore.ops.scatter(x, axis, index, src)
```

For more information, see [mindspore.ops.scatter](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.scatter.html).

## Differences

PyTorch: For all dimensions `d` , `index.size(d) <= src.size(d)` is required, i.e. `index` can select some or all of the data of `src` to be scattered into `input` .

MindSpore: The shape of `index` must be the same as the shape of `src` , i.e. all data of `src` will be scattered into `x` by `index` .

There is no difference in function.

## Code Example

```python
# PyTorch
import torch
import numpy as np
x = torch.tensor(np.zeros((5, 5)), dtype=torch.float32)
src = torch.tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=torch.float32)
index = torch.tensor(np.array([[0, 1], [0, 1], [0, 1]]), dtype=torch.int64)
out = torch.scatter(x=x, dim=1, index=index, src=src)
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
out = ms.ops.scatter(x=x, axis=1, index=index, src=src)
print(out)
# [[1. 2. 3. 0. 0.]
#  [4. 5. 6. 0. 0.]
#  [7. 8. 9. 0. 0.]
#  [0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]]
```
