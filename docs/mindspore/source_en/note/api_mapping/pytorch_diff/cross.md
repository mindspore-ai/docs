# Function Differences with torch.cross

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/cross.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.cross

```python
class torch.cross(
    input,
    other,
    dim=-1,
    out=None
)
```

For more information, see  [torch.cross](https://pytorch.org/docs/1.5.0/torch.html#torch.cross).

## mindspore.numpy.cross

```python
class mindspore.numpy.cross(
    a,
    b,
    axisa=- 1,
    axisb=- 1,
    axisc=- 1,
    axis=None
)
```

For more information, see  [mindspore.numpy.cross](https://mindspore.cn/docs/api/en/master/api_python/numpy/mindspore.numpy.cross.html#mindspore.numpy.cross).

## Differences

PyTorch: Returns the cross product of vectors in dimension dim of input and other. The inputs must have the same size, and the size of their dim dimension should be 3. If dim is not given, it defaults to the first dimension found with the size 3.

MindSpore: If a and b are arrays of vectors, the vectors are defined by the last axis of a and b by default, and these axes can have dimensions 2 or 3. Where the dimension of either a or b is 2, the third component of the input vector is assumed to be zero and the cross product calculated accordingly. In cases where both input vectors have dimension 2, the z-component of the cross product is returned.

## Code Example

```python
import mindspore.numpy as np
import torch

# MindSpore
x = np.array([[1,2,3], [4,5,6]])
y = np.array([[4,5,6], [1,2,3]])
output = np.cross(x, y)
print(output)
# [[-3  6 -3]
# [ 3 -6  3]]
output = np.cross(x, y, axisc=0)
print(output)
# [[-3  3]
# [ 6 -6]
# [-3  3]]
x = np.array([[1,2], [4,5]])
y = np.array([[4,5], [1,2]])
print(np.cross(x, y))
# Tensor(shape=[2], dtype=Int32, value= [-3,  3])

# PyTorch
a = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.int8)
b = torch.tensor([[4,5,6], [1,2,3]], dtype=torch.int8)
print(torch.cross(a, b, dim=1))
# tensor([[-3,  6, -3],
#         [ 3, -6,  3]], dtype=torch.int8)
print(torch.cross(a, b))
# tensor([[-3,  6, -3],
#         [ 3, -6,  3]], dtype=torch.int8)
a = torch.tensor([[1,2], [4,5]], dtype=torch.int8)
b = torch.tensor([[4,5], [1,2]], dtype=torch.int8)
print(torch.cross(a, b))
# RuntimeError: no dimension of size 3 in input
```
