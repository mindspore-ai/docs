# Function Differences with torch.Tensor.repeat

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/npTile.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## torch.Tensor.repeat

```python
torch.Tensor.repeat(*sizes)
```

For more information, see [torch.Tensor.repeat](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.repeat).

## mindspore.numpy.tile

```python
mindspore.numpy.tile(a, reps)
```

For more information, see [mindspore.numpy.tile](https://www.mindspore.cn/docs/api/en/r1.5/api_python/numpy/mindspore.numpy.tile.html#mindspore.numpy.tile).

## Differences

- MindSpore: Constructs an array by repeating `a` the number of times given by `reps`. If `reps` has length `d`, `a` has dimensions `a.dim`, the rules for repeat operation is:

  If `a.ndim` = `d`: copy `a` for `reps` times in the corresponding axis ;

  If `a.ndim` < `d`:  `a` is promoted to be d-dimensional by prepending new axis, and then copied;

  If `a.ndim` > `d`:  The `reps` will be promoted to `a.ndim` by adding 1 in the front, and then copied.

- PyTorch: The length of input args `size` must be greater than or equal to the dimension of the self tensor, that is, the above third case is not supported.

## Code Example

MindSpore:

```python
import mindspore.numpy as np

a = np.array([[0, 2, 1], [3, 4, 5]])

b = np.tile(a, 2)
print(b)

# out:
# [[0 2 1 0 2 1]
#  [3 4 5 3 4 5]]

c = np.tile(a, (2, 1))
print(c)

# out:
# [[0 2 1]
#  [3 4 5]
# [0 2 1]
#  [3 4 5]]

d = np.tile(a, (2, 1, 2))
print(d)

# out
# [[[0 2 1 0 2 1]
#   [3 4 5 3 4 5]]

#  [[0 2 1 0 2 1]
#   [3 4 5 3 4 5]]]
```

PyTorch:

```python
import torch

a = torch.tensor([[0, 2, 1], [3, 4, 5]])

b = a.repeat(2)

# error:
# RuntimeError: Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor

c = a.repeat(2, 1)
print(c)

# out:
#tensor([[0, 2, 1],
#         [3, 4, 5],
#         [0, 2, 1],
#         [3, 4, 5]])

d = a.repeat(2, 1, 2)
print(d)

# out:
#tensor([[[0, 2, 1, 0, 2, 1],
#          [3, 4, 5, 3, 4, 5]],
#
#         [[0, 2, 1, 0, 2, 1],
#          [3, 4, 5, 3, 4, 5]]])
```
