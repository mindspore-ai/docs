# 比较与torch.Tensor.repeat的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/npTile.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## torch.Tensor.repeat

```python
torch.Tensor.repeat(*sizes)
```

更多内容详见[torch.Tensor.repeat](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.repeat)。

## mindspore.numpy.tile

```python
mindspore.numpy.tile(a, reps)
```

更多内容详见[mindspore.numpy.tile](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/numpy/mindspore.numpy.tile.html#mindspore.numpy.tile)。

## 使用方式

- MindSpore：把输入的张量`a`复制`reps`指定的次数来构造一个数组。假设`reps`长度为`d`，`a`的维度为`a.dim`，复制的规则是：

  如果`a.ndim` = `d`：把`a`沿着各轴复制对应的`reps`次。

  如果`a.ndim` < `d`：通过添加新轴将 `a.dim` 提升为`d`维，再进行复制；

  如果`a.ndim` > `d`：将通过在前面补1把`reps`提升为`a.ndim`，再进行复制。

- PyTorch：输入为复制的次数`size`，且限制`size`的长度需大于等于原始张量的维度，即不支持上述第三种情况。

## 代码示例

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

PyTorch：

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
