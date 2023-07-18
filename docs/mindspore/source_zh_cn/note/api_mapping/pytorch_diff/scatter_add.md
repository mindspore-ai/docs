# 比较与torch.scatter_add的差异

<a href="https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/scatter_add.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.png"></a>

## torch.scatter_add

```python
torch.scatter_add(input, dim, index, src)
```

更多内容详见[torch.scatter_add](https://pytorch.org/docs/1.8.1/generated/torch.scatter_add.html)。

## mindspore.ops.tensor_scatter_elements

```python
mindspore.ops.tensor_scatter_elements(input_x, indices, updates, axis, reduction)
```

更多内容详见[mindspore.ops.tensor_scatter_elements](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.tensor_scatter_elements.html)。

## 差异对比

PyTorch：在任意维度 `d` 上，要求 `index.size(d) <= src.size(d)` ，即 `index` 可以选择 `src` 的部分或全部数据分散到 `input` 里。

MindSpore： `indices` 的shape必须和 `updates` 的shape一致，即 `updates` 的所有数据都会被 `indices` 分散到 `input_x` 里。

功能上无差异。

| 分类       | 子类         | PyTorch      | MindSpore      | 差异          |
| ---------- | ------------ | ------------ | ---------      | ------------- |
| 参数       | 参数 1       | input         | input_x        | 功能一致，参数名不同 |
|            | 参数 2       | dim           | axis          | 功能一致，参数名不同 |
|            | 参数 3       | index         | indices       | MindSpore的 `indices` 的shape必须和 `updates` 的shape一致，PyTorch要求在任意维度 `d` 上， `index.size(d) <= src.size(d)` |
|            | 参数 4       | src           | updates       | 功能一致      |
|            | 参数 5       |               | reduction     | MindSpore的 `reduction` 必须设置为 "add"|

### 代码示例

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
