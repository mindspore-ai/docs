# 比较与torch.scatter的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3.q1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3.q1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/scatter.md)

以下映射关系均可参考本文。

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|   torch.scatter    |   mindspore.ops.scatter    |
|    torch.Tensor.scatter   |  mindspore.Tensor.scatter   |

## torch.scatter

```python
torch.scatter(input, dim, index, src)
```

更多内容详见[torch.scatter](https://pytorch.org/docs/1.8.1/generated/torch.scatter.html)。

## mindspore.ops.scatter

```python
mindspore.ops.scatter(input, axis, index, src)
```

更多内容详见[mindspore.ops.scatter](https://www.mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.scatter.html)。

## 差异对比

MindSpore此API功能与PyTorch不一致。

PyTorch：在任意维度 `d` 上，要求 `index.size(d) <= src.size(d)` ，即 `index` 可以选择 `src` 的部分或全部数据分散到 `input` 里。

MindSpore： `index` 的shape必须和 `src` 的shape一致，即 `src` 的所有数据都会被 `index` 分散到 `input` 里。

功能上无差异。

| 分类       | 子类         | PyTorch      | MindSpore      | 差异          |
| ---------- | ------------ | ------------ | ---------      | ------------- |
| 参数       | 参数 1       | input         | input         | 一致           |
|            | 参数 2       | dim           | axis          | 参数名不一致 |
|            | 参数 3       | index         | index         | MindSpore的 `index` 的shape必须和 `src` 的shape一致，PyTorch要求在任意维度 `d` 上， `index.size(d) <= src.size(d)` |
|            | 参数 4       | src           | src           | 一致           |

## 代码示例 1

> 对 `src` 的部分数据进行scatter操作。

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

# MindSpore目前无法支持该功能。
```

## 代码示例 2

> 对 `src` 的全部数据进行scatter操作。

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
