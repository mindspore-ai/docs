# 比较与torch.scatter的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/scatter.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

> `mindspore.Tensor.scatter` 和 `torch.Tensor.scatter` 的功能差异，参考 `mindspore.ops.scatter` 和 `torch.scatter` 的功能差异比较。

## torch.scatter

```python
torch.scatter(input, dim, index, src)
```

更多内容详见[torch.scatter](https://pytorch.org/docs/1.8.1/generated/torch.scatter.html)。

## mindspore.ops.scatter

```python
mindspore.ops.scatter(x, axis, index, src)
```

更多内容详见[mindspore.ops.scatter](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.scatter.html)。

## 差异对比

PyTorch：在任意维度 `d` 上，要求 `index.size(d) <= src.size(d)` ，即 `index` 可以选择 `src` 的部分或全部数据分散到 `input` 里。

MindSpore： `index` 的shape必须和 `src` 的shape一致，即 `src` 的所有数据都会被 `index` 分散到 `x` 里。

功能上无差异。

## 差异分析与示例

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
