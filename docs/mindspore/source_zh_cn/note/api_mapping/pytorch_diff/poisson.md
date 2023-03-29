# 比较与torch.poisson的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/poisson.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

## torch.poisson

```python
torch.poisson(input, generator=None)
```

更多内容详见[torch.poisson](https://pytorch.org/docs/1.8.1/generated/torch.poisson.html)。

## mindspore.ops.random_poisson

```python
mindspore.ops.random_poisson(shape, rate, seed=None, dtype=mstype.float32)
```

更多内容详见[mindspore.ops.random_poisson](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.random_poisson.html)。

## 差异对比

PyTorch: 返回值的shape和数据类型和 `input` 一致。

MindSpore: `shape` 决定了每个分布下采样的随机数张量的形状，返回值的shape是 `mindspore.concat([shape, mindspore.shape(rate)], axis=0)` 。当 `shape` 的值为 `Tensor([])` 时，返回值的shape和PyTorch一样，与 `rate` 的shape一致。返回值的数据类型由 `dtype` 决定。

功能上无差异。

| 分类       | 子类         | PyTorch      | MindSpore      | 差异          |
| ---------- | ------------ | ------------ | ---------      | ------------- |
| 参数       | 参数 1       | -             | shape         | MindSpore下每个分布下采样的随机数张量的形状，值为 `Tensor([])` 时返回值的shape和PyTorch一样 |
| 参数       | 参数 2       | input         | rate          | 泊松分布的参数 |
|            | 参数 3       | generator     | seed          | MindSpore使用随机数种子生成随机数 |
|            | 参数 4       | -             | dtype         | MindSpore下返回值的数据类型，支持int32/64，float16/32/64 |

## 差异分析与示例

```python
# PyTorch
import torch
import numpy as np

rate = torch.tensor(np.array([[5.0, 10.0], [5.0, 1.0]]), dtype=torch.float32)
output = torch.poisson(rate)
print(output.shape)
# torch.Size([2, 2])

# MindSpore
import mindspore as ms
import numpy as np

shape = ms.Tensor(np.array([]), ms.int32)
rate = ms.Tensor(np.array([[5.0, 10.0], [5.0, 1.0]]), dtype=ms.float32)
output = ms.ops.random_poisson(shape, rate, dtype=ms.float32)
print(output.shape)
# (2, 2)
```
