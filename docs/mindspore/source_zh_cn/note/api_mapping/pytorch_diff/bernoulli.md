# 比较与torch.bernoulli的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/bernoulli.md)

以下映射关系均可参考本文。

|     PyTorch APIs          |      MindSpore APIs           |
| :-------------------:     | :-----------------------:     |
| torch.bernoulli           | mindspore.ops.bernoulli       |
| torch.Tensor.bernoulli    | mindspore.Tensor.bernoulli    |

## torch.bernoulli

```python
torch.bernoulli(input, *, generator=None, out=None)
```

更多内容详见[torch.bernoulli](https://pytorch.org/docs/1.8.1/generated/torch.bernoulli.html)。

## mindspore.ops.bernoulli

```python
mindspore.ops.bernoulli(input, p=0.5, seed=None)
```

更多内容详见[mindspore.ops.bernoulli](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.bernoulli.html)。

## 差异对比

MindSpore此API功能与PyTorch一致。

PyTorch: 参数 `input` 里保存了伯努利分布的概率值，返回值的shape和 `input` 一致。

MindSpore: 参数 `p` 里保存了伯努利分布的概率值，默认值为0.5。 `p` 的shape需要和 `input` 的shape一致，返回值的shape和 `input` 的shape一致。

| 分类       | 子类         | PyTorch      | MindSpore      | 差异          |
| ---------- | ------------ | ------------ | ---------      | ------------- |
| 参数       | 参数 1       | -             | input         | Mindspore下返回值的shape和数据类型和 `input` 的shape一致 |
|            | 参数 2       | input         | p             | 保存伯努利分布的概率值。PyTorch下返回值的shape和 `input` 一致。MindSpore下 `p` 为可选参数，默认值是0.5 |
|            | 参数 3       | generator     | seed          | MindSpore使用随机数种子生成随机数 |
|            | 参数 4       | out           | -             | 不涉及            |

## 代码示例

```python
# PyTorch
import torch
import numpy as np

p0 = np.array([0.0, 1.0, 1.0])
input_torch = torch.tensor(p0, dtype=torch.float32)
output = torch.bernoulli(input_torch)
print(output.shape)
# torch.Size([3])

# MindSpore
import mindspore as ms
import numpy as np

input0 = np.array([1, 2, 3])
p0 = np.array([0.0, 1.0, 1.0])

input = ms.Tensor(input0, ms.float32)
p = ms.Tensor(p0, ms.float32)
output = ms.ops.bernoulli(input, p)
print(output.shape)
# (3,)
```
