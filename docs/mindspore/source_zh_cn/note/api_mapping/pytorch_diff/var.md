# 比较与torch.var的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/var.md)

以下映射关系均可参考本文。

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|   torch.var    |   mindspore.ops.var    |
|    torch.Tensor.var   |  mindspore.Tensor.var   |

## torch.var

```python
torch.var(input, dim, unbiased=True, keepdim=False, *, out=None)
```

更多内容详见[torch.var](https://pytorch.org/docs/1.8.1/generated/torch.var.html)。

## mindspore.ops.var

```python
mindspore.ops.var(input, axis=None, ddof=0, keepdims=False)
```

更多内容详见[mindspore.ops.var](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.var.html)。

## 差异对比

PyTorch：输出Tensor各维度上的方差，也可以按照 `dim` 对指定维度求方差。`unbiased` 如果为True，使用Bessel校正；如果是False，使用偏置估计来计算方差。`keepdim` 控制输出和输入的维度是否相同。

MindSpore：输出Tensor各维度上的方差，也可以按照 `axis` 对指定维度求方差。如果 `ddof` 是布尔值，和 `unbiased` 作用相同； `ddof` 为整数，计算中使用的除数是 N−ddof，其中N表示元素的数量。`keepdim` 控制输出和输入的维度是否相同。

| 分类       | 子类         | PyTorch      | MindSpore      | 差异          |
| ---------- | ------------ | ------------ | ---------      | ------------- |
| 参数       | 参数 1       | input         | input          | 功能一致，参数名不同 |
|            | 参数 2       | dim          | axis |  功能一致，参数名不同  |
|            | 参数 3       | unbiased          | ddof | `ddof` 为布尔值时，和 `unbiased` 功能一致 |
|            | 参数 4       | keepdim      | keepdims | 功能一致，参数名不同 |
|            | 参数 5       | out       | - |  MindSpore无此参数  |

### 代码示例

```python
# PyTorch
import torch

input = torch.tensor([[[9, 7, 4, -10],
                       [-9, -2, 1, -2]]], dtype=torch.float32)
print(torch.var(input, dim=2, unbiased=True, keepdim=True))
# tensor([[[73.6667],
#          [18.0000]]])

# MindSpore
import mindspore as ms

input = ms.Tensor([[[9, 7, 4, -10],
                    [-9, -2, 1, -2]]], ms.float32)
print(ms.ops.var(input, axis=2, ddof=True, keepdims=True))
# [[[73.666664]
#   [17.999998]]]
```
