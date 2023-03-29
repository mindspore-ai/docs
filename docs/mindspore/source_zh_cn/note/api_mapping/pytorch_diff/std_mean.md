# 比较与torch.std_mean的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/std_mean.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

## torch.std_mean

```python
torch.std_mean(input, dim, unbiased=True, keepdim=False, *, out=None)
```

更多内容详见[torch.std_mean](https://pytorch.org/docs/1.8.1/generated/torch.std_mean.html)。

## mindspore.ops.std_mean

```python
mindspore.ops.std_mean(input, axis=None, ddof=0, keepdims=False)
```

更多内容详见[mindspore.ops.std_mean](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.std_mean.html)。

## 差异对比

PyTorch：输出Tensor各维度上的标准差和均值，也可以按照 `dim` 对指定维度求标准差和均值。`unbiased` 如果为True，使用Bessel校正；如果是False，使用偏置估计来计算标准差。`keepdim` 控制输出和输入的维度是否相同。

MindSpore：输出Tensor各维度上的标准差和均值，也可以按照 `axis` 对指定维度求标准差和均值。如果 `ddof` 是布尔值，和 `unbiased` 作用相同； `ddof` 为整数，计算中使用的除数是 N−ddof，其中N表示元素的数量。`keepdim` 控制输出和输入的维度是否相同。

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
print(torch.std_mean(input, dim=2, unbiased=True, keepdim=True))
# (tensor([[[8.5829],
#          [4.2426]]]), tensor([[[ 2.5000],
#          [-3.0000]]]))

# MindSpore
import mindspore as ms

input = ms.Tensor([[[9, 7, 4, -10],
                    [-9, -2, 1, -2]]], ms.float32)
print(ms.ops.std_mean(input, axis=2, ddof=True, keepdims=True))
# (Tensor(shape=[1, 2, 1], dtype=Float32, value=
# [[[ 8.58292866e+00],
#   [ 4.24264050e+00]]]), Tensor(shape=[1, 2, 1], dtype=Float32, value=
# [[[ 2.50000000e+00],
#   [-3.00000000e+00]]]))
```
