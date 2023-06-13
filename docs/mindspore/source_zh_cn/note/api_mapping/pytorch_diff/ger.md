# 比较与torch.ger的差异

<a href="https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/ger.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source.png"></a>

以下映射关系均可参考本文。

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
| torch.ger             | mindspore.ops.ger         |
| torch.Tensor.ger      | mindspore.Tensor.ger      |

## torch.ger

```python
torch.ger(input, vec2, *, out=None)
```

更多内容详见[torch.ger](https://pytorch.org/docs/1.8.1/generated/torch.ger.html)。

## mindspore.ops.ger

```python
mindspore.ops.ger(input, other)
```

更多内容详见[mindspore.ops.ger](https://www.mindspore.cn/docs/zh-CN/r1.11/api_python/ops/mindspore.ops.ger.html)。

## 差异对比

MindSpore此API功能与PyTorch不一致。

PyTorch: 参数 `input` 和 `vec2` 支持uint, int和float下的所有数据类型，且可以是不同的数据类型，返回值的数据类型选择输入参数中范围更大的数据类型。

MindSpore: 参数 `input` 和 `other` 的数据类型支持float16/32/64，必须是相同的数据类型，返回值的数据类型和输入一致。

| 分类       | 子类         | PyTorch      | MindSpore      | 差异          |
| ---------- | ------------ | ------------ | ---------      | ------------- |
| 参数       | 参数 1       | input         | input          | PyTorch支持uint、int和float下的所有数据类型，MindSpore仅支持float16/32/64。 |
|            | 参数 2       | vec2          | other         | PyTorch支持uint、int和float下的所有数据类型，MindSpore仅支持float16/32/64。 |
|            | 参数 3       | out           | -             | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/r1.11/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |

## 代码示例 1

> 输入的数据类型是int，返回值的数据类型也是int。

```python
# PyTorch
import torch
import numpy as np

x1 = np.arange(3)
x2 = np.arange(6)

input = torch.tensor(x1, dtype=torch.int32)
other = torch.tensor(x2, dtype=torch.int32)
output = torch.ger(input, other)
print(output)
print(output.dtype)
# tensor([[ 0,  0,  0,  0,  0,  0],
#         [ 0,  1,  2,  3,  4,  5],
#         [ 0,  2,  4,  6,  8, 10]], dtype=torch.int32)
# torch.int32
# MindSpore目前无法支持该功能
```

## 代码示例 2

> 输入的数据类型是float，返回值的数据类型也是float。

```python
# PyTorch
import torch
import numpy as np
x1 = np.arange(3)
x2 = np.arange(6)
input = torch.tensor(x1, dtype=torch.float32)
other = torch.tensor(x2, dtype=torch.float32)
output = torch.ger(input, other)
print(output)
print(output.dtype)
# tensor([[ 0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  1.,  2.,  3.,  4.,  5.],
#         [ 0.,  2.,  4.,  6.,  8., 10.]])
# torch.float32

# MindSpore
import mindspore as ms
import numpy as np

x1 = np.arange(3)
x2 = np.arange(6)

input = ms.Tensor(x1, ms.float32)
other = ms.Tensor(x2, ms.float32)
output = ms.ops.ger(input, other)
print(output)
print(output.dtype)
# [[ 0.  0.  0.  0.  0.  0.]
#  [ 0.  1.  2.  3.  4.  5.]
#  [ 0.  2.  4.  6.  8. 10.]]
# Float32
```
