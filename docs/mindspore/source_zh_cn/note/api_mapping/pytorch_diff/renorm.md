# 比较与torch.renorm的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/renorm.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

以下映射关系均可参考本文。

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|    torch.renorm     |  mindspore.ops.renorm   |
|   torch.Tensor.renorm    |   mindspore.Tensor.renorm    |

## torch.renorm

```text
torch.renorm(input, p, dim, maxnorm, *, out=None) -> Tensor
```

更多内容详见[torch.renorm](https://pytorch.org/docs/1.8.1/generated/torch.renorm.html)。

## mindspore.ops.renorm

```text
mindspore.ops.renorm(input, p, axis, maxnorm)
```

更多内容详见[mindspore.ops.renorm](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.renorm.html)。

## 差异对比

PyTorch：沿维度 `dim` 重新规范输入 `input` 的子张量，并且每个子张量的p范数不超过给定的最大范数 maxnorm 。

MindSpore：MindSpore此API实现功能与PyTorch一致，仅参数类型有差异。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 |input | input | -  |
| | 参数2 | p | p | torch上该参数为float类型，mindspore上为int类型 |
|  | 参数3 | dim        | axis |  功能一致，参数名不同 |
| | 参数4 | maxnorm | maxnorm |  - |
| | 参数5 | out | - | 不涉及 |

### 代码示例1

```python
# PyTorch
import torch
x = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=torch.float32)
out = torch.renorm(x, 2, 0, 5)
print(out.numpy())
# [[0.        1.        2.        3.       ]
#  [1.7817416 2.2271771 2.6726124 3.1180477]
#  [2.0908334 2.3521876 2.6135418 2.874896 ]]


# MindSpore
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=mindspore.float32)
out = ops.renorm(x, 2, 0, 5.0)
print(out.numpy())
# [[0.        1.        2.        3.       ]
#  [1.7817416 2.2271771 2.6726124 3.118048 ]
#  [2.0908334 2.3521876 2.6135418 2.874896 ]]
```
