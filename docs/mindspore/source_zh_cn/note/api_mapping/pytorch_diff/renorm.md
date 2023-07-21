# 比较与torch.renorm的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/renorm.md)

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

更多内容详见[mindspore.ops.renorm](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.renorm.html)。

## 差异对比

MindSpore此API功能与PyTorch一致。

PyTorch：参数 `p` 的数据类型是 ``float`` 。

MindSpore：参数 `p` 的数据类型是 ``int`` 。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 |input | input | -  |
| | 参数2 | p | p | PyTorch支持的数据类型是 ``float`` ，MindSpore支持的数据类型是 ``int`` |
|  | 参数3 | dim        | axis |  参数名不同 |
| | 参数4 | maxnorm | maxnorm |  - |
| | 参数5 | out | - | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/r2.1/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |

### 代码示例

```python
# PyTorch
import torch
x = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=torch.float32)
out = torch.renorm(x, 2.0, 0, 5.0)
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
