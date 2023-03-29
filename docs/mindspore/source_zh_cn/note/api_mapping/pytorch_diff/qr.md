# 比较与torch.linalg.qr的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/qr.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

## torch.linalg.qr

```python
torch.linalg.qr(input, mode='reduced', *, out=None) -> (Tensor, Tensor)
```

更多内容详见[torch.linalg.qr](https://pytorch.org/docs/1.8.1/linalg.html#torch.linalg.qr)。

## mindspore.ops.qr

```python
mindspore.ops.qr(input, mode='reduced') -> (Tensor, Tensor)
```

更多内容详见[mindspore.ops.qr](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.qr.html)。

## 差异对比

PyTorch：返回一个或多个矩阵的QR（正交三角）分解。如果 `mode` 被设为'reduced'(默认值)，则计算Q的P列，其中P是 `input` 的2个最内层维度中的最小值。如果 `some` 被设为'complete'，则计算全尺寸Q和R。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                                         |
| ---- | ----- | ------- | --------- | -----------------------------------------------------------|
| 参数 | 参数1 | input | input | - |
|      | 参数2 | mode | mode | MindSpore不支持mode为'r' |
|      | 参数3 | out  | - | - |

## 代码示例

```python
# PyTorch

import torch
input = torch.tensor([[20, -31, 7], [4, 270, -90], [-8, 17, -32]], dtype=torch.float32)
q, r = torch.linalg.qr(input)
print(q)
# tensor([[-0.9129,  0.1637,  0.3740],
#         [-0.1826, -0.9831, -0.0154],
#         [ 0.3651, -0.0824,  0.9273]])
print(r)
# tensor([[ -21.9089,  -14.7885,   -1.6432],
#         [   0.0000, -271.9031,   92.2582],
#         [   0.0000,    0.0000,  -25.6655]])

# MindSpore
import mindspore as ms
from mindspore import ops
from mindspore import Tensor

input = Tensor([[20, -31, 7], [4, 270, -90], [-8, 17, -32]], ms.float32)
q, r = ops.qr(input)
print(q)
#[[-0.912871    0.16366126  0.37400758]
# [-0.18257418 -0.9830709  -0.01544376]
# [ 0.36514837 -0.08238228  0.92729706]]
print(r)
#[[ -21.908903  -14.788506  -1.6431675]
# [    0.       -271.9031    92.25824  ]
# [    0.          0.       -25.665514 ]]
```
