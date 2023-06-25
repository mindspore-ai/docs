# Differences with torch.linalg.qr

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/qr.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.linalg.qr

```python
torch.linalg.qr(input, mode='reduced', *, out=None) -> (Tensor, Tensor)
```

For more information, see [torch.linalg.qr](https://pytorch.org/docs/1.8.1/linalg.html#torch.linalg.qr).

## mindspore.ops.qr

```python
mindspore.ops.qr(input, mode='reduced') -> (Tensor, Tensor)
```

For more information, see [mindspore.ops.qr](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.qr.html).

## Differences

PyTorch: Returns the QR decomposition of one or more matrices. If `mode` is 'reduced'(the default), compute the P columns of Q where P is minimum of the 2 innermost dimensions of input. If `mode` is 'complete', compute full-sized Q and R.

MindSpore: MindSpore API implements the same functions as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference                                    |
| ---- | ----- | ------- | --------- | -----------------------------------------------------------|
| Parameters | Parameter 1 | input | input | - |
|      | Parameter 2 | mode | mode | MindSpore can not support mode set to 'r' |
|      | Parameter 3 | out  | - | - |

## Code Example

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
