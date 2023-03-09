# Function Differences with torch.renorm

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/renorm.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The following mapping relationships can be found in this file.

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|    torch.renorm     |  mindspore.ops.renorm   |
|   torch.Tensor.renorm    |   mindspore.Tensor.renorm    |

## torch.renorm

```text
torch.renorm(input, p, dim, maxnorm, *, out=None) -> Tensor
```

For more information, see [torch.renorm](https://pytorch.org/docs/1.8.1/generated/torch.renorm.html).

## mindspore.ops.renorm

```text
mindspore.ops.renorm(input, p, axis, maxnorm)
```

For more information, see [mindspore.ops.renorm](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.renorm.html).

## Differences

PyTorch: Renormalize the sub-tensor of input `input` along dimension `dim` and the p norm of each sub-tensor does not exceed the given maximum norm maxnorm.

MindSpore: MindSpore API implements the same function as PyTorch, with only differences in parameter types.

| Categories | Subcategories | PyTorch | MindSpore | Differences  |
| --- |---------------|---------| --- |-------------|
| Parameters | Parameter 1 |input | input | -  |
| | Parameter 2 | p | p | The parameter is of type float in torch and is int in mindspore |
|  | Parameter 3 | dim        | axis | Same function, different parameter names  |
| | Parameter 4 | maxnorm | maxnorm |  - |
| | Parameter 5 | out | - | Not involved |

### Code Example 1

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
