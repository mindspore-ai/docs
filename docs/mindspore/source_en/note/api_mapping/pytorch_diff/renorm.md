# Differences with torch.renorm

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_en/note/api_mapping/pytorch_diff/renorm.md)

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

For more information, see [mindspore.ops.renorm](https://mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.renorm.html).

## Differences

API function of MindSpore is consistent with that of PyTorch.

PyTorch: The data type of parameter `p` is ``float`` .

MindSpore: The data type of parameter `p` is ``int`` .

| Categories | Subcategories | PyTorch | MindSpore | Differences  |
| --- |---------------|---------| --- |-------------|
| Parameters | Parameter 1 |input | input | -  |
| | Parameter 2 | p | p | The data type supported by PyTorch is ``float`` , the data type supported by MindSpore is ``int`` . |
|  | Parameter 3 | dim        | axis | Different parameter names  |
| | Parameter 4 | maxnorm | maxnorm |  - |
| | Parameter 5 | out | - | For details, see [General Difference Parameter Table](https://www.mindspore.cn/docs/en/r2.2/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table) |

### Code Example

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
