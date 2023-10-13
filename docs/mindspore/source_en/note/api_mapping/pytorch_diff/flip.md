# Differences with torch.Tensor.flip

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_en/note/api_mapping/pytorch_diff/flip.md)

## torch.Tensor.flip

```python
torch.Tensor.flip(dims)
```

For more details, see [torch.Tensor.flip](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.flip).

## mindspore.Tensor.flip

```python
mindspore.Tensor.flip(dims)
```

For more details, see [mindspore.Tensor.flip](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/mindspore/Tensor/mindspore.Tensor.flip.html).

## Differences

PyTorch: The `torch.Tensor.flip` interface has differences from `torch.flip`. Compared to `torch.flip`, `Tensor.flip` additionally supports scenarios where the `dims` input is of type int.

MindSpore: The `mindspore.flip` and `mindspore.Tensor.flip` interfaces have the same functionality as `torch.flip` and do not support input of type int.

| Categories | Subcategories | PyTorch | MindSpore | Differences |
|----------|-------------|---------|-----------|------------|
| Parameters | Parameter 1 | dims | dims | Same functionality, MindSpore does not support int input |

## Code Example

```python
# PyTorch
import numpy as np
import torch
input = torch.tensor(np.arange(1, 9).reshape((2, 2, 2)))
output = input.flip(1)
print(output)
# tensor([[[3, 4],
#          [1, 2]],

#         [[7, 8],
#          [5, 6]]])

# MindSpore
import mindspore as ms
import mindspore.ops as ops

input = ms.Tensor(np.arange(1, 9).reshape((2, 2, 2)))
output = input.flip((1, ))
print(output)
# [[[3 4]
#   [1 2]]

#  [[7 8]
#   [5 6]]]
```