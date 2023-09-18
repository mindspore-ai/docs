# Differences with torch.nn.functional.fold

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/mindspore.ops.fold.md)

## torch.nn.functional.fold

```text
torch.nn.functional.fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)
```

For more information, see [torch.nn.functional.fold](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.fold).

## mindspore.ops.fold

```text
mindspore.ops.fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)
```

For more information, see [mindspore.ops.fold](https://www.mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.fold.html).

## Differences

PyTorch：Combines an array of sliding local blocks into a large containing tensor.

MindSpore：MindSpore API implements basically the same function as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
|Parameter | Parameter1 | input | input | |
| | Parameter2 | output_size | output_size | Pytorch: int or tuple, MindSpore: 1D tensor with 2 elements of data type int. |
| | Parameter3 | kernel_size | kernel_size |- |
| | Parameter4 | dilation | dilation |- |
| | Parameter5 | padding | padding |- |
| | Parameter6 | stride | stride |- |

### Code Example 1

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
import numpy as np
x = np.random.randn(1, 3 * 2 * 2, 12)
input = torch.tensor(x, dtype=torch.float32)
output = torch.nn.functional.fold(input, output_size=(4, 5), kernel_size=(2, 2))
print(output.detach().shape)
# torch.Size([1, 3, 4, 5])

# MindSpore
import mindspore
import numpy as np
x = np.random.randn(1, 3 * 2 * 2, 12)
input = mindspore.Tensor(x, mindspore.float32)
output_size = mindspore.Tensor((4, 5), mindspore.int32)
output = mindspore.ops.fold(input, output_size, kernel_size=(2, 2))
print(output.shape)
# (1, 3, 4, 5)
```
