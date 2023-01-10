# Function Differences with torch.nn.AdaptiveAvgPool2d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/AdaptiveAvgPool2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.AdaptiveAvgPool2d

```text
torch.nn.AdaptiveAvgPool2d(output_size)(input) -> Tensor
```

For more information, see [torch.nn.AdaptiveAvgPool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveAvgPool2d.html).

## mindspore.nn.AdaptiveAvgPool2d

```text
class mindspore.nn.AdaptiveAvgPool2d(output_size)(x) -> Tensor
```

For more information, see [mindspore.nn.AdaptiveAvgPool2d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.AdaptiveAvgPool2d.html).

## Differences

PyTorch: For the input 3D or 4D Tensor, 2D adaptive averaging pooling operation is used, specifying the output size as H x W and the number of features in the output equal to the number of features in the input. output_size can be a tuple (H, W) of int types H and W, or an int value representing the same H and W, or None to indicate that the output size will be the same as the input. The input and output data formats can be "NCHW" and "CHW", where N is the batch size, C is the number of channels, H is the feature height and W is the feature width.

MindSpore: MindSpore API implements the same function as PyTorch, and only the parameter names are different.

| Categories | Subcategories | PyTorch | MindSpore| Differences |
| ---- | ----- | ------- | --------- | ------------ |
| Input | Single input | input   | x | Both are input 3D or 4D Tensor |
| Parameter | Parameter 1 | output_size | output_size | - |

### Code Example 1

> The two APIs achieve the same function and have the same usage. The input is a 3D Tensor with data size (C, H, W) and output_size=(None, new_W), and the output of PyTorch and MindSpore AdaptiveAvgPool2D is the same with data size (C, H, new_W).

```python
# case 1: output_size = (None, 2)
# PyTorch
import torch

# torch_input.shape = (1, 3, 3)
torch_input = torch.tensor([[[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]]], dtype=torch.float32)
output_size = (None, 2)
torch_adaptive_avg_pool_2d = torch.nn.AdaptiveAvgPool2d(output_size)
# torch_output = (1, 3, 2)
torch_output = torch_adaptive_avg_pool_2d(torch_input)
torch_out_np = torch_output.numpy()
print(torch_out_np)
# [[[1.5 2.5]
#   [4.5 5.5]
#   [7.5 8.5]]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

# ms_input.shape = (1, 3, 3)
ms_input = Tensor(np.array([[[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]]]), mindspore.float32)
output_size = (None, 2)
ms_adaptive_avg_pool_2d = mindspore.nn.AdaptiveAvgPool2d(output_size)
# ms_output = (1, 3, 2)
ms_output = ms_adaptive_avg_pool_2d(ms_input)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# [[[1.5 2.5]
#   [4.5 5.5]
#   [7.5 8.5]]]
```
