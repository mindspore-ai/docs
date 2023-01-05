# Function Differences with torch.cat

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/cat.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.cat

```text
torch.cat(
    tensors,
    dim=0,
    *,
    out=None
) -> Tensor
```

For more information, see [torch.cat](https://pytorch.org/docs/1.8.1/generated/torch.cat.html).

## mindspore.ops.cat

```text
mindspore.ops.cat(tensors, axis=0) -> Tensor
```

For more information, see [mindspore.ops.cat](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.cat.html).

## Differences

PyTorch: Splice the input Tensor on the specified axis. When the data type of input tensor is different, the low precision tensor will be automatically converted to high precision tensor.

MindSpore: The implementation function of the API in MindSpore is basically the same as that of PyTorch. Currently, the data type of the input tensors are required to remain the same. If not, the low-precision tensor can be converted to a high-precision tensor through ops.cast and then call the concat operator.

|Categories|Subcategories|PyTorch|MindSpore|Differences|
| --- | --- | --- | --- |---|
| Input | Single input | tensors  | tensors | Consistent function |
|Parameters | Parameter 1 | dim | axis |Consistent function, different parameter names |
|  | Parameter 2  | out | - | Not involved          |

## Code Example 1

> MindSpore currently requires that the data type of the input tensor is consistent. If it is inconsistent, the low-precision tensor can be converted to a high-precision type through ops.cast before calling the concat operator.

```python
# PyTorch
import torch

torch_x1 = torch.Tensor([[0, 1], [2, 3]]).type(torch.float32)
torch_x2 = torch.Tensor([[0, 1], [2, 3]]).type(torch.float32)
torch_x3 = torch.Tensor([[0, 1], [2, 3]]).type(torch.float16)

torch_output = torch.cat((torch_x1, torch_x2, torch_x3))
print(torch_output.numpy())
# [[0. 1.]
#  [2. 3.]
#  [0. 1.]
#  [2. 3.]
#  [0. 1.]
#  [2. 3.]]

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

# In MindSpore，converting low precision to high precision is needed before cat.
ms_x1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
ms_x2 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
ms_x3 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float16))

ms_x3 = mindspore.ops.cast(ms_x2, mindspore.float32)
output = mindspore.ops.cat((ms_x1, ms_x2, ms_x3))
print(output)
# [[0. 1.]
#  [2. 3.]
#  [0. 1.]
#  [2. 3.]
#  [0. 1.]
#  [2. 3.]]
```