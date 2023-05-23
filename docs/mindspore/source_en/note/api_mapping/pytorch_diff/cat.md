# Differences with torch.cat

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

MindSpore's API function is consistent with PyTorch.

PyTorch: Splice the input Tensor on the specified axis. When the data precision of the input Tensors is different, the low precision Tensor will be automatically converted to high precision Tensor.

MindSpore: Currently, the data type and precision of the the input Tensors are required to remain the same. If not, the low-precision Tensor can be converted to a high-precision Tensor through ops.cast and then call the concat operator.

|Categories|Subcategories|PyTorch|MindSpore|Differences|
| --- | --- | --- | --- |---|
| Input | Single input | tensors  | tensors | The data type and precision of the `tensors` in MindSpore are required to remain the same, while the precision of the `tensors` in PyTorch can be different |
|Parameters | Parameter 1 | dim | axis | Different parameter names |
|  | Parameter 2  | out | - | For details, see [General Difference Parameter Table](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table) |

### Code Example

> MindSpore currently requires that the data type and precision of the input Tensors are consistent. If it is inconsistent, the low-precision tensor can be converted to a high-precision type through ops.cast before calling the concat operator.

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
