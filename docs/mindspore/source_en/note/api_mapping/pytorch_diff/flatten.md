# Differences with torch.flatten

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/flatten.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.flatten

```python
torch.flatten(
    input,
    start_dim=0,
    end_dim=-1
)
```

For more information, see [torch.flatten](https://pytorch.org/docs/1.8.1/generated/torch.flatten.html).

## mindspore.ops.flatten

```python
mindspore.ops.flatten(input, order='C', *, start_dim=1, end_dim=-1)
```

For more information,
see [mindspore.ops.flatten](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.flatten.html).

## Differences

PyTorch: Supports the flatten operation of elements by specified dimensions, where `start_dim` defaults to 0 and `end_dim` defaults to -1.

MindSpore：Supports the flatten operation of elements by specified dimensions, where `start_dim` defaults to 1 and `end_dim` defaults to -1. Prioritizes row or column flatten by `order` to "C" or "F".

| Categories | Subcategories | PyTorch   | MindSpore | Differences                                                   |
|------------|---------------|-----------|-----------|---------------------------------------------------------------|
| Parameter  | Parameter 1   | input     | input     | Same function                                                 |
|            | Parameter 2   | -         | order     | Flatten order, PyTorch does not have this Parameter           |
|            | Parameter 3   | start_dim | start_dim | Same function                                                 |
|            | Parameter 4   | end_dim   | end_dim   | Same function                                                 |

## Code Example

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# MindSpore
input_tensor = ms.Tensor(np.ones(shape=[1, 2, 3, 4]), ms.float32)
output = ops.flatten(input_tensor)
print(output.shape)
# Out：
# (1, 24)

input_tensor = ms.Tensor(np.ones(shape=[1, 2, 3, 4]), ms.float32)
output = ops.flatten(input_tensor, start_dim=2)
print(output.shape)
# Out：
# (1, 2, 12)

# PyTorch
input_tensor = torch.Tensor(np.ones(shape=[1, 2, 3, 4]))
output1 = torch.flatten(input=input_tensor, start_dim=1)
print(output1.shape)
# Out：
# torch.Size([1, 24])

input_tensor = torch.Tensor(np.ones(shape=[1, 2, 3, 4]))
output2 = torch.flatten(input=input_tensor, start_dim=2)
print(output2.shape)
# Out：
# torch.Size([1, 2, 12])
```
