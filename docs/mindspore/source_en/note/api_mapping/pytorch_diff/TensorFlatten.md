# Function Differences with torch.Tensor.flatten

<a href="https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TensorFlatten.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source_en.png"></a>

## torch.Tensor.flatten

```python
torch.Tensor.flatten(input, start_dim=0, end_dim=-1)
```

For more information, see [torch.Tensor.flatten](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.flatten).

## mindspore.Tensor.flatten

```python
mindspore.Tensor.flatten(order='C', *, start_dim=0, end_dim=-1)
```

For more information, see [mindspore.Tensor.flatten](https://www.mindspore.cn/docs/en/r1.11/api_python/mindspore/Tensor/mindspore.Tensor.flatten.html#mindspore.Tensor.flatten).

## Usage

`torch.Tensor.flatten` does not support the `order` option for prioritizing row or column flatten.

`mindspore.Tensor.flatten` prioritizes row or column flatten by `order` to "C" or "F".

## Code Example

```python
import mindspore as ms

a = ms.Tensor([[1,2], [3,4]], ms.int32)
print(a.flatten())
# [1 2 3 4]
print(a.flatten('F'))
# [1 3 2 4]
print(a.flatten(start_dim=1))
# [[1 2]
#  [3 4]]

import torch

b = torch.tensor([[1, 2], [3, 4]])
print(torch.Tensor.flatten(b))
# tensor([1, 2, 3, 4])
print(torch.Tensor.flatten(b, start_dim=1))
# tensor([[1, 2],
#         [3, 4]])
```
