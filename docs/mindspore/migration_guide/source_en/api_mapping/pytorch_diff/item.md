# Function Differences with torch.Tensor.item

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/item.md)

## torch.Tensor.item

```python
torch.Tensor.item()
```

For more information, see  [torch.Tensor.item](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.item).

## mindspore.Tensor.item

```python
mindspore.Tensor.item(index=None)
```

For more information, see  [torch.Tensor.item](https://www.mindspore.cn/docs/api/en/r1.6/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.item).

## Differences

PyTorch: Returns the value of this tensor, applicable to tensors with only one element.

MindSpore：Returns the value corresponding to the specified index in the tensor, applicable to tensors with one or more elements.

## Code Example

```python
from mindspore import Tensor
import numpy as np
import torch

x = Tensor(np.array([[1,2,3],[4,5,6]], dtype=np.float32))
print(x.item((0,1)))
# Out：
# 2.0

y = Tensor([1.0])
print(y.item())
# Out:
# [1.]

z = torch.tensor([1.0])
print(z.item())
# Out:
# 1.0
```
