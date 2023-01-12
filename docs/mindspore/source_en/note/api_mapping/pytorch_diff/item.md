# Function Differences with torch.Tensor.item

<a href="https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_en/note/api_mapping/pytorch_diff/item.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png"></a>

## torch.Tensor.item

```python
torch.Tensor.item()
```

For more information, see  [torch.Tensor.item](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.item).

## mindspore.Tensor.item

```python
mindspore.Tensor.item(index=None)
```

For more information, see  [mindspore.Tensor.item](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore/Tensor/mindspore.Tensor.item.html#mindspore.Tensor.item).

## Differences

PyTorch: Returns the value of this tensor, applicable to tensors with only one element.

MindSpore：Returns the value corresponding to the specified index in the tensor, applicable to tensors with one or more elements.

## Code Example

```python
import mindspore as ms
import numpy as np
import torch

x = ms.Tensor(np.array([[1,2,3],[4,5,6]], dtype=np.float32))
print(x.item((0,1)))
# Out：
# 2.0

y = ms.Tensor([1.0])
print(y.item())
# Out:
# 1.0

z = torch.tensor([1.0])
print(z.item())
# Out:
# 1.0
```
