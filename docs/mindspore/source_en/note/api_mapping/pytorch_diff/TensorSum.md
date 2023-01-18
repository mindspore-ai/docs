# Function Differences with torch.Tensor.sum

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TensorSum.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.Tensor.sum

```python
torch.Tensor.sum(dim=None, keepdim=False, dtype=None)
```

For more information, see [torch.Tensor.sum](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.sum).

## mindspore.Tensor.sum

```python
mindspore.Tensor.sum(axis=None, dtype=None, keepdims=False, initial=None)
```

For more information, see [mindspore.Tensor.sum](https://www.mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.sum.html#mindspore.Tensor.sum).

## Usage

The basic function is the same. `mindspore.Tensor.sum` can be configured with the input parameter `initial` to set the starting value of the summation, and the other input parameters are the same for both interfaces.

## Code Example

```python
import mindspore as ms

a = ms.Tensor([10, -5], ms.float32)
print(a.sum()) # 5.0
print(a.sum(initial=2)) # 7.0

import torch
b = torch.Tensor([10, -5])
print(torch.Tensor.sum(b)) # tensor(5.)
```
