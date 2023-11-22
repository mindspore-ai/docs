# Differences with torch.Tensor.size

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/shape.md)

## torch.Tensor.size

```text
torch.Tensor.size() -> Tensor
```

For more information, see [torch.Tensor.size](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.size).

## mindspore.Tensor.shape

```text
mindspore.Tensor.shape
```

For more information, see [mindspore.Tensor.shape](https://www.mindspore.cn/docs/en/r2.3/api_python/mindspore/Tensor/mindspore.Tensor.shape.html).

## Differences

PyTorch: The size() method returns the shape of the Tensor.

MindSpore: Functionally consistent, but mindspore.Tensor.shape is a property, not a method.

### Code Example

```python
# PyTorch
import torch

input = torch.randn(3, 4, 5)
print(input.size())
# torch.Size([3, 4, 5])

# MindSpore
import mindspore.ops as ops

input = ops.randn(3, 4, 5)
print(input.shape)
# (3, 4, 5)
```
