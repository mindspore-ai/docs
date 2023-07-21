# Differences with torch.Tensor.sum

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TensorSum.md)

## torch.Tensor.sum

```python
torch.Tensor.sum(dim=None, keepdim=False, dtype=None)
```

For more information, see [torch.Tensor.sum](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.sum).

## mindspore.Tensor.sum

```python
mindspore.Tensor.sum(axis=None, dtype=None, keepdims=False, initial=None)
```

For more information, see [mindspore.Tensor.sum](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore/Tensor/mindspore.Tensor.sum.html#mindspore.Tensor.sum).

## Differences

MindSpore API has the same function as that of PyTorch, but the number and order of parameters are not the same.

PyTorch: No parameter `initial`. The relative order of parameters `keepdim` and `dtype` is different from MindSpore.

MindSpore: The starting value of the summation can be configured with the parameter `initial`. The relative order of the parameters `keepdim` and `dtype` differs from that of PyTorch.

| Categories | Subcategories | PyTorch | MindSpore | Differences  |
| --- |---------------|---------| --- |-------------|
| Parameters | Parameter 1 | dim | axis | Both parameters have different names, and both indicate the specified dimension of the summation |
|      | Parameter 2 | keepdim | dtype | The relative order of `keepdim` and `dtype` are different |
|      | Parameter 3 | dtype | keepdims | The relative order of `keepdims` and `dtype` are different |
|      | Parameter 4 | - | initial | MindSpore can configure the starting value of the summation with the parameter `initial`, and PyTorch has no parameter `initial`. |

## Code Example

```python
# PyTorch
import torch

b = torch.Tensor([10, -5])
print(torch.Tensor.sum(b))
# tensor(5.)

# MindSpore
import mindspore as ms

a = ms.Tensor([10, -5], ms.float32)
print(a.sum())
# 5.0
print(a.sum(initial=2))
# 7.0
```