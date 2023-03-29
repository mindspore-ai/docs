# Function Differences with torch.var

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/var.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

The following mapping relationships can be found in this file.

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|    torch.var     |  mindspore.ops.var   |
|   torch.Tensor.var    |   mindspore.Tensor.var    |

## torch.var

```python
torch.var(input, dim, unbiased=True, keepdim=False, *, out=None)
```

For more information, see [torch.var](https://pytorch.org/docs/1.8.1/generated/torch.var.html).

## mindspore.ops.var

```python
mindspore.ops.var(input, axis=None, ddof=0, keepdims=False)
```

For more information, see [mindspore.ops.var](https://www.mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.var.html).

## Differences

PyTorch: Output the variance of the Tensor in each dimension, or the variance of the specified dimension according to `dim`. If `unbiased` is True, use Bessel for correction; if False, use bias estimation to calculate the variance. `keepdim` controls whether the output and input dimensions are the same.

MindSpore: Output the variance of the Tensor in each dimension, or the variance of the specified dimension according to `axis`. If `ddof` is a boolean, it has the same effect as `unbiased`; if `ddof` is an integer, the divisor used in the calculation is N-ddof, where N denotes the number of elements. `keepdim` controls whether the output and the input have the same dimensionality.

| Categories | Subcategories | PyTorch | MindSpore | Differences  |
| --- |---------------|---------| --- |-------------|
| Parameters       | Parameter 1       | input         | input          | Same function, different parameter names |
|            | Parameter 2       | dim          | axis |  Same function, different parameter names  |
|            | Parameter 3       | unbiased          | ddof | `ddof` is the same as `unbiased` when it is a boolean value |
|            | Parameter 4       | keepdim      | keepdims | Same function, different parameter names |
|            | Parameter 5       | out       | - |  MindSpore does not have this parameter  |

### Code Example

```python
# PyTorch
import torch

input = torch.tensor([[[9, 7, 4, -10],
                       [-9, -2, 1, -2]]], dtype=torch.float32)
print(torch.var(input, dim=2, unbiased=True, keepdim=True))
# tensor([[[73.6667],
#          [18.0000]]])

# MindSpore
import mindspore as ms

input = ms.Tensor([[[9, 7, 4, -10],
                    [-9, -2, 1, -2]]], ms.float32)
print(ms.ops.var(input, axis=2, ddof=True, keepdims=True))
# [[[73.666664]
#   [17.999998]]]
```
