# Differences with torch.var_mean

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/var_mean.md)

## torch.var_mean

```python
torch.var_mean(input, dim, unbiased=True, keepdim=False, *, out=None)
```

For more information, see [torch.var_mean](https://pytorch.org/docs/1.8.1/generated/torch.var_mean.html).

## mindspore.ops.var_mean

```python
mindspore.ops.var_mean(input, axis=None, ddof=0, keepdims=False)
```

For more information, see [mindspore.ops.var_mean](https://www.mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.var_mean.html).

## Differences

PyTorch: Output the variance and mean value of the Tensor in each dimension, or the variance and mean value of the specified dimension according to `dim`. If `unbiased` is True, use Bessel for correction; if False, use bias estimation to calculate the variance. `keepdim` controls whether the output and input dimensions are the same.

MindSpore: Output the variance and mean value of the Tensor in each dimension, or the variance and mean value of the specified dimension according to `axis`. If `ddof` is a boolean, it has the same effect as `unbiased`; if `ddof` is an integer, the divisor used in the calculation is N-ddof, where N denotes the number of elements. `keepdim` controls whether the output and the input have the same dimensionality.

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
print(torch.var_mean(input, dim=2, unbiased=True, keepdim=True))
# (tensor([[[73.6667],
#          [18.0000]]]), tensor([[[ 2.5000],
#          [-3.0000]]]))

# MindSpore
import mindspore as ms

input = ms.Tensor([[[9, 7, 4, -10],
                    [-9, -2, 1, -2]]], ms.float32)
print(ms.ops.var_mean(input, axis=2, ddof=True, keepdims=True))
# (Tensor(shape=[1, 2, 1], dtype=Float32, value=
# [[[ 7.36666641e+01],
#   [ 1.79999981e+01]]]), Tensor(shape=[1, 2, 1], dtype=Float32, value=
# [[[ 2.50000000e+00],
#   [-3.00000000e+00]]]))
```
