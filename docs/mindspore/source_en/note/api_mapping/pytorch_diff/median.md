# Function Differences with torch.median

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/median.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The following mapping relationships can be found in this file.

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|   torch.median    |   mindspore.ops.median    |
|    torch.Tensor.median   |  mindspore.Tensor.median   |

## torch.median

```text
torch.median(input, dim=-1, keepdim=False, *, out=None) -> Tensor
```

For more information, see [torch.median](https://pytorch.org/docs/1.8.1/generated/torch.median.html#torch.median).

## mindspore.ops.median

```text
mindspore.ops.median(x, axis=-1, keepdims=False) -> Tensor
```

For more information, see [mindspore.ops.median](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.median.html).

## Differences

PyTorch: Output the median and index of `input` according to the specified `dim`. `keepdim` controls whether the output and input have the same dimension. Return the median of all elements when the input has only `input`, or the median and index of the specified dimension when the input contains `dim`. `out` can get the output.

MindSpore: Output the median and index of `x` according to the specified `axis`. The `keepdims` function is identical to PyTorch. Unlike Pytorch, MindSpore returns the median and index in the specified dimension, regardless of whether the input contains `axis` or not. MindSpore does not have `out` parameter.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| Parameters  | Parameter 1 | input   | x         | Same function, different parameter names                    |
|      | Parameter 2 | dim   | axis      | Same function, different parameter names |
|      | Parameter 3 | keepdim   | keepdims | Same function, different parameter names |
|      | Parameter 4 | out   | -         | PyTorch's `out` can get the output. MindSpore does not have this parameter |

### Code Example

```python
# PyTorch
import torch

input = torch.tensor([[1, 2.5, 3, 1], [2.5, 3, 2, 1]], dtype=torch.float32)
print(torch.median(input))
# tensor(2.)
print(torch.median(input, dim=1, keepdim=True))
# torch.return_types.median(
# values=tensor([[1.],
#         [2.]]),
# indices=tensor([[3],
#         [2]]))

# MindSpore
import mindspore

x = mindspore.Tensor([[1, 2.5, 3, 1], [2.5, 3, 2, 1]], dtype=mindspore.float32)
print(mindspore.ops.median(x, axis=1, keepdims=True))
# (Tensor(shape=[2, 1], dtype=Float32, value=
# [[ 1.00000000e+00],
#  [ 2.00000000e+00]]), Tensor(shape=[2, 1], dtype=Int64, value=
# [[3],
#  [2]]))
```
