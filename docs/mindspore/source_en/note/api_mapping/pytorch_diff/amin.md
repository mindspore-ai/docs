# Differences with torch.amin

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/amin.md)

The following mapping relationships can be found in this file.

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|    torch.amin     |  mindspore.ops.amin   |
|   torch.Tensor.amin    |   mindspore.Tensor.amin    |

## torch.amin

```text
torch.amin(input, dim, keepdim=False, *, out=None) -> Tensor
```

For more information, see [torch.amin](https://pytorch.org/docs/1.8.1/generated/torch.amin.html#torch.amin).

## mindspore.ops.amin

```text
mindspore.ops.amin(x, axis=(), keepdims=False) -> Tensor
```

For more information, see [mindspore.ops.amin](https://mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.amin.html).

## Differences

PyTorch: Find the minimum element of `input` according to the specified `dim`. `keepdim` controls whether the output and the input have the same dimension. `out` can get the output.

MindSpore: Find the minimum element of `x` according to the specified `axis`. The `keepdims` function is identical to PyTorch. MindSpore does not have `out` parameter. MindSpore `axis` has a default value, and finds the minimum value of all elements of `x` if `axis` is the default value.

| Categories | Subcategories| PyTorch | MindSpore |Differences |
| ---- | ----- | ------- | --------- |------------------ |
| Parameters | Parameter 1 | input   | x         | Same function, different parameter names                    |
|      | Parameter 2 | dim   | axis      | MindSpore `axis` has a default value, while PyTorch `dim` has no default value |
|      | Parameter 3 | keepdim   | keepdims | Same function, different parameter names |
|      | Parameter 4 | out   | -         | PyTorch `out` can get the output. MindSpore does not have this parameter |

### Code Example

```python
# PyTorch
import torch

input = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.float32)
print(torch.amin(input, dim=0, keepdim=True))
# tensor([[1., 2., 1.]])

# MindSpore
import mindspore

x = mindspore.Tensor([[1, 2, 3], [3, 2, 1]], dtype=mindspore.float32)
print(mindspore.ops.amin(x, axis=0, keepdims=True))
# [[1. 2. 1.]]
```
