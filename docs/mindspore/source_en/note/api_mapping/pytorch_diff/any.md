# Function Differences with torch.any

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/any.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

> For function differences between `mindspore.Tensor.any` and `torch.Tensor.any`, refer to the function differences between `mindspore.ops.any` and `torch.any`.

## torch.any

```text
torch.any(input, dim, keepdim=False, *, out=None) -> Tensor
```

For more information, see [torch.any](https://pytorch.org/docs/1.8.1/generated/torch.any.html#torch.any).

## mindspore.ops.any

```text
mindspore.ops.any((x, axis=(), keep_dims=False) -> Tensor
```

For more information, see [mindspore.ops.any](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.any.html).

## Differences

PyTorch: Perform logic OR on the elements of `input` according to the specified `dim`. `keepdim` controls whether the output and input have the same dimension. `out` can fetch the output.

MindSpore: Perform logic OR on the elements of `x` according to the specified `axis`. The `keep_dims` has the same function as PyTorch, and MindSpore does not have the `out` parameter. MindSpore has a default value for `axis`, and performs the logical OR on all elements of `x` if `axis` is the default.

| Categories | Subcategories| PyTorch | MindSpore |Differences |
| ---- | ----- | ------- | --------- |------------------ |
| Parameters | Parameter 1 | input   | x         | Same function, different parameter names                    |
|      | Parameter 2 | dim   | axis      | PyTorch must pass `dim` and only one integer. MindSpore `axis` can be passed as an integer, a tuples of integers or a list of integers |
|      | Parameter 3 | keepdim   | keep_dims | Same function, different parameter names |
|      | Parameter 4 | out   | -         | PyTorch `out` can get the output. MindSpore does not have this parameter |

### Code Example

```python
# PyTorch
import torch

input = torch.tensor([[False, True, False, True], [False, True, False, False]])
print(torch.any(input, dim=0, keepdim=True))
# tensor([[False,  True, False,  True]])

# MindSpore
import mindspore

x = mindspore.Tensor([[False, True, False, True], [False, True, False, False]])
print(mindspore.ops.any(x, axis=0, keep_dims=True))
# [[False  True False  True]]
```
