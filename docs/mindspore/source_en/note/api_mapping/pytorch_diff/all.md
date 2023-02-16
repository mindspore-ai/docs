# Function Differences with torch.all

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/all.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

> For function differences between `mindspore.Tensor.all` and `torch.Tensor.all`, refer to the function differences between `mindspore.ops.all` and `torch.all`.

## torch.all

```text
torch.all(input, dim, keepdim=False, *, out=None) -> Tensor
```

For more information, see [torch.all](https://pytorch.org/docs/1.8.1/generated/torch.all.html#torch.all).

## mindspore.ops.all

```text
mindspore.ops.all(x, axis=(), keep_dims=False) -> Tensor
```

For more information, see [mindspore.ops.all](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.all.html).

## Differences

PyTorch: Perform logic AND on the elements of `input` according to the specified `dim`. `keepdim` controls whether the output and input have the same dimension. `out` can fetch the output.

MindSpore: Perform logic AND on the elements of `x` according to the specified `axis`. The `keep_dims` has the same function as PyTorch, and MindSpore does not have the `out` parameter. MindSpore has a default value for `axis`, and performs the logical AND on all elements of `x` if `axis` is the default.

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
print(torch.all(input, dim=0, keepdim=True))
# tensor([[False,  True, False, False]])

# MindSpore
import mindspore

x = mindspore.Tensor([[False, True, False, True], [False, True, False, False]])
print(mindspore.ops.all(x, axis=0, keep_dims=True))
# [[False  True False False]]
```
