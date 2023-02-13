# Function Differences with torch.prod

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/prod.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

> For function differences between `mindspore.Tensor.prod` and `torch.Tensor.prod`, refer to the function differences between `mindspore.ops.prod` and `torch.prod`.

## torch.prod

```text
torch.prod(input, dim, keepdim=False, *, dtype=None) -> Tensor
```

For more information, see [torch.prod](https://pytorch.org/docs/1.8.1/generated/torch.prod.html#torch.prod).

## mindspore.ops.prod

```text
mindspore.ops.prod((x, axis=(), keep_dims=False) -> Tensor
```

For more information, see [mindspore.ops.prod](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.prod.html.

## Differences

PyTorch: Find the product on elements in `input` based on the specified `dim`. `keepdim` controls whether the output and input have the same dimension. `dtype` sets the data type of the output Tensor.

MindSpore: Find the product on the elements in `x` by the specified `axis`. The function of `keep_dims` is the same as PyTorch. MindSpore does not have a `dtype` parameter. MindSpore has a default value for `axis`, which is the product of all elements of `x` if `axis` is the default value.

| Categories | Subcategories| PyTorch | MindSpore |Differences |
| ---- | ----- | ------- | --------- |------------------ |
| Parameters | Parameter 1 | input   | x         | Same function, different parameter names                    |
|      | Parameter 2 | dim   | axis      | PyTorch must pass `dim` and only one integer. MindSpore `axis` can be passed as an integer, a tuples of integers or a list of integers |
|      | Parameter 3 | keepdim   | keep_dims | Same function, different parameter names |
|      | Parameter 4 | dtype   | -         | PyTorch `dtype` can set the data type of the output Tensor. MindSpore does not have this parameter |

### Code Example

```python
# PyTorch
import torch

input = torch.tensor([[1, 2.5, 3, 1], [2.5, 3, 2, 1]], dtype=torch.float32)
print(torch.prod(input, dim=1, keepdim=True))
# tensor([[ 7.5000],
#         [15.0000]])
print(torch.prod(input, dim=1, keepdim=True, dtype=torch.int32))
# tensor([[ 6],
#         [12]], dtype=torch.int32)

# MindSpore
import mindspore

x = mindspore.Tensor([[1, 2.5, 3, 1], [2.5, 3, 2, 1]], dtype=mindspore.float32)
print(mindspore.ops.prod(x, axis=1, keep_dims=True))
# [[ 7.5]
#  [15. ]]
```
