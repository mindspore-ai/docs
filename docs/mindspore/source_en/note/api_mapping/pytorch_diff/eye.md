# Function Differences with torch.eye

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/eye.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.eye

```text
torch.eye(
    n,
    m=None,
    *,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False) -> Tensor
```

For more information, see [torch.eye](https://pytorch.org/docs/1.8.1/generated/torch.eye.html).

## mindspore.ops.eye

```text
mindspore.ops.eye(n, m, t) -> Tensor
```

For more information, see [mindspore.ops.eye](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.eye.html).

## Differences

PyTorch: PyTorch allows you to specify a tensor that accepts output, the layout of the returned tensor, requires_grad, and a specified device in the parameters.

MindSpore: The parameter `m` is optional in PyTorch. Without it, a tensor with the same number of columns and rows is returned, while it is required in MindSpore.

`dtype` is optional in PyTorch. Without it, defaults to `torch.float32`, while it is required in MindSpore.

There is no difference in function.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1 | n    | n     | -    |
|      | Parameter 2 | m   | m    | Specify the columns of tensor. Optional in PyTorch. Without this parameter, a tensor with the same number of columns and rows is returned. Required in MindSpore |
|     | Parameter 3 | out | - | Not involved |
|     | Parameter 4 | dtype   | t | Same function, different parameter names. Optional in PyTorch, if not default to `torch.float32`. Required in MindSpore |
|       | Parameter 5 | layout | - | Not involved |
|       | Parameter 6 | device | - | Not involved |
|       | Parameter 7 | requires_grad | - | MindSpore does not have this parameter and supports reverse derivation by default |

## Difference Analysis and Examples

### Code Example 1

> The `m` parameter can be defaulted in PyTorch. Other functions are the same.

```python
# PyTorch
import torch

# Parameter m and dtype can be defaulted
e1 = torch.eye(3)
print(e1.numpy())
# [[1., 0., 0.]
#  [0., 1., 0.]
#  [0., 0., 1.]]

# MindSpore
import mindspore
import mindspore.ops as ops
e1 = ops.eye(3, 3, mindspore.float32)
print(e1)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

### Code Example 2

> The `dtype` parameter can be defaulted in PyTorch. Other functions are the same.

```python
# PyTorch
import torch

# The parameter dtype can be defaulted
e2 = torch.eye(3, 2)
print(e2.numpy())
# [[1, 0]
#  [0, 1]
#  [0, 0]]

# MindSpore
import mindspore
import mindspore.ops as ops
e2 = ops.eye(3, 2, mindspore.float32)
print(e2)
# [[1. 0.]
#  [0. 1.]
#  [0. 0.]]
```
