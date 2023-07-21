# Function Differences with torch.full

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_en/note/api_mapping/pytorch_diff/full.md)

## torch.full

```text
torch.full(
    size,
    fill_value,
    *,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False
) -> Tensor
```

For more information, see [torch.full](https://pytorch.org/docs/1.8.1/generated/torch.full.html).

## mindspore.ops.full

```text
mindspore.ops.full(size, fill_value, *, dtype=None) -> Tensor
```

For more information, see [mindspore.ops.full](https://mindspore.cn/docs/en/r1.11/api_python/ops/mindspore.ops.full.html).

## Differences

PyTorch: Return the tensor of the given size filled with fill_value.

MindSpore:  MindSpore API implements basically the same function as PyTorch, but with different parameter names.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
|Parameters | Parameter 1 | size | size |Consistent function |
| | Parameter 2 | fill_value | fill_value | For the parameter fill_value, PyTorch full operator supports the number type, and MindSpore does not support the plural type. |
|  | Parameter 3 | dtype         | dtype     | Consistent function       |
| | Parameter 4 | out           | -         | Not involved |
| | Parameter 5 | layout | - | Not involved |
| | Parameter 6 | device | - | Not involved |
| | Parameter 7 | requires_grad | - | MindSpore does not have this parameter and supports reverse derivation by default |

### Code Example 1

```python
# PyTorch
import torch

torch_output = torch.full((2, 3), 1)
print(torch_output.numpy())
# [[1 1 1]
#  [1 1 1]]

# MindSpore
import mindspore

ms_tensor_output = mindspore.ops.full((2, 3), 1)
print(ms_tensor_output)
# [[1 1 1]
#  [1 1 1]]
```
