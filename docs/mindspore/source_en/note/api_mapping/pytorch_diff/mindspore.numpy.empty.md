# Function Differences with torch.empty

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/mindspore.numpy.empty.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.empty

```text
torch.empty(
    *size,
    *,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False
    memory_format=torch.contiguous_format
) -> Tensor
```

For more information, see [torch.empty](https://pytorch.org/docs/1.8.1/generated/torch.empty.html).

## mindspore.numpy.empty

```text
mindspore.numpy.empty(shape, dtype=mstype.float32) -> Tensor
```

For more information, see [mindspore.numpy.empty](https://mindspore.cn/docs/en/master/api_python/numpy/mindspore.numpy.empty.html).

## Differences

PyTorch: Return an uninitialized tensor, the shape of which is defined by size.

MindSpore: MindSpore API basically implements the same function as PyTorch, but the default value of the dtype parameter is different.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
|Parameters | Parameter 1 | size | shape |Same function, different parameter names |
| | Parameter 2 | out           | -         | Not involved |
|  | Parameter 3 | dtype         | dtype     | Same function, different default values |
| | Parameter 4 | layout | - | Not involved |
| | Parameter 5 | device | - | Not involved |
| | Parameter 6 | requires_grad | - | MindSpore does not have this parameter and supports reverse derivation by default |
| | Parameter 7 | pin_memory | - | Not involved |
| | Parameter 8 | memory_format | - | Not involved |

### Code Example 1

> For the parameter dtype, PyTorch defaults to None, and the output type is torch.float32, while MindSpore defaults to mstype.float32.

```python
# PyTorch
import torch

torch_output = torch.empty(2, 3)
print(list(torch_output.shape))
# [2, 3]

# MindSpore
import mindspore

ms_output = mindspore.numpy.empty((2, 3))
print(ms_output.shape)
# [2, 3]
```
