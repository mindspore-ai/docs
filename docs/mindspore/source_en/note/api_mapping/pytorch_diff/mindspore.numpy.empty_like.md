# Function Differences with torch.empty_like

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/mindspore.numpy.empty_like.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.empty_like

```text
torch.empty_like(
    input,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=torch.preserve_format
) -> Tensor
```

For more information, see [torch.empty_like](https://pytorch.org/docs/1.8.1/generated/torch.empty_like.html).

## mindspore.numpy.empty_like

```text
mindspore.numpy.empty_like(prototype, dtype=None, shape=None) -> Tensor
```

For more information, see [mindspore.numpy.empty_like](https://mindspore.cn/docs/en/master/api_python/numpy/mindspore.numpy.empty_like.html).

## Differences

PyTorch: Return an uninitialized tensor of the same size and type as the input, and input only supports Tensor type inputs.

MindSpore: MindSpore API basically implements the same function as PyTorch. However, there are differences in the supported input types and parameter names. The input name of MindSpore operator is prototype, and supports three types of input: Tensor, list, and tuple. In addition, MindSpore adds a new parameter shape than PyTorch to achieve rewriting the shape of the result.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Input | Single input |input | prototype | Same function. MindSpore supports more input types |
|Parameters | Parameter 1 | dtype         | dtype     | - |
|  | Parameter 2 | -             | shape     | Implement overwritten shape. PyTorch does not have this parameter |
| | Parameter 3 | layout | - | Not involved |
| | Parameter 4 | device | - | Not involved |
| | Parameter 5 | requires_grad | - | MindSpore does not have this parameter and supports reverse derivation by default |
| | Parameter 6 | memory_format | - | Not involved |

### Code Example 1

> For the parameter shape, PyTorch empty_like operator does not have this parameter, and MindSpore shape parameter defaults to None, by which the result shape can be rewritten.

```python
# PyTorch
import torch

input_torch = torch.ones((2, 3))
torch_output = torch.empty_like(input_torch)
print(list(torch_output.shape))
# [2, 3]

# MindSpore
import mindspore

input_ms = mindspore.numpy.ones((4,1,2))
ms_output = mindspore.numpy.empty_like(input_ms, shape=[2, 3])
print(ms_output.shape)
# [2, 3]
```

### Code Example 2

> PyTorch empty_like operator supports the input type Tensor, but MindSpore supports three input types Tensor, list, and tuple. When the input is of type array, the arrays must have the same size in dimension. If the input type is not Tensor, the default data type is float32 (if dtype is not provided).

```python
# PyTorch
import torch

input_tensor_torch = torch.ones((2, 3))
torch_output = torch.empty_like(input_tensor_torch)
print(list(torch_output.shape))
# [2, 3]

# MindSpore
import mindspore

input_tensor_ms = mindspore.numpy.ones((2, 3))
ms_tensor_output = mindspore.numpy.empty_like(input_tensor_ms)
print(ms_tensor_output.shape)
# [2, 3]

input_list_ms = [[1, 2, 3],[4, 5, 6]]
ms_list_output = mindspore.numpy.empty_like(input_list_ms)
print(ms_list_output.shape)
# [2, 3]

input_tuple_ms = ((1, 2, 3),(4, 5, 6))
ms_tuple_output = mindspore.numpy.empty_like(input_tuple_ms)
print(ms_tuple_output.shape)
# [2, 3]
```
