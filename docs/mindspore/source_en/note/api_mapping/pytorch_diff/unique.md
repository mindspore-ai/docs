# Function Differences with torch.unique

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/unique.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.unique

```python
torch.unique(
    *args,
    **kwargs
)
```

For more information, see [torch.unique](https://pytorch.org/docs/1.8.1/generated/torch.unique.html#torch.unique).

## mindspore.ops.unique

```python
mindspore.ops.unique(x)
```

For more information, see [mindspore.ops.unique](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.unique.html#mindspore.ops.unique).

## Differences

PyTorch: Deduplicate the elements in the Tensor. The parameter `sorted` can be set to determine whether the output is sorted in ascending order. Set the parameter `return_inverse` to determine whether to output the index of each element of the input Tensor in the output Tensor. Set the parameter `return_counts` to determine whether to output the number of each unique value in the input Tensor; set the parameter `dim` to specify the dimension of the unique. MindSpore does not support these functions.

MindSpore: Deduplicate the elements in the Tensor, as well as return the position index of each element of the input Tensor in the output Tensor.

 Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter1 | input   | x | Input Tensor with different parameter names |
|  | Parameter2 | sorted | - | When sorted is True, the output Tensor is sorted in ascending order; when sorted is False, it is sorted in the original order |
|  | Parameter3 | return_inverse | - | When return_inverse is True, the index position of each element of the input Tensor in the output Tensor is returned |
|  | Parameter4 | return_counts | - | When return_counts is True, the number of each element of the output Tensor in the input Tensor is returned |
|  | Parameter5 | dim | - | Specify the dimension of unique |

## Code Example

```python
# In MindSpore
import mindspore

x = mindspore.Tensor([1, 3, 2, 3], mindspore.float32)
output, idx = mindspore.ops.unique(x)
print(output)
# [1. 3. 2.]
print(idx)
# [0 1 2 1]

# In PyTorch
import torch

output, inverse_indices, counts = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long), sorted=True, return_inverse=True, return_counts=True)
print(output)
# tensor([1, 2, 3])
print(inverse_indices)
# tensor([0, 2, 1, 2])
print(counts)
# tensor([1, 1, 2])

# Example of using unique with dim
output, inverse_indices = torch.unique(torch.tensor([[3, 1], [1, 2]], dtype=torch.long), sorted=True, return_inverse=True, dim=0)
print(output)
# tensor([[1, 2],
#         [3, 1]])
print(inverse_indices)
# tensor([1, 0])
```