# Differences with torch.take

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/take.md)

## torch.Tensor.take

```python
torch.Tensor.take(indices)
```

For more information, see [torch.Tensor.take](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.take).

## mindspore.Tensor.take

```python
mindspore.Tensor.take(indices, axis=None, mode='clip')
```

For more information, see [mindspore.Tensor.take](https://www.mindspore.cn/docs/en/r2.3/api_python/mindspore/Tensor/mindspore.Tensor.take.html).

## Usage

MindSpore API function is basically the same as pytorch.

PyTorch: Obtain the elements in the Tensor. No dimension can be specified, and use the expanded input array. Throws an exception if the index is out of range.

MindsPore: Obtain the elements in the Tensor in the specified dimension. The dimension can be specified, and the expanded input array is used by default. If the index is out of range: throw an exception if mode is 'raise'; wrap if mode is 'wrap'; crop to range if mode is 'raise'.

| Categories | Subcategories| PyTorch | MindSpore |Differences |
| ---- | ----- | ------- | --------- |------------------ |
| Parameters       | Parameter 1       | indices        | indices   |  None  |
|            | Parameter 2       |               | axis       | Specify the index to get, which is not supported by Pytorch. |
|            | Parameter 3       |               | mode       | Pytorch does not support behavior mode selection if the index is out of range |

## Code Example 1

```python
# PyTorch
import torch
input_x1 = torch.tensor([[4, 3, 5], [6, 7, 8]])
indices = torch.tensor([0, 2, 4])
output = input_x1.take(indices)
print(output)
# tensor([4, 5, 7])

# MindSpore
import mindspore as ms
input_x1 = ms.Tensor([[4, 3, 5], [6, 7, 8]])
indices = ms.Tensor([0, 2, 4])
output = input_x1.take(indices)
print(output)
# [4 5 7]
```

## Code Example 2

```python
# PyTorch
import torch
input_x1 = torch.tensor([[4, 3, 5], [6, 7, 8]])
indices = torch.tensor([0, 2, 8])
output = input_x1.take(indices)
print(output)
# IndexError: out of range: tried to access index 8 on a tensor of 6 elements

# MindSpore
import mindspore as ms
input_x1 = ms.Tensor([[4, 3, 5], [6, 7, 8]])
indices = ms.Tensor([0, 2, 8])
output = input_x1.take(indices, mode='clip')
print(output)
# [4 5 8]
```
