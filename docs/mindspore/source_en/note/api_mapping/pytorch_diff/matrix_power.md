# Function Differences with torch.matrix_power

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/matrix_power.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The following mapping relationships can be found in this file.

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|   torch.matrix_power    |   mindspore.ops.matrix_power    |
|    torch.Tensor.matrix_power   |  mindspore.Tensor.matrix_power   |

## torch.matrix_power

```python
torch.matrix_power(input, n)
```

For more information, see [torch.matrix_power](https://pytorch.org/docs/1.8.1/generated/torch.matrix_power.html).

## mindspore.ops.matrix_power

```python
mindspore.ops.matrix_power(input, n)
```

For more information, see [mindspore.ops.matrix_power](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.matrix_power.html).

## Differences

PyTorch:

- The dimensions of input tensor can be 2-D or higher.

- Supported data types of input tensor are uint8, int8/16/32/64 and float16/32/64.

MindSpore:

- The dimensions of input tensor can only be 3-D.

- Supported data types of input tensor are float16 and float32.

There is no difference in function.

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameter 1   | input        | input         | MindSpore only supports 3-D, types of float16 and float32; PyTorch supports 2-D or higher dimensions, types of uint8, int8/16/32/64 and float16/32/64. |
|            | Parameter 2   | n            | n             | Same function |

## Code Example

```python
# PyTorch
import torch
input = torch.tensor([[0, 1], [-1, 0]], dtype=torch.int32)
y = torch.matrix_power(input, 2)
print(x.shape)
print(y)
# torch.Size([2, 2])
# tensor([[-1,  0],
#         [ 0, -1]], dtype=torch.int32)

# MindSpore
import mindspore as ms
input = ms.Tensor([[[0, 1], [-1, 0]]], dtype=ms.float32)
y = ms.ops.matrix_power(input, 2)
print(x.shape)
print(y)
# (1, 2, 2)
# [[[-1.  0.]
#   [-0. -1.]]]
```
