# Function Differences with torch.matrix_power

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/matrix_power.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

> For the functional differences between `mindspore.Tensor.matrix_power` and `torch.Tensor.matrix_power` , refer to the functional differences between `mindspore.ops.matrix_power` and `torch.matrix_power` .

## torch.matrix_power

```python
torch.matrix_power(input, n)
```

For more information, see [torch.matrix_power](https://pytorch.org/docs/1.8.1/generated/torch.matrix_power.html).

## mindspore.ops.scatter

```python
mindspore.ops.matrix_power(x, n)
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

## Code Example

```python
# PyTorch
import torch
x = torch.tensor([[0, 1], [-1, 0]], dtype=torch.int32)
y = torch.matrix_power(x, 2)
print(x.shape)
print(y)
# torch.Size([2, 2])
# tensor([[-1,  0],
#         [ 0, -1]], dtype=torch.int32)

# MindSpore
import mindspore as ms
x = ms.Tensor([[[0, 1], [-1, 0]]], dtype=ms.float32)
y = ms.ops.matrix_power(x, 2)
print(x.shape)
print(y)
# (1, 2, 2)
# [[[-1.  0.]
#   [-0. -1.]]]
```
