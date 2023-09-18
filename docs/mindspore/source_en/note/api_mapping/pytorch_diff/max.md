# Differences with torch.max

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/max.md)

## torch.max

```python
torch.max(input, dim, keepdim=False, *, out=None)

torch.max(input, other, *, out=None)
```

For more information, see [torch.max](https://pytorch.org/docs/1.8.1/torch.html#torch.max).

## mindspore.ops.max

```python
mindspore.ops.max(input, axis=None, keepdims=False, *, initial=None, where=None)
```

For more information, see [mindspore.ops.max](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.max.html).

## Differences

PyTorch: Output of `torch.max(input, dim, keepdim=False, *, out=None)` is tuple(max, index of max).

MindSpore: When the axis is None or the shape is empty in MindSpore, the keepdims and subsequent parameters are not effective, and the function is consistent with torch.max(input), and the index returned is fixed at 0. Otherwise, the output is a tuple (max, index of max), which is consistent with torch.max(input, dim, keepdim=False, *, out=None).

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
|Parameters | Parameter 1 | input        | input       | Consistent |
|      | Parameter 2 | dim       | axis      | Same function, different parameter names |
| | Parameter 3 | keepdim    | keepdims     | Same function, different parameter names       |
| | Parameter 4 | -      |initial    | Not involved        |
| | Parameter 5 |  -     |where    | Not involved        |
| | Parameter 6 | out    | -         | Not involved |

PyTorch: `torch.max(input, other, *, out=None)` is used in the same way as `mindspore.ops.maximum` .

## Code Example

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

np_x = np.array([[-0.0081, -0.3283, -0.7814, -0.0934],
                 [1.4201, -0.3566, -0.3848, -0.1608],
                 [-0.0446, -0.1843, -1.1348, 0.5722],
                 [-0.6668, -0.2368, 0.2790, 0.0453]]).astype(np.float32)

# torch.max(input, dim, keepdim=False, *, out=None)
input_x = torch.tensor(np_x)
output, index = torch.max(input_x, dim=1)
print(output)
# tensor([-0.0081,  1.4201,  0.5722,  0.2790])
print(index)
# tensor([0, 0, 3, 2])

# mindspore.ops.max
input_x = ms.Tensor(np_x)
output, index = ops.max(input_x, axis=1)
print(output)
# [-0.0081  1.4201  0.5722  0.279 ]
print(index)
# [0 0 3 2]

# torch.max(input, other, *, out=None)
torch_x = torch.tensor([1.0, 5.0, 3.0], dtype=torch.float32)
torch_y = torch.tensor([4.0, 2.0, 6.0], dtype=torch.float32)
torch_output = torch.max(torch_x, torch_y)
print(torch_output)
# tensor([4., 5., 6.])

# mindspore.ops.maximum
x = ms.Tensor([1.0, 5.0, 3.0], ms.float32)
y = ms.Tensor([4.0, 2.0, 6.0], ms.float32)
output = ms.ops.maximum(x, y)
print(output)
# [4. 5. 6.]
```
