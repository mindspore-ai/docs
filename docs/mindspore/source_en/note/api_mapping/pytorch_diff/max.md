# Differences with torch.max

<a href="https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/max.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png"></a>

## torch.max

```python
torch.max(input, dim, keepdim=False, *, out=None)
```

For more information, see [torch.max](https://pytorch.org/docs/1.8.1/torch.html#torch.max).

## mindspore.ops.max

```python
mindspore.ops.max(input, axis=None, keepdims=False, *, initial=None, where=None)
```

For more information, see [mindspore.ops.max](https://mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.max.html).

## Differences

PyTorch: Output tuple(max, index of max).

MindSpore: When the axis is None or the shape is empty in MindSpore, the keepdims and subsequent parameters are not effective, and the function is consistent with torch.max(input), and the index returned is fixed at 0. Otherwise, the output is a tuple (max, index of max), which is consistent with torch.max(input, dim, keepdim=False, *, out=None).

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
|Parameters | Parameter 1 | input        | input       | Consistent |
|      | Parameter 2 | dim       | axis      | Same function, different parameter names |
| | Parameter 3 | keepdim    | keepdims     | Same function, different parameter names       |
| | Parameter 4 | -      |initial    | Not involved        |
| | Parameter 5 |  -     |where    | Not involved        |
| | Parameter 6 | out    | -         | Not involved |

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
# mindspore
input_x = ms.Tensor(np_x)
output, index = ops.max(input_x, axis=1)
print(output)
# [-0.0081  1.4201  0.5722  0.279 ]
print(index)
# [0 0 3 2]

# torch
input_x = torch.tensor(np_x)
output, index = torch.max(input_x, dim=1)
print(output)
# tensor([-0.0081,  1.4201,  0.5722,  0.2790])
print(index)
# tensor([0, 0, 3, 2])
```
