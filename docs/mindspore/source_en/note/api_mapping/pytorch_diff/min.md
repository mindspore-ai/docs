# Function Differences with torch.min

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_en/note/api_mapping/pytorch_diff/min.md)

## torch.min

```python
torch.min(input, dim, keepdim=False, *, out=None)
```

For more information, see [torch.min](https://pytorch.org/docs/1.8.1/torch.html#torch.min).

## mindspore.ops.min

```python
mindspore.ops.min(input, axis=None, keepdims=False, *, initial=None, where=None)
```

For more information, see [mindspore.ops.min](https://mindspore.cn/docs/en/r1.11/api_python/ops/mindspore.ops.min.html).

## Differences

PyTorch: Output tuple(min, index of min).

MindSpore: When the axis is None or the shape is empty in MindSpore, the keepdims and subsequent parameters are not effective, and the function is consistent with torch.min(input), and the index returned is fixed at 0. Otherwise, the output is a tuple (min, index of min), which is consistent with torch.min(input, dim, keepdim=False, *, out=None).

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
output, index = ops.min(input_x, axis=1)
print(output)
# [-0.7814 -0.3848 -1.1348 -0.6668]
print(index)
# [2 2 2 0]

# torch
input_x = torch.tensor(np_x)
output, index = torch.min(input_x, dim=1)
print(output)
# tensor([-0.7814, -0.3848, -1.1348, -0.6668])
print(index)
# tensor([2, 2, 2, 0])
```
