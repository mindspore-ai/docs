# Function Differences with torch.nn.Upsample

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ResizeBicubic.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Upsample

```text
torch.nn.Upsample(
    size=None,
    scale_factor=None,
    mode='nearest',
    align_corners=None
)(input) -> Tensor
```

For more information, see [torch.nn.Upsample](https://pytorch.org/docs/1.8.1/generated/torch.nn.Upsample.html#torch.nn.Upsample).

## mindspore.ops.ResizeBicubic

```text
class mindspore.ops.ResizeBicubic(
    align_corners=False,
    half_pixel_centers=False
)(images, size) -> Tensor
```

For more information, see [mindspore.ops.ResizeBicubic](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ResizeBicubic.html#mindspore.ops.ResizeBicubic).

## Differences

PyTorch: When mode is `bicubic` , Bicubic interpolation is performed on the input image.

MindSpore：There are some differences between MindSpore and PyTorch when the mode is `bicubic` . When align_corners=True and half_pixel_centers=False, MindSpore's implementation is nearly same as PyTorch with align_corners=True. However, when align_corners=False and half_pixel_centers=True, MindSpore has two differences from PyTorch with align_corner=False:

- In MindSpore, the coefficient a used in bicubic interpolation is `-0.5` , which is `-0.75` in PyTorch;

- The weights of the sampling positions outside Tensor are set to 0 and renormalized, so the sum of all weights is 1.0, and PyTorch changes the sampling outside Tensor to sample the boundary positions, and the weights remain unchanged, which results that the boundary positions will be given greater weight.

Therefore, in the second mode, there is a difference between the calculation results of MindSpore and PyTorch.

| Categories | Subcategories  | PyTorch   | MindSpore | Differences                                   |
| ---- | ----- | --------- | --------- | ------------------------------------------------------------ |
| Parameters | Parameter 1 | size  |   -   |         -                                                   |
|      | Parameter 2 | scale_factor |   -    | Multiplier for spatial size. MindSpore does not have this feature |
|      | Parameter 3 | mode    |    -        | The upsampling algorithm. Bicubic interpolation will be applied when mode is `bicubic`  |
|      | Parameter 4 | align_corners | align_corners   |         -                                             |
|      | Parameter 5 |   -    | half_pixel_centers | In MindSpore, it's the flag of half-pixel center alignment, while PyTorch implementation with align_corners=False uses half-pixel center alignment |
| Inputs | Input 1 | input      | images    |                -                                           |
|       | Input 2 | -          | size      |                 -                                          |

### Code Example 1

```python
# PyTorch
import numpy as np
import torch

x_np = np.array([1, 2, 3, 4]).astype(np.float32).reshape(1, 1, 2, 2)
size = (1, 4)
upsample = torch.nn.Upsample(size=size, mode='bicubic', align_corners=True)
out = upsample(torch.from_numpy(x_np))
print(out.detach().numpy())
# [[[[1., 1.3148143, 1.6851856, 2.]]]]

# MindSpore
import mindspore
from mindspore import Tensor
from mindspore.ops as ops

resize_bicubic_op = ops.ResizeBicubic(align_corners=True,half_pixel_centers=False)
images = Tensor(x_np)
size = Tensor(size, mindspore.int32)
output = resize_bicubic_op(images, size)
print(output.asnumpy())
# [[[[1., 1.3144622, 1.6855378, 2.]]]]
```

### Code Example 2

```python
# PyTorch
import numpy as np
import torch

x_np = np.array([1, 2, 3, 4]).astype(np.float32).reshape(1, 1, 2, 2)
size = (1, 4)
upsample = torch.nn.Upsample(size=size, mode='bicubic', align_corners=False)
out = upsample(torch.from_numpy(x_np))
print(out.detach().numpy())
# [[[[0.70703125, 1.0390625, 1.5859375, 1.9179688]]]]

# MindSpore
import mindspore
from mindspore import Tensor
from mindspore.ops as ops

resize_bicubic_op = ops.ResizeBicubic(align_corners=False, half_pixel_centers=True)
images = Tensor(x_np)
size = Tensor(size, mindspore.int32)
output = resize_bicubic_op(images, size)
print(output.asnumpy())
# [[[[1.9117649, 2.2071428, 2.7928572, 3.0882356]]]]
```