# Function Differences with mindspore.ops.ResizeBilinear

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/ResizeBilinear.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Upsample

```python
torch.nn.Upsample(
    input,
    size,
    scale_factor,
    mode='nearest',
    align_corners=None
)
```

For more information, see[torch.nn.Upsample](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Upsample).

## mindspore.ops.ResizeBilinear

```python
class mindspore.ops.ResizeBilinear(
    size,
    align_corners=False
)(input)
```

For more information, see[mindspore.ops.ResizeBilinear](https://mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.ResizeBilinear.html#mindspore.ops.ResizeBilinear).

## Differences

PyTorch: Multiple choice of modes to resize the image.

MindSpore：Only the mode of `bilinear` is supported.

## Code Example

```python
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, it is predetermined to use bilinear to resize the input image.
x = np.random.randn(1, 2, 3, 4).astype(np.float32)
resize = ops.ResizeBilinear((5, 5))
tensor = Tensor(x)
output = resize(tensor)
print(output.shape)
# Out：
# (1, 2, 5, 5)

# In torch, parameter mode should be passed to determine which method to apply for resizing input image.
x = np.random.randn(1, 2, 3, 4).astype(np.float32)
resize = torch.nn.Upsample(size=(5, 5), mode='bilinear')
tensor = torch.tensor(x)
output = resize(tensor)
print(output.shape)
# Out：
# torch.Size([1, 2, 5, 5])
```