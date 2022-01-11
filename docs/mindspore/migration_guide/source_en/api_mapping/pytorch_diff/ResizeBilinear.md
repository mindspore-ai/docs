# Function Differences with torch.nn.Upsample

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/ResizeBilinear.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## torch.nn.Upsample

```python
torch.nn.Upsample(
    size=None,
    scale_factor=None,
    mode='nearest',
    align_corners=None
)(input)
```

For more information, see [torch.nn.Upsample](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Upsample).

## mindspore.nn.ResizeBilinear

```python
class mindspore.nn.ResizeBilinear()(x, size=None, scale_factor=None, align_corners=False)
```

For more information, see [mindspore.nn.ResizeBilinear](https://mindspore.cn/docs/api/en/r1.6/api_python/nn/mindspore.nn.ResizeBilinear.html#mindspore.nn.ResizeBilinear).

## Differences

PyTorch: Multiple modes can be chosen when upsampling data.

MindSpore：Only supports `bilinear` mode to sample data.

## Code Example

```python
from mindspore import Tensor
import mindspore.nn as nn
import torch
import numpy as np

# In MindSpore, it is predetermined to use bilinear to resize the input image.
x = np.random.randn(1, 2, 3, 4).astype(np.float32)
resize = nn.ResizeBilinear()
tensor = Tensor(x)
output = resize(tensor, (5, 5))
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