﻿# Function differences with torchvision.transforms.ToPILImage

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/ToPIL.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## torchvision.transforms.ToPILImage

```python
class torchvision.transforms.ToPILImage(
    mode=None
    )
```

## mindspore.dataset.vision.py_transforms.ToPIL

```python
class mindspore.dataset.vision.py_transforms.ToPIL
```

## Differences

PyTorch: Converts a tensor or numpy array to PIL Image. The input can be a torch Tensor in the format of <C, H, W>, or a numpy array in the format of <H, W, C>.

MindSpore: The input is a decoded numpy array, which is converted into a PIL type image.

## Code Example

```python
import numpy as np
import torch as T
from torchvision.transforms import ToPILImage
import mindspore.dataset.vision.py_transforms as py_vision

# In MindSpore, ToPIL transform the numpy.ndarray to PIL Image.

image = np.random.random((64,64))
img = py_vision.ToPIL()(image)
img.show()
# Out:
# window of PIL image

# In torch, ToPILImage transforms the input to PIL Image.
image = T.randn((64, 64))
img = ToPILImage()(image)
img.show()
# Out:
# window of PIL image
```
