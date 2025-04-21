# Differences with torchvision.transforms.ToPILImage

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ToPIL.md)

## torchvision.transforms.ToPILImage

```python
class torchvision.transforms.ToPILImage(
    mode=None
    )
```

For more information, see [torchvision.transforms.ToPILImage](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.ToPILImage).

## mindspore.dataset.vision.ToPIL

```python
class mindspore.dataset.vision.ToPIL
```

For more information, see [mindspore.dataset.vision.ToPIL](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset_vision/mindspore.dataset.vision.ToPIL.html#mindspore.dataset.vision.ToPIL).

## Differences

PyTorch: Converts a tensor or Numpy array to PIL Image. The input can be a torch Tensor in the format of <C, H, W>, or a numpy array in the format of <H, W, C>.

MindSpore: Convert a Numpy array in <H, W, C> format (such as decoded image) into a PIL image, color space is not support to specified.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | mode    | -    | Color space of input data |

## Code Example

```python
import numpy as np
import torch as T
from torchvision.transforms import ToPILImage
import mindspore.dataset.vision as vision

# In MindSpore, ToPIL transform the numpy.ndarray to PIL Image.

image = np.random.random((64,64))
img = vision.ToPIL()(image)
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
