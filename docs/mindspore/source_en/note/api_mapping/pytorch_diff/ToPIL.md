# Function differences with torchvision.transforms.ToPILImage

<a href="https://gitee.com/mindspore/docs/blob/r1.8/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ToPIL.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torchvision.transforms.ToPILImage

```python
class torchvision.transforms.ToPILImage(
    mode=None
    )
```

For more information, see [torchvision.transforms.ToPILImage](https://pytorch.org/vision/0.10/transforms.html#torchvision.transforms.ToPILImage).

## mindspore.dataset.vision.ToPIL

```python
class mindspore.dataset.vision.ToPIL
```

For more information, see [mindspore.dataset.vision.ToPIL](https://mindspore.cn/docs/en/r1.8/api_python/dataset_vision/mindspore.dataset.vision.ToPIL.html#mindspore.dataset.vision.ToPIL).

## Differences

PyTorch: Converts a tensor or numpy array to PIL Image. The input can be a torch Tensor in the format of <C, H, W>, or a numpy array in the format of <H, W, C>.

MindSpore: The input is a decoded numpy array, which is converted into a PIL type image.

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
