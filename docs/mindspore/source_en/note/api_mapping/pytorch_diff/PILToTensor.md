# Function Differences with torchvision.transforms.PILToTensor

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/PILToTensor.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torchvision.transforms.PILToTensor

```python
class torchvision.transforms.PILToTensor
```

For more information, see [torchvision.transforms.PILToTensor](https://pytorch.org/vision/0.14/generated/torchvision.transforms.PILToTensor.html).

## mindspore.dataset.vision.ToNumpy

```python
class mindspore.dataset.vision.ToNumpy
```

For more information, see [mindspore.dataset.vision.ToNumpy](https://mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.ToNumpy.html).

## Differences

PyTorch: Convert a PIL Image in <H, W, C> format to a tensor of the same type in <C, H, W> format.

MindSpore: Convert a PIL Image in <H, W, C> format to a Numpy of the same type in <H, W, C> format.

## Code Example

```python
import numpy as np
from PIL import Image
from torchvision import transforms
import mindspore.dataset.vision as vision

from download import download
from PIL import Image

url = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# In torch, PILToTensor transforms the input to tensor.
image_transform = transforms.PILToTensor()
img = Image.open('flamingos.jpg')
img_data = image_transform(img)
print(img_data.shape)
# Out:
# torch.Size([3, 292, 471])

# In MindSpore, ToNumpy convert PIL Image into numpy array.
img = Image.open('flamingos.jpg')
to_numpy = vision.ToNumpy()
img_data = to_numpy(img)
print(img_data.shape)
# Out:
# (292, 471, 3)
```
