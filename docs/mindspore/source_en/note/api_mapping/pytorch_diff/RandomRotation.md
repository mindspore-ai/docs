# Differences with torchvision.transforms.RandomRotation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RandomRotation.md)

## torchvision.transforms.RandomRotation

```python
class torchvision.transforms.RandomRotation(degrees, interpolation=<InterpolationMode.NEAREST: 'nearest'>, expand=False, center=None, fill=0, resample=None)
```

For more information, see [torchvision.transforms.RandomRotation](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.RandomRotation).

## mindspore.dataset.vision.RandomRotation

```python
class mindspore.dataset.vision.RandomRotation(degrees, resample=Inter.NEAREST, expand=False, center=None, fill_value=0)
```

For more information, see [mindspore.dataset.vision.RandomRotation](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset_vision/mindspore.dataset.vision.RandomRotation.html).

## Differences

PyTorch: Rotate the input image randomly within a specified range of degrees.

MindSpore: Rotate the input image randomly within a specified range of degrees.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | degrees  | degrees  | - |
|     | Parameter2 | interpolation    | resample  |- |
|     | Parameter3 | expand    | expand   |- |
|     | Parameter4 | center   | center   | - |
|     | Parameter5 | fill   | fill_value  | - |
|     | Parameter6 | resample   | - | Deprecated in PyTorch, same with interpolation |

## Code Example

```python
from download import download
from PIL import Image

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torch
import torchvision.transforms as T
torch.manual_seed(1)

affine_transfomer = T.RandomRotation(degrees=(30, 70), center=(0, 0))
img_torch = affine_transfomer(orig_img)
print(img_torch.size)
# Out: (471, 292)

# MindSpore
import mindspore as ms
import mindspore.dataset.vision as vision
ms.dataset.config.set_seed(2)

affine_transfomer = vision.RandomRotation(degrees=(30, 70), center=(0, 0))
img_ms = affine_transfomer(orig_img)
print(img_ms.size)
# Out: (471, 292)
```