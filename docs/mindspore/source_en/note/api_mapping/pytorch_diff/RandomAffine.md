# Differences with torchvision.transforms.RandomAffine

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RandomAffine.md)

## torchvision.transforms.RandomAffine

```python
class torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, interpolation=<InterpolationMode.NEAREST: 'nearest'>, fill=0, fillcolor=None, resample=None)
```

For more information, see [torchvision.transforms.RandomAffine](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.RandomAffine).

## mindspore.dataset.vision.RandomAffine

```python
class mindspore.dataset.vision.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=Inter.NEAREST, fill_value=0)
```

For more information, see [mindspore.dataset.vision.RandomAffine](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset_vision/mindspore.dataset.vision.RandomAffine.html).

## Differences

PyTorch: Apply random affine transformation to a tensor image. The rotation center position can be specified.

MindSpore: Apply random affine transformation to the input image. The rotation center is in the center of the image.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | degrees  | degrees  | - |
|     | Parameter2 | translate    | translate  |- |
|     | Parameter3 | scale    | scale   |- |
|     | Parameter4 | shear   | shear   | - |
|     | Parameter5 | interpolation   | resample  | - |
|     | Parameter6 | fill   | fill_value | - |
|     | Parameter7 | fillcolor   | -  | Deprecated in PyTorch, same with fill |
|     | Parameter8 | resample    | -  | Deprecated in PyTorch, same with interpolation |

## Code Example

```python
from download import download
from PIL import Image

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

affine_transfomer = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), fill=0)
img_torch = affine_transfomer(orig_img)

# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

affine_transfomer = vision.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), fill_value=0)
img_ms = affine_transfomer(orig_img)
```