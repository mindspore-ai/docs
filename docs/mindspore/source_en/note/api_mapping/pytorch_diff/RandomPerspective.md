# Differences with torchvision.transforms.RandomPerspective

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RandomPerspective.md)

## torchvision.transforms.RandomPerspective

```python
class torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=0)
```

For more information, see [torchvision.transforms.RandomPerspective](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.RandomPerspective).

## mindspore.dataset.vision.RandomPerspective

```python
class mindspore.dataset.vision.RandomPerspective(distortion_scale=0.5, prob=0.5, interpolation=Inter.BICUBIC)
```

For more information, see [mindspore.dataset.vision.RandomPerspective](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset_vision/mindspore.dataset.vision.RandomPerspective.html).

## Differences

PyTorch: Performs a random perspective transformation of the given image with a given probability. Pixel fill value for the area outside the transformed image can be specified.

MindSpore: Performs a random perspective transformation of the given image with a given probability. Pixel fill value for the area outside the transformed image will always be black.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | distortion_scale   | distortion_scale   | - |
|     | Parameter2 | p     | prob   |- |
|     | Parameter3 | interpolation     | interpolation    | The default value is different |
|     | Parameter4 | fill    | -   | Pixel fill value for the area outside the transformed image |

## Code Example

```python
from download import download
from PIL import Image

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

transform = T.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=T.InterpolationMode.BILINEAR)
img_torch = T.Compose([transform])(orig_img)


# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

transform = vision.RandomPerspective(distortion_scale=0.5, prob=0.5, interpolation=vision.Inter.BILINEAR)
img_ms = transforms.Compose([transform])(orig_img)
```