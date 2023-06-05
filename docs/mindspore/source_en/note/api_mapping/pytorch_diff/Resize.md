# Function Differences with torchvision.transforms.Resize

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Resize.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torchvision.transforms.Resize

```python
class torchvision.transforms.Resize(size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None)
```

For more information, see [torchvision.transforms.Resize](https://pytorch.org/vision/0.14/generated/torchvision.transforms.Resize.html).

## mindspore.dataset.vision.Resize

```python
class mindspore.dataset.vision.Resize(size, interpolation=Inter.LINEAR)
```

For more information, see [mindspore.dataset.vision.Resize](https://mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.Resize.html).

## Differences

PyTorch: Resize the input image to the given size, support antialias for BILINEAR、BICUBIC interpolation.

MindSpore: Resize the input image to the given size. Constraint the max size of logger size of image is not supported. Antialias is not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | size    | size    | - |
|     | Parameter2 | interpolation      | interpolation    |- |
|     | Parameter3 | max_size      | -    | The maximum allowed for the longer edge of the resized image |
|     | Parameter4 | antialias    | -   | Antialias flag for BILINEAR、BICUBIC interpolation |

## Code Example

```python
from download import download
from PIL import Image

url = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

transform = T.Resize(size=(200, 200), interpolation=T.InterpolationMode.BILINEAR)
img_torch = T.Compose([transform])(orig_img)
print(img_torch.size)

# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

transform = vision.Resize(size=(200, 200), interpolation=vision.Inter.BILINEAR)
img_ms = transforms.Compose([transform])(orig_img)
print(img_ms[0].size)
```