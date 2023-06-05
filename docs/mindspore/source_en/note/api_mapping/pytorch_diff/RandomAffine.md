# Function Differences with torchvision.transforms.RandomAffine

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RandomAffine.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torchvision.transforms.RandomAffine

```python
class torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, interpolation=InterpolationMode.NEAREST, fill=0, center=None)
```

For more information, see [torchvision.transforms.RandomAffine](https://pytorch.org/vision/0.14/generated/torchvision.transforms.RandomAffine.html).

## mindspore.dataset.vision.RandomAffine

```python
class mindspore.dataset.vision.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=Inter.NEAREST, fill_value=0)
```

For more information, see [mindspore.dataset.vision.RandomAffine](https://mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.RandomAffine.html).

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
|     | Parameter7 | center   | -  | Specify the center of rotation |

## Code Example

```python
from download import download
from PIL import Image

url = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

affine_transfomer = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), center=(0, 0))
img_torch = affine_transfomer(orig_img)

# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

# If need to specify center of rotation, use RandomRotation + RandomAffine
rotation_transfomer = vision.RandomRotation(degrees=(30, 70), center=(0, 0))
affine_transfomer = vision.RandomAffine(degrees=(0, 0), translate=(0.1, 0.3))
transformer = transforms.Compose([rotation_transfomer, affine_transfomer])
img_ms = transformer(orig_img)
```