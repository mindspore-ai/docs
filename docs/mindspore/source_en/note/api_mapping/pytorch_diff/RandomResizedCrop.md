# Differences with torchvision.transforms.RandomResizedCrop

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RandomResizedCrop.md)

## torchvision.transforms.RandomResizedCrop

```python
class torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=InterpolationMode.BILINEAR)
```

For more information, see [torchvision.transforms.RandomResizedCrop](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.RandomResizedCrop).

## mindspore.dataset.vision.RandomResizedCrop

```python
class mindspore.dataset.vision.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Inter.BILINEAR, max_attempts=10)
```

For more information, see [mindspore.dataset.vision.RandomResizedCrop](https://mindspore.cn/docs/en/br_base/api_python/dataset_vision/mindspore.dataset.vision.RandomResizedCrop.html).

## Differences

PyTorch: Crop a random portion of image and resize it to a given size.

MindSpore: Crop a random portion of image and resize it to a given size.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | size    | size    | - |
|     | Parameter2 | scale      | scale   |- |
|     | Parameter3 | ratio     | ratio    | - |
|     | Parameter4 | interpolation     | interpolation   | - |
|     | Parameter5 | -     | max_attempts   | The maximum number of attempts to propose a valid crop_area |

## Code Example

```python
from download import download
from PIL import Image

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

op = T.RandomResizedCrop((32, 32), scale=(0.08, 1.0), ratio=(0.75, 0.8))
img_torch =op(orig_img)
print(img_torch.size)
# Out: (32, 32)

# MindSpore
import mindspore.dataset.vision as vision

op = vision.RandomResizedCrop((32, 32), scale=(0.08, 1.0), ratio=(0.75, 0.8))
img_ms = op(orig_img)
print(img_ms.size)
# Out: (32, 32)
```