# Differences with torchvision.transforms.ConvertImageDtype

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TypeCast.md)

## torchvision.transforms.ConvertImageDtype

```python
class torchvision.transforms.ConvertImageDtype(
    dtype: torch.dtype
    )
```

For more information, see [torchvision.transforms.ConvertImageDtype](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.ConvertImageDtype).

## mindspore.dataset.transforms.TypeCast

```python
class mindspore.dataset.transforms.TypeCast(
    output_type
    )
```

For more information, see [mindspore.dataset.transforms.TypeCast](https://mindspore.cn/docs/en/br_base/api_python/dataset_transforms/mindspore.dataset.transforms.TypeCast.html#mindspore.dataset.transforms.TypeCast).

## Differences

PyTorch: Convert a tensor image to the given dtype and scale the values accordingly. This function does not support PIL Image.

MindSpore: Convert the input numpy.ndarray image to the desired dtype.

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

to_tensor = T.ToTensor()
convert = T.ConvertImageDtype(torch.float)
img_torch = T.Compose([to_tensor, convert])((orig_img))
print(img_torch.dtype)
# Out: torch.float32

# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

to_tensor = vision.ToTensor()
convert = transforms.TypeCast("float32")
img_ms = transforms.Compose([to_tensor, convert])((orig_img))
print(img_ms[0].dtype)
# Out: float32
```