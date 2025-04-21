# Differences with torchvision.transforms.Normalize

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Normalize.md)

## torchvision.transforms.Normalize

```python
class torchvision.transforms.Normalize(mean, std, inplace=False)
```

For more information, see [torchvision.transforms.Normalize](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.Normalize).

## mindspore.dataset.vision.Normalize

```python
class mindspore.dataset.vision.Normalize(mean, std, is_hwc=True)
```

For more information, see [mindspore.dataset.vision.Normalize](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset_vision/mindspore.dataset.vision.Normalize.html).

## Differences

PyTorch: Normalize the input image based on the mean and standard deviation, specified format is not supported.

MindSpore: Normalize the input image based on the mean and standard deviation, in-place option is not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | mean   | mean     | - |
|     | Parameter2 | std    |std   | - |
|     | Parameter3 | inplace | -   | Whether to make this operation in-place |
|     | Parameter4 | -   | is_hwc    | Whether the input image is HWC or CHW |

## Code Example

```python
from download import download
from PIL import Image

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

normalize = T.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
to_tensor = T.ToTensor()
img_torch = T.Compose([to_tensor, normalize])((orig_img))
print(img_torch.shape)
# Torch tensor is in C,H,W format
# Out: torch.Size([3, 292, 471])

# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

normalize = vision.Normalize(mean=[0, 0, 0], std=[1, 1, 1], is_hwc=False)
to_tensor = vision.ToTensor()
img_ms = transforms.Compose([to_tensor, normalize])((orig_img))
print(img_ms[0].shape)
# vision.ToTensor change the format from HWC to CHW, so normalize have to specify `is_hwc=False`
# Out: (3, 292, 471)
```