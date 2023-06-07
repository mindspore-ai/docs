# Function Differences with torchvision.transforms.RandomPosterize

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RandomPosterize.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torchvision.transforms.RandomPosterize

```python
class torchvision.transforms.RandomPosterize(bits, p=0.5)
```

For more information, see [torchvision.transforms.RandomPosterize](https://pytorch.org/vision/0.14/generated/torchvision.transforms.RandomPosterize.html).

## mindspore.dataset.vision.RandomPosterize

```python
class mindspore.dataset.vision.RandomPosterize(bits=(8, 8))
```

For more information, see [mindspore.dataset.vision.RandomPosterize](https://mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.RandomPosterize.html).

## Differences

PyTorch: Posterize the image randomly with a given probability by reducing the number of bits for each color channel, the probability can be set manually.

MindSpore: Posterize the image randomly with a given probability by reducing the number of bits for each color channel, the probability can not be specified.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | bits    | bits    | - |
|     | Parameter2 | p      | -   | Probability of the image being posterized |

## Code Example

```python
from download import download
from PIL import Image

url = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

transform = T.RandomPosterize(bits=(4, 8))
img_torch = T.Compose([transform])(orig_img)


# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

transform = vision.RandomPosterize(bits=(4, 8))
img_ms = transforms.Compose([transform])(orig_img)
```