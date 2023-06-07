# Function Differences with torchvision.transforms.RandomSolarize

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RandomSolarize.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torchvision.transforms.RandomSolarize

```python
class torchvision.transforms.RandomSolarize(
    threshold,
    p=0.5
    )
```

For more information, see [torchvision.transforms.RandomSolarize](https://pytorch.org/vision/0.14/generated/torchvision.transforms.RandomSolarize).

## mindspore.dataset.vision.RandomSolarize

```python
class mindspore.dataset.vision.RandomSolarize(
    threshold=(0, 255)
    )
```

For more information, see [mindspore.dataset.vision.RandomSolarize](https://mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.RandomSolarize.html#mindspore.dataset.vision.RandomSolarize).

## Differences

PyTorch: Randomly expose the image with a given probability by inverting all pixel values above the threshold.

MindSpore: Select a random subrange from the specified threshold range and adjust the image pixel values within the subrange.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | threshold   | threshold   | - |
|     | Parameter2 | p      | -   | Probability of the image being solarized |

## Code Example

```python
from download import download
from PIL import Image

url = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

transform = T.RandomSolarize(threshold=128, p=1)
img_torch = T.Compose([transform])(orig_img)


# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

transform = vision.RandomSolarize(threshold=(128, 128))
img_ms = transforms.Compose([transform])(orig_img)


```