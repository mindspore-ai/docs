# Function Differences with torchvision.transforms.RandomSolarize

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RandomSolarize.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

## torchvision.transforms.RandomSolarize

```python
class torchvision.transforms.RandomSolarize(
    threshold,
    p=0.5
    )
```

For more information, see [torchvision.transforms.RandomSolarize](https://pytorch.org/vision/0.10/transforms.html#torchvision.transforms.RandomSolarize).

## mindspore.dataset.vision.RandomSolarize

```python
class mindspore.dataset.vision.RandomSolarize(
    threshold=(0, 255)
    )
```

For more information, see [mindspore.dataset.vision.RandomSolarize](https://mindspore.cn/docs/en/r2.0/api_python/dataset_vision/mindspore.dataset.vision.RandomSolarize.html#mindspore.dataset.vision.RandomSolarize).

## Differences

PyTorch: Randomly expose the image with a given probability by inverting all pixel values above the threshold.

MindSpore: Select a random subrange from the specified threshold range and adjust the image pixel values within the subrange.

## Code Example

```python
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import mindspore.dataset.vision as vision

orig_img = Image.open(Path('.') / 'test.jpg')

def show_diff_image(image_original, image_transformed):

    num = 2

    plt.subplot(1, num, 1)
    plt.imshow(image_original)
    plt.title("Original image")

    plt.subplot(1, num, 2)
    plt.imshow(image_transformed)
    plt.title("Random Solaried image")

    plt.show()


# In MindSpore, randomly selects a subrange within the specified threshold range and sets the pixel value within the subrange to (255 - pixel).

solarizer  = vision.RandomSolarize(threshold=(10,100))
rand_sola_img = solarizer(orig_img)
show_diff_image(orig_img, rand_sola_img)

# Out:
# Original image and Solarized image are showed with matplotlib tools


# In torch, the RandomSolarize transform randomly solarizes the image by inverting all pixel values above the threshold.

solarizer = T.RandomSolarize(threshold=192.0)
solarized_imgs = solarizer(orig_img)
show_diff_image(orig_img, solarized_imgs)

# Out:
# Original image and Solarized image are showed with matplotlib tools
```