# Function Differences with torchvision.transforms.RandomErasing

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RandomErasing.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torchvision.transforms.RandomErasing

```python
class torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
```

For more information, see [torchvision.transforms.RandomErasing](https://pytorch.org/vision/0.14/generated/torchvision.transforms.RandomErasing.html).

## mindspore.dataset.vision.RandomErasing

```python
class mindspore.dataset.vision.RandomErasing(prob=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False, max_attempts=10)
```

For more information, see [mindspore.dataset.vision.RandomErasing](https://mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.RandomErasing.html).

## Differences

PyTorch: Randomly selects a rectangle region in an torch Tensor image and erases its pixels.

MindSpore: Randomly selects a rectangle region in an torch Tensor image and erases its pixels.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | p  | prob     | - |
|     | Parameter2 | scale      | scale    | - |
|     | Parameter3 | ratio      | ratio   | - |
|     | Parameter4 | value      | value   | - |
|     | Parameter5 | inplace      | inplace   | - |
|     | Parameter6 | -     | max_attempts    | The maximum number of attempts to propose a valid erased area, beyond which the original image will be returned |

## Code Example

```python
from download import download
from PIL import Image

url = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

to_tensor = T.ToTensor()
transform = T.RandomErasing(p=0.5, scale=(0.02, 0.33))
img_torch = T.Compose([to_tensor, transform])(orig_img)


# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

to_tensor = vision.ToTensor()
transform = vision.RandomErasing(prob=0.5, scale=(0.02, 0.33))
img_ms = transforms.Compose([to_tensor, transform])(orig_img)
```